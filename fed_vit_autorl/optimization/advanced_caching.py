"""Advanced multi-level caching system for federated learning."""

import asyncio
import time
import logging
import threading
import hashlib
import pickle
import gzip
import os
import tempfile
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import OrderedDict, defaultdict
import torch
import json
from concurrent.futures import ThreadPoolExecutor
import weakref

from ..error_handling import with_error_handling, handle_error, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels in the hierarchy."""
    L1_MEMORY = "l1_memory"      # In-memory cache (fastest)
    L2_COMPRESSED = "l2_compressed"  # Compressed in-memory cache
    L3_DISK = "l3_disk"          # Disk-based cache
    L4_DISTRIBUTED = "l4_distributed"  # Distributed cache across nodes


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    TTL = "ttl"              # Time To Live
    SIZE_BASED = "size_based"  # Based on data size
    IMPORTANCE = "importance"  # Based on importance score
    ADAPTIVE = "adaptive"     # Adaptive policy


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    data: Any
    size_bytes: int
    created_time: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    importance_score: float = 1.0
    ttl: Optional[float] = None
    compression_ratio: float = 1.0
    cache_level: CacheLevel = CacheLevel.L1_MEMORY
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_time > self.ttl
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def calculate_priority(self, policy: CachePolicy) -> float:
        """Calculate priority for eviction."""
        now = time.time()
        
        if policy == CachePolicy.LRU:
            return now - self.last_accessed
        elif policy == CachePolicy.LFU:
            return -self.access_count
        elif policy == CachePolicy.TTL:
            return self.ttl - (now - self.created_time) if self.ttl else float('inf')
        elif policy == CachePolicy.SIZE_BASED:
            return self.size_bytes
        elif policy == CachePolicy.IMPORTANCE:
            return -self.importance_score
        else:  # ADAPTIVE
            # Combine multiple factors
            recency = 1.0 / (1.0 + now - self.last_accessed)
            frequency = np.log(1.0 + self.access_count)
            importance = self.importance_score
            size_penalty = np.log(1.0 + self.size_bytes / 1024 / 1024)  # MB
            
            return -(recency * frequency * importance / size_penalty)


class CompressionManager:
    """Manages compression for cache entries."""
    
    @staticmethod
    def compress_data(data: Any, compression_level: int = 6) -> Tuple[bytes, float]:
        """Compress data and return compressed bytes with ratio."""
        try:
            # Serialize data
            serialized = pickle.dumps(data)
            original_size = len(serialized)
            
            # Compress
            compressed = gzip.compress(serialized, compresslevel=compression_level)
            compressed_size = len(compressed)
            
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            return compressed, compression_ratio
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            # Fallback to uncompressed
            return pickle.dumps(data), 1.0
    
    @staticmethod
    def decompress_data(compressed_data: bytes) -> Any:
        """Decompress data."""
        try:
            decompressed = gzip.decompress(compressed_data)
            return pickle.loads(decompressed)
        except Exception as e:
            # Try direct unpickling (for uncompressed data)
            try:
                return pickle.loads(compressed_data)
            except Exception:
                logger.error(f"Decompression failed: {e}")
                raise


class CacheLevel1(OrderedDict):
    """L1 Memory Cache - Fastest access, limited size."""
    
    def __init__(self, max_size_mb: int = 512, policy: CachePolicy = CachePolicy.ADAPTIVE):
        super().__init__()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.policy = policy
        self.access_stats = defaultdict(int)
        self._lock = threading.RLock()
        
    @with_error_handling(max_retries=1, auto_recover=True)
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from L1 cache."""
        with self._lock:
            if key in self:
                entry = super().__getitem__(key)
                if entry.is_expired():
                    self._remove_entry(key)
                    return None
                
                entry.update_access()
                self.move_to_end(key)  # LRU behavior
                return entry
            return None
    
    @with_error_handling(max_retries=1, auto_recover=True)
    def put(self, key: str, entry: CacheEntry) -> bool:
        """Put item in L1 cache."""
        with self._lock:
            # Remove existing entry if present
            if key in self:
                self._remove_entry(key)
            
            # Check if we have space
            if self.current_size_bytes + entry.size_bytes > self.max_size_bytes:
                if not self._make_space(entry.size_bytes):
                    logger.warning(f"Cannot fit entry {key} in L1 cache")
                    return False
            
            # Add entry
            self[key] = entry
            self.current_size_bytes += entry.size_bytes
            entry.cache_level = CacheLevel.L1_MEMORY
            
            return True
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update size."""
        if key in self:
            entry = self[key]
            self.current_size_bytes -= entry.size_bytes
            del self[key]
    
    def _make_space(self, required_bytes: int) -> bool:
        """Make space by evicting entries."""
        if required_bytes > self.max_size_bytes:
            return False
        
        # Calculate priorities for eviction
        entries_with_priority = [
            (key, entry.calculate_priority(self.policy))
            for key, entry in self.items()
        ]
        
        # Sort by priority (higher priority = more likely to evict)
        entries_with_priority.sort(key=lambda x: x[1], reverse=True)
        
        # Evict entries until we have enough space
        space_freed = 0
        for key, priority in entries_with_priority:
            if self.current_size_bytes - space_freed + required_bytes <= self.max_size_bytes:
                break
            
            entry = self[key]
            space_freed += entry.size_bytes
            self._remove_entry(key)
            logger.debug(f"Evicted {key} from L1 cache (priority: {priority:.3f})")
        
        return self.current_size_bytes + required_bytes <= self.max_size_bytes
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'entries': len(self),
                'size_mb': self.current_size_bytes / 1024 / 1024,
                'utilization': self.current_size_bytes / self.max_size_bytes,
                'policy': self.policy.value,
            }


class CacheLevel2:
    """L2 Compressed Cache - Larger capacity with compression."""
    
    def __init__(self, max_size_mb: int = 2048, compression_level: int = 6):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.compression_level = compression_level
        self.entries: Dict[str, bytes] = {}
        self.metadata: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
    @with_error_handling(max_retries=1, auto_recover=True)
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from L2 cache."""
        with self._lock:
            if key not in self.entries:
                return None
            
            entry_meta = self.metadata[key]
            if entry_meta.is_expired():
                self._remove_entry(key)
                return None
            
            # Decompress data
            try:
                compressed_data = self.entries[key]
                decompressed_data = CompressionManager.decompress_data(compressed_data)
                
                # Create new entry with decompressed data
                entry = CacheEntry(
                    key=key,
                    data=decompressed_data,
                    size_bytes=entry_meta.size_bytes,
                    created_time=entry_meta.created_time,
                    last_accessed=time.time(),
                    access_count=entry_meta.access_count + 1,
                    importance_score=entry_meta.importance_score,
                    ttl=entry_meta.ttl,
                    compression_ratio=entry_meta.compression_ratio,
                    cache_level=CacheLevel.L2_COMPRESSED
                )
                
                # Update metadata
                self.metadata[key] = entry
                
                return entry
                
            except Exception as e:
                logger.error(f"Failed to decompress data for key {key}: {e}")
                self._remove_entry(key)
                return None
    
    @with_error_handling(max_retries=1, auto_recover=True)
    def put(self, key: str, entry: CacheEntry) -> bool:
        """Put item in L2 cache with compression."""
        with self._lock:
            try:
                # Compress data
                compressed_data, compression_ratio = CompressionManager.compress_data(
                    entry.data, self.compression_level
                )
                compressed_size = len(compressed_data)
                
                # Remove existing entry if present
                if key in self.entries:
                    self._remove_entry(key)
                
                # Check if we have space
                if self.current_size_bytes + compressed_size > self.max_size_bytes:
                    if not self._make_space(compressed_size):
                        return False
                
                # Store compressed data and metadata
                self.entries[key] = compressed_data
                
                # Update entry metadata
                entry.compression_ratio = compression_ratio
                entry.cache_level = CacheLevel.L2_COMPRESSED
                self.metadata[key] = entry
                
                self.current_size_bytes += compressed_size
                
                logger.debug(f"Stored {key} in L2 cache (compression: {compression_ratio:.2f}x)")
                return True
                
            except Exception as e:
                logger.error(f"Failed to store in L2 cache: {e}")
                return False
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update size."""
        if key in self.entries:
            compressed_size = len(self.entries[key])
            self.current_size_bytes -= compressed_size
            del self.entries[key]
            del self.metadata[key]
    
    def _make_space(self, required_bytes: int) -> bool:
        """Make space by evicting entries."""
        if required_bytes > self.max_size_bytes:
            return False
        
        # Calculate priorities for eviction
        entries_with_priority = [
            (key, meta.calculate_priority(CachePolicy.ADAPTIVE))
            for key, meta in self.metadata.items()
        ]
        
        # Sort by priority
        entries_with_priority.sort(key=lambda x: x[1], reverse=True)
        
        # Evict entries
        space_freed = 0
        for key, priority in entries_with_priority:
            if self.current_size_bytes - space_freed + required_bytes <= self.max_size_bytes:
                break
            
            compressed_size = len(self.entries[key])
            space_freed += compressed_size
            self._remove_entry(key)
            logger.debug(f"Evicted {key} from L2 cache")
        
        return self.current_size_bytes + required_bytes <= self.max_size_bytes
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            if not self.metadata:
                avg_compression = 1.0
            else:
                avg_compression = np.mean([
                    meta.compression_ratio for meta in self.metadata.values()
                ])
            
            return {
                'entries': len(self.entries),
                'size_mb': self.current_size_bytes / 1024 / 1024,
                'utilization': self.current_size_bytes / self.max_size_bytes,
                'avg_compression_ratio': avg_compression,
            }


class CacheLevel3:
    """L3 Disk Cache - Persistent storage with larger capacity."""
    
    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: int = 10240):
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "fed_vit_cache")
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.metadata_file = os.path.join(self.cache_dir, "metadata.json")
        self.metadata: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        
        # Initialize cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                
                # Calculate current size
                self.current_size_bytes = 0
                for key, meta in self.metadata.items():
                    file_path = self._get_file_path(key)
                    if os.path.exists(file_path):
                        self.current_size_bytes += os.path.getsize(file_path)
                    else:
                        # Remove stale metadata
                        del self.metadata[key]
        except Exception as e:
            logger.error(f"Failed to load L3 cache metadata: {e}")
            self.metadata = {}
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Failed to save L3 cache metadata: {e}")
    
    def _get_file_path(self, key: str) -> str:
        """Get file path for cache key."""
        # Use hash to avoid filesystem issues with long keys
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    @with_error_handling(max_retries=1, auto_recover=True)
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from L3 cache."""
        with self._lock:
            if key not in self.metadata:
                return None
            
            meta = self.metadata[key]
            
            # Check expiration
            if meta.get('ttl') and time.time() - meta['created_time'] > meta['ttl']:
                self._remove_entry(key)
                return None
            
            file_path = self._get_file_path(key)
            
            try:
                with open(file_path, 'rb') as f:
                    compressed_data = f.read()
                
                # Decompress data
                data = CompressionManager.decompress_data(compressed_data)
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    data=data,
                    size_bytes=meta['size_bytes'],
                    created_time=meta['created_time'],
                    last_accessed=time.time(),
                    access_count=meta.get('access_count', 0) + 1,
                    importance_score=meta.get('importance_score', 1.0),
                    ttl=meta.get('ttl'),
                    compression_ratio=meta.get('compression_ratio', 1.0),
                    cache_level=CacheLevel.L3_DISK
                )
                
                # Update metadata
                meta['last_accessed'] = entry.last_accessed
                meta['access_count'] = entry.access_count
                self._save_metadata()
                
                return entry
                
            except Exception as e:
                logger.error(f"Failed to read from L3 cache key {key}: {e}")
                self._remove_entry(key)
                return None
    
    @with_error_handling(max_retries=1, auto_recover=True)
    def put(self, key: str, entry: CacheEntry) -> bool:
        """Put item in L3 cache."""
        with self._lock:
            try:
                # Compress data
                compressed_data, compression_ratio = CompressionManager.compress_data(entry.data)
                compressed_size = len(compressed_data)
                
                # Remove existing entry if present
                if key in self.metadata:
                    self._remove_entry(key)
                
                # Check if we have space
                if self.current_size_bytes + compressed_size > self.max_size_bytes:
                    if not self._make_space(compressed_size):
                        return False
                
                # Write to disk
                file_path = self._get_file_path(key)
                with open(file_path, 'wb') as f:
                    f.write(compressed_data)
                
                # Update metadata
                self.metadata[key] = {
                    'size_bytes': entry.size_bytes,
                    'created_time': entry.created_time,
                    'last_accessed': entry.last_accessed,
                    'access_count': entry.access_count,
                    'importance_score': entry.importance_score,
                    'ttl': entry.ttl,
                    'compression_ratio': compression_ratio,
                    'file_size': compressed_size,
                }
                
                self.current_size_bytes += compressed_size
                self._save_metadata()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to store in L3 cache: {e}")
                return False
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from disk and metadata."""
        if key in self.metadata:
            file_path = self._get_file_path(key)
            
            try:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    self.current_size_bytes -= file_size
            except Exception as e:
                logger.error(f"Failed to remove cache file {file_path}: {e}")
            
            del self.metadata[key]
    
    def _make_space(self, required_bytes: int) -> bool:
        """Make space by evicting entries."""
        if required_bytes > self.max_size_bytes:
            return False
        
        # Calculate priorities for eviction
        entries_with_priority = []
        for key, meta in self.metadata.items():
            # Create temporary entry for priority calculation
            temp_entry = CacheEntry(
                key=key,
                data=None,  # Don't need actual data for priority
                size_bytes=meta['size_bytes'],
                created_time=meta['created_time'],
                last_accessed=meta.get('last_accessed', meta['created_time']),
                access_count=meta.get('access_count', 0),
                importance_score=meta.get('importance_score', 1.0),
                ttl=meta.get('ttl')
            )
            priority = temp_entry.calculate_priority(CachePolicy.ADAPTIVE)
            entries_with_priority.append((key, priority))
        
        # Sort by priority
        entries_with_priority.sort(key=lambda x: x[1], reverse=True)
        
        # Evict entries
        space_freed = 0
        for key, priority in entries_with_priority:
            if self.current_size_bytes - space_freed + required_bytes <= self.max_size_bytes:
                break
            
            file_size = self.metadata[key].get('file_size', 0)
            space_freed += file_size
            self._remove_entry(key)
            logger.debug(f"Evicted {key} from L3 cache")
        
        self._save_metadata()
        return self.current_size_bytes + required_bytes <= self.max_size_bytes
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'entries': len(self.metadata),
                'size_mb': self.current_size_bytes / 1024 / 1024,
                'utilization': self.current_size_bytes / self.max_size_bytes,
                'cache_dir': self.cache_dir,
            }
    
    def cleanup(self) -> None:
        """Cleanup expired entries."""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, meta in self.metadata.items():
                if meta.get('ttl') and current_time - meta['created_time'] > meta['ttl']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                self._save_metadata()
                logger.info(f"Cleaned up {len(expired_keys)} expired entries from L3 cache")


class AdvancedMultiLevelCache:
    """Advanced multi-level caching system for federated learning."""
    
    def __init__(
        self,
        l1_size_mb: int = 512,
        l2_size_mb: int = 2048,
        l3_size_mb: int = 10240,
        cache_dir: Optional[str] = None,
        enable_l3: bool = True,
        auto_promote: bool = True,
        cleanup_interval: float = 3600.0,  # 1 hour
    ):
        """Initialize multi-level cache.
        
        Args:
            l1_size_mb: L1 cache size in MB
            l2_size_mb: L2 cache size in MB
            l3_size_mb: L3 cache size in MB
            cache_dir: Directory for L3 cache
            enable_l3: Whether to enable L3 disk cache
            auto_promote: Whether to auto-promote frequently accessed items
            cleanup_interval: Interval for cleanup operations
        """
        self.l1_cache = CacheLevel1(max_size_mb=l1_size_mb)
        self.l2_cache = CacheLevel2(max_size_mb=l2_size_mb)
        self.l3_cache = CacheLevel3(cache_dir=cache_dir, max_size_mb=l3_size_mb) if enable_l3 else None
        
        self.auto_promote = auto_promote
        self.cleanup_interval = cleanup_interval
        self.enable_l3 = enable_l3
        
        # Statistics
        self.stats = {
            'hits': defaultdict(int),
            'misses': 0,
            'promotions': defaultdict(int),
            'evictions': defaultdict(int),
        }
        
        # Background cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()
        
        logger.info("Initialized advanced multi-level cache")
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        def cleanup_loop():
            while not self._stop_cleanup.is_set():
                try:
                    if self.l3_cache:
                        self.l3_cache.cleanup()
                    self._stop_cleanup.wait(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    self._stop_cleanup.wait(60)  # Wait a minute on error
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    @with_error_handling(max_retries=2, auto_recover=True)
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, checking all levels."""
        # Try L1 first
        entry = self.l1_cache.get(key)
        if entry:
            self.stats['hits'][CacheLevel.L1_MEMORY] += 1
            return entry.data
        
        # Try L2
        entry = self.l2_cache.get(key)
        if entry:
            self.stats['hits'][CacheLevel.L2_COMPRESSED] += 1
            
            # Auto-promote to L1 if enabled
            if self.auto_promote and entry.access_count > 2:
                if self.l1_cache.put(key, entry):
                    self.stats['promotions'][CacheLevel.L1_MEMORY] += 1
            
            return entry.data
        
        # Try L3 if enabled
        if self.l3_cache:
            entry = self.l3_cache.get(key)
            if entry:
                self.stats['hits'][CacheLevel.L3_DISK] += 1
                
                # Auto-promote to L2 if frequently accessed
                if self.auto_promote and entry.access_count > 5:
                    if self.l2_cache.put(key, entry):
                        self.stats['promotions'][CacheLevel.L2_COMPRESSED] += 1
                
                return entry.data
        
        # Cache miss
        self.stats['misses'] += 1
        return None
    
    @with_error_handling(max_retries=2, auto_recover=True)
    def put(
        self,
        key: str,
        data: Any,
        importance_score: float = 1.0,
        ttl: Optional[float] = None,
        target_level: Optional[CacheLevel] = None
    ) -> bool:
        """Put item in cache.
        
        Args:
            key: Cache key
            data: Data to cache
            importance_score: Importance score for eviction decisions
            ttl: Time to live in seconds
            target_level: Specific cache level to target
            
        Returns:
            True if successfully cached
        """
        # Calculate data size
        try:
            size_bytes = len(pickle.dumps(data))
        except Exception:
            size_bytes = 1024  # Default size estimate
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            data=data,
            size_bytes=size_bytes,
            importance_score=importance_score,
            ttl=ttl
        )
        
        # Determine target level
        if target_level is None:
            # Auto-select based on size and importance
            if size_bytes < 1024 * 1024 and importance_score > 2.0:  # < 1MB, high importance
                target_level = CacheLevel.L1_MEMORY
            elif size_bytes < 10 * 1024 * 1024:  # < 10MB
                target_level = CacheLevel.L2_COMPRESSED
            else:
                target_level = CacheLevel.L3_DISK if self.enable_l3 else CacheLevel.L2_COMPRESSED
        
        # Try to store in target level
        if target_level == CacheLevel.L1_MEMORY:
            if self.l1_cache.put(key, entry):
                return True
            # Fall back to L2
            target_level = CacheLevel.L2_COMPRESSED
        
        if target_level == CacheLevel.L2_COMPRESSED:
            if self.l2_cache.put(key, entry):
                return True
            # Fall back to L3
            if self.enable_l3:
                target_level = CacheLevel.L3_DISK
            else:
                return False
        
        if target_level == CacheLevel.L3_DISK and self.enable_l3:
            return self.l3_cache.put(key, entry)
        
        return False
    
    def invalidate(self, key: str) -> None:
        """Invalidate item from all cache levels."""
        # Remove from all levels
        if key in self.l1_cache:
            self.l1_cache._remove_entry(key)
        
        if key in self.l2_cache.entries:
            self.l2_cache._remove_entry(key)
        
        if self.l3_cache and key in self.l3_cache.metadata:
            self.l3_cache._remove_entry(key)
            self.l3_cache._save_metadata()
    
    def clear(self) -> None:
        """Clear all cache levels."""
        self.l1_cache.clear()
        self.l1_cache.current_size_bytes = 0
        
        self.l2_cache.entries.clear()
        self.l2_cache.metadata.clear()
        self.l2_cache.current_size_bytes = 0
        
        if self.l3_cache:
            # Clear disk cache
            for key in list(self.l3_cache.metadata.keys()):
                self.l3_cache._remove_entry(key)
            self.l3_cache._save_metadata()
        
        logger.info("Cleared all cache levels")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'hit_rate': self._calculate_hit_rate(),
            'hits_by_level': dict(self.stats['hits']),
            'misses': self.stats['misses'],
            'promotions_by_level': dict(self.stats['promotions']),
            'l1': self.l1_cache.get_statistics(),
            'l2': self.l2_cache.get_statistics(),
        }
        
        if self.l3_cache:
            stats['l3'] = self.l3_cache.get_statistics()
        
        return stats
    
    def _calculate_hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        total_hits = sum(self.stats['hits'].values())
        total_requests = total_hits + self.stats['misses']
        
        if total_requests == 0:
            return 0.0
        
        return total_hits / total_requests
    
    def optimize(self) -> None:
        """Optimize cache performance by rebalancing."""
        # Analyze access patterns and rebalance if needed
        # This is a placeholder for more sophisticated optimization
        logger.info("Cache optimization completed")
    
    def shutdown(self) -> None:
        """Shutdown cache system."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        
        if self.l3_cache:
            self.l3_cache._save_metadata()
        
        logger.info("Cache system shutdown")


# Specialized cache for federated learning objects
class FederatedLearningCache(AdvancedMultiLevelCache):
    """Specialized cache for federated learning objects."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Specialized importance calculators
        self.importance_calculators = {
            'model_gradients': self._calculate_gradient_importance,
            'model_weights': self._calculate_weight_importance,
            'aggregated_updates': self._calculate_update_importance,
            'client_data': self._calculate_data_importance,
        }
    
    def cache_model_gradients(
        self,
        client_id: str,
        round_num: int,
        gradients: torch.Tensor,
        ttl: float = 3600.0
    ) -> bool:
        """Cache model gradients with specialized handling."""
        key = f"gradients:{client_id}:{round_num}"
        importance = self._calculate_gradient_importance(gradients)
        
        return self.put(
            key=key,
            data=gradients,
            importance_score=importance,
            ttl=ttl,
            target_level=CacheLevel.L2_COMPRESSED
        )
    
    def cache_model_weights(
        self,
        model_id: str,
        weights: Dict[str, torch.Tensor],
        ttl: float = 7200.0
    ) -> bool:
        """Cache model weights."""
        key = f"weights:{model_id}"
        importance = self._calculate_weight_importance(weights)
        
        return self.put(
            key=key,
            data=weights,
            importance_score=importance,
            ttl=ttl,
            target_level=CacheLevel.L1_MEMORY if importance > 3.0 else CacheLevel.L2_COMPRESSED
        )
    
    def _calculate_gradient_importance(self, gradients: torch.Tensor) -> float:
        """Calculate importance score for gradients."""
        if not isinstance(gradients, torch.Tensor):
            return 1.0
        
        # Higher importance for larger gradients (more informative updates)
        grad_norm = torch.norm(gradients).item()
        return min(5.0, 1.0 + np.log(1.0 + grad_norm))
    
    def _calculate_weight_importance(self, weights: Dict[str, torch.Tensor]) -> float:
        """Calculate importance score for model weights."""
        if not weights:
            return 1.0
        
        # Global model weights are more important
        total_params = sum(w.numel() for w in weights.values())
        return min(5.0, 2.0 + np.log(1.0 + total_params / 1000000))  # Log of millions of params
    
    def _calculate_update_importance(self, update: Any) -> float:
        """Calculate importance score for aggregated updates."""
        return 4.0  # High importance for aggregated updates
    
    def _calculate_data_importance(self, data: Any) -> float:
        """Calculate importance score for client data."""
        return 2.0  # Medium importance for client data