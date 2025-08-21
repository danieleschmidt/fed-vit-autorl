"""Advanced caching system for Fed-ViT-AutoRL performance optimization."""

import time
import threading
import logging
import hashlib
import pickle
import json
import os
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
from dataclasses import dataclass
from enum import Enum
import weakref
from collections import OrderedDict
from functools import wraps, lru_cache
import asyncio
import concurrent.futures

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "least_recently_used"
    LFU = "least_frequently_used"
    TTL = "time_to_live"
    ADAPTIVE = "adaptive"


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    ttl: Optional[float] = None
    size_bytes: int = 0

    def __post_init__(self):
        """Initialize computed fields."""
        if self.last_accessed == 0.0:
            self.last_accessed = self.timestamp
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()

    def _calculate_size(self) -> int:
        """Estimate size of cached value in bytes."""
        try:
            return len(pickle.dumps(self.value))
        except:
            return len(str(self.value).encode('utf-8'))

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class AdaptiveCache:
    """High-performance adaptive cache with multiple eviction policies."""

    def __init__(self,
                 max_size: int = 1000,
                 max_memory_mb: float = 100.0,
                 default_ttl: Optional[float] = None,
                 policy: CachePolicy = CachePolicy.ADAPTIVE,
                 persistence_file: Optional[str] = None):
        """Initialize adaptive cache.

        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time to live in seconds
            policy: Cache eviction policy
            persistence_file: Optional file for cache persistence
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        self.policy = policy
        self.persistence_file = persistence_file

        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = OrderedDict()  # For LRU
        self._frequency_count: Dict[str, int] = {}  # For LFU
        self._total_size = 0
        self._lock = threading.RLock()

        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        # Load persistent cache if exists
        if self.persistence_file and os.path.exists(self.persistence_file):
            self._load_from_disk()

        logger.info(f"Initialized adaptive cache (policy={policy.value}, max_size={max_size})")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.misses += 1
                return None

            # Update access metadata
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._frequency_count[key] = self._frequency_count.get(key, 0) + 1

            # Update LRU order
            if key in self._access_order:
                del self._access_order[key]
            self._access_order[key] = True

            self.hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time to live override

        Returns:
            True if successfully cached, False otherwise
        """
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl
            )

            # Check if entry would exceed memory limit
            if entry.size_bytes > self.max_memory_bytes:
                logger.warning(f"Entry too large for cache: {entry.size_bytes} bytes")
                return False

            # Evict entries if necessary
            while (len(self._cache) >= self.max_size or
                   self._total_size + entry.size_bytes > self.max_memory_bytes):
                if not self._evict_one():
                    logger.warning("Cache eviction failed")
                    return False

            # Add entry
            self._cache[key] = entry
            self._access_order[key] = True
            self._frequency_count[key] = 1
            self._total_size += entry.size_bytes

            return True

    def delete(self, key: str) -> bool:
        """Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._frequency_count.clear()
            self._total_size = 0

    def _evict_one(self) -> bool:
        """Evict one entry based on policy.

        Returns:
            True if entry was evicted, False if cache is empty
        """
        if not self._cache:
            return False

        if self.policy == CachePolicy.LRU:
            key = next(iter(self._access_order))
        elif self.policy == CachePolicy.LFU:
            key = min(self._frequency_count.items(), key=lambda x: x[1])[0]
        elif self.policy == CachePolicy.TTL:
            # Evict expired entries first, then oldest
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            if expired_keys:
                key = expired_keys[0]
            else:
                key = min(self._cache.items(), key=lambda x: x[1].timestamp)[0]
        else:  # ADAPTIVE
            key = self._adaptive_evict()

        self._remove_entry(key)
        self.evictions += 1
        return True

    def _adaptive_evict(self) -> str:
        """Adaptive eviction algorithm combining multiple factors."""
        current_time = time.time()
        scores = {}

        for key, entry in self._cache.items():
            # Factors: recency, frequency, size, expiration proximity
            recency_score = current_time - entry.last_accessed
            frequency_score = 1.0 / max(entry.access_count, 1)
            size_score = entry.size_bytes / self.max_memory_bytes

            if entry.ttl:
                ttl_score = (entry.ttl - (current_time - entry.timestamp)) / entry.ttl
                ttl_score = max(0, ttl_score)
            else:
                ttl_score = 1.0

            # Weighted combination (higher score = better candidate for eviction)
            composite_score = (
                0.4 * recency_score +
                0.3 * frequency_score +
                0.2 * size_score +
                0.1 * (1 - ttl_score)
            )
            scores[key] = composite_score

        return max(scores.items(), key=lambda x: x[1])[0]

    def _remove_entry(self, key: str) -> None:
        """Remove entry and update metadata."""
        if key in self._cache:
            entry = self._cache[key]
            self._total_size -= entry.size_bytes
            del self._cache[key]

        if key in self._access_order:
            del self._access_order[key]

        if key in self._frequency_count:
            del self._frequency_count[key]

    def _load_from_disk(self) -> None:
        """Load cache from persistent storage."""
        try:
            with open(self.persistence_file, 'rb') as f:
                data = pickle.load(f)
                for key, entry_data in data.items():
                    entry = CacheEntry(**entry_data)
                    if not entry.is_expired():
                        self._cache[key] = entry
                        self._access_order[key] = True
                        self._frequency_count[key] = entry.access_count
                        self._total_size += entry.size_bytes
            logger.info(f"Loaded {len(self._cache)} entries from disk cache")
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")

    def save_to_disk(self) -> None:
        """Save cache to persistent storage."""
        if not self.persistence_file:
            return

        try:
            with self._lock:
                # Convert to serializable format
                data = {}
                for key, entry in self._cache.items():
                    data[key] = {
                        'key': entry.key,
                        'value': entry.value,
                        'timestamp': entry.timestamp,
                        'access_count': entry.access_count,
                        'last_accessed': entry.last_accessed,
                        'ttl': entry.ttl,
                        'size_bytes': entry.size_bytes
                    }

            with open(self.persistence_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved {len(data)} entries to disk cache")
        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / max(total_requests, 1)) * 100

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': self._total_size / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'policy': self.policy.value
            }


class DistributedCache:
    """Distributed cache for federated learning systems."""

    def __init__(self, nodes: List[str], local_cache: Optional[AdaptiveCache] = None):
        """Initialize distributed cache.

        Args:
            nodes: List of cache node addresses
            local_cache: Optional local cache for L1 caching
        """
        self.nodes = nodes
        self.local_cache = local_cache or AdaptiveCache(max_size=100)
        self.node_ring = self._create_consistent_hash_ring()

        logger.info(f"Initialized distributed cache with {len(nodes)} nodes")

    def _create_consistent_hash_ring(self) -> Dict[int, str]:
        """Create consistent hash ring for node selection."""
        ring = {}
        for node in self.nodes:
            for i in range(100):  # Virtual nodes
                hash_key = hashlib.md5(f"{node}:{i}".encode()).hexdigest()
                ring[int(hash_key, 16)] = node
        return dict(sorted(ring.items()))

    def _get_node(self, key: str) -> str:
        """Get responsible node for key using consistent hashing."""
        if not self.nodes:
            return "localhost"

        hash_key = int(hashlib.md5(key.encode()).hexdigest(), 16)

        # Find next node in ring
        for ring_key in sorted(self.node_ring.keys()):
            if hash_key <= ring_key:
                return self.node_ring[ring_key]

        # Wrap around to first node
        return self.node_ring[min(self.node_ring.keys())]

    async def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache."""
        # Try local cache first
        value = self.local_cache.get(key)
        if value is not None:
            return value

        # Try remote node
        node = self._get_node(key)
        # Simulate remote cache access (would use actual network call)
        # For now, just return None as remote cache is not implemented
        return None

    async def put(self, key: str, value: Any) -> bool:
        """Put value in distributed cache."""
        # Store in local cache
        local_success = self.local_cache.put(key, value)

        # Store in remote node
        node = self._get_node(key)
        # Simulate remote cache storage (would use actual network call)
        remote_success = True

        return local_success and remote_success


def cached(ttl: Optional[float] = None,
           cache: Optional[AdaptiveCache] = None,
           key_func: Optional[Callable] = None):
    """Decorator for function result caching.

    Args:
        ttl: Time to live in seconds
        cache: Cache instance to use
        key_func: Custom key generation function
    """
    if cache is None:
        cache = AdaptiveCache(max_size=1000, default_ttl=ttl)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = _generate_cache_key(func.__name__, args, kwargs)

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Compute result and cache
            result = func(*args, **kwargs)
            cache.put(key, result, ttl)
            return result

        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        return wrapper

    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate cache key for function call."""
    key_data = {
        'function': func_name,
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    key_str = json.dumps(key_data, default=str, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


class CacheManager:
    """Global cache manager for coordinating multiple caches."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.caches: Dict[str, AdaptiveCache] = {}
            self.distributed_cache: Optional[DistributedCache] = None
            self.initialized = True
            logger.info("Initialized global cache manager")

    def get_cache(self, name: str, **kwargs) -> AdaptiveCache:
        """Get or create named cache.

        Args:
            name: Cache name
            **kwargs: Cache configuration parameters

        Returns:
            Cache instance
        """
        if name not in self.caches:
            self.caches[name] = AdaptiveCache(**kwargs)
            logger.info(f"Created cache: {name}")
        return self.caches[name]

    def setup_distributed_cache(self, nodes: List[str]) -> None:
        """Setup distributed cache.

        Args:
            nodes: List of cache node addresses
        """
        self.distributed_cache = DistributedCache(nodes)
        logger.info("Setup distributed cache")

    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}
        for name, cache in self.caches.items():
            stats[name] = cache.get_stats()
        return stats

    def clear_all(self) -> None:
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        logger.info("Cleared all caches")
