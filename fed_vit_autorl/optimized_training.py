"""Generation 3: Optimized Federated ViT-AutoRL Training Implementation

Adds performance optimization, caching, concurrent processing, resource pooling,
load balancing, and auto-scaling triggers for production-ready scalability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import time
import asyncio
import threading
import queue
import json
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock, RLock, Event
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import gc
from functools import lru_cache, wraps
import weakref

# Set multiprocessing start method
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Configure optimized logging with async handler
logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Advanced performance profiler for optimization insights."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        self.lock = threading.Lock()
    
    def start_timer(self, name: str):
        """Start performance timer."""
        with self.lock:
            self.start_times[name] = time.perf_counter()
    
    def end_timer(self, name: str) -> float:
        """End performance timer and return duration."""
        with self.lock:
            if name in self.start_times:
                duration = time.perf_counter() - self.start_times[name]
                self.metrics[name].append(duration)
                del self.start_times[name]
                return duration
            return 0.0
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        with self.lock:
            stats = {}
            for name, durations in self.metrics.items():
                if durations:
                    stats[name] = {
                        'count': len(durations),
                        'mean': np.mean(durations),
                        'std': np.std(durations),
                        'min': np.min(durations),
                        'max': np.max(durations),
                        'p50': np.percentile(durations, 50),
                        'p95': np.percentile(durations, 95),
                        'p99': np.percentile(durations, 99)
                    }
            return stats


class AdaptiveCachingSystem:
    """Intelligent caching system with LRU and adaptive policies."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.lock = RLock()
        
        # Start cache cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    @lru_cache(maxsize=128)
    def _hash_key(self, key: str) -> str:
        """Create hash key for cache."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            hkey = self._hash_key(key)
            if hkey in self.cache:
                # Check TTL
                if time.time() - self.cache[hkey]['timestamp'] < self.ttl_seconds:
                    self.access_times[hkey] = time.time()
                    self.access_counts[hkey] += 1
                    return self.cache[hkey]['data']
                else:
                    # Expired
                    del self.cache[hkey]
                    if hkey in self.access_times:
                        del self.access_times[hkey]
                    if hkey in self.access_counts:
                        del self.access_counts[hkey]
            return None
    
    def put(self, key: str, data: Any):
        """Put item in cache."""
        with self.lock:
            hkey = self._hash_key(key)
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size and hkey not in self.cache:
                self._evict_lru()
            
            self.cache[hkey] = {
                'data': data,
                'timestamp': time.time()
            }
            self.access_times[hkey] = time.time()
            self.access_counts[hkey] = 1
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        # Find LRU item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from all structures
        del self.cache[lru_key]
        del self.access_times[lru_key]
        if lru_key in self.access_counts:
            del self.access_counts[lru_key]
    
    def _cleanup_loop(self):
        """Background cleanup of expired items."""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                with self.lock:
                    current_time = time.time()
                    expired_keys = []
                    
                    for key, item in self.cache.items():
                        if current_time - item['timestamp'] > self.ttl_seconds:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.cache[key]
                        if key in self.access_times:
                            del self.access_times[key]
                        if key in self.access_counts:
                            del self.access_counts[key]
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': sum(self.access_counts.values()) / max(1, len(self.access_counts)),
                'total_accesses': sum(self.access_counts.values())
            }


class ResourcePool:
    """Dynamic resource pool for concurrent training."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = None, 
                 resource_factory: Callable = None):
        self.min_workers = min_workers
        self.max_workers = max_workers or mp.cpu_count()
        self.resource_factory = resource_factory or (lambda: {})
        
        self.available_resources = queue.Queue()
        self.in_use_resources = set()
        self.total_created = 0
        self.lock = threading.Lock()
        
        # Pre-create minimum resources
        for _ in range(min_workers):
            resource = self._create_resource()
            self.available_resources.put(resource)
    
    def _create_resource(self) -> Any:
        """Create new resource."""
        with self.lock:
            resource_id = f"resource_{self.total_created}"
            self.total_created += 1
            
        resource = {
            'id': resource_id,
            'data': self.resource_factory(),
            'created_at': time.time(),
            'use_count': 0
        }
        return resource
    
    def acquire(self, timeout: float = 30.0) -> Optional[Any]:
        """Acquire resource from pool."""
        try:
            # Try to get available resource
            resource = self.available_resources.get(timeout=min(timeout, 1.0))
        except queue.Empty:
            # Create new resource if under limit
            with self.lock:
                if self.total_created < self.max_workers:
                    resource = self._create_resource()
                else:
                    return None
        
        with self.lock:
            self.in_use_resources.add(resource['id'])
            resource['use_count'] += 1
        
        return resource
    
    def release(self, resource: Any):
        """Release resource back to pool."""
        if resource and 'id' in resource:
            with self.lock:
                if resource['id'] in self.in_use_resources:
                    self.in_use_resources.remove(resource['id'])
                    
                    # Check if resource should be recycled
                    if (resource['use_count'] > 100 or 
                        time.time() - resource['created_at'] > 3600):
                        # Don't return over-used or old resources
                        pass
                    else:
                        self.available_resources.put(resource)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            return {
                'total_created': self.total_created,
                'available': self.available_resources.qsize(),
                'in_use': len(self.in_use_resources),
                'min_workers': self.min_workers,
                'max_workers': self.max_workers
            }


class LoadBalancer:
    """Intelligent load balancer for client distribution."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.client_loads = defaultdict(float)
        self.client_performance = defaultdict(lambda: {'avg_time': 1.0, 'success_rate': 1.0})
        self.lock = threading.Lock()
    
    def get_optimal_client_assignment(self, batch_size: int) -> List[int]:
        """Get optimal client assignment for batch processing."""
        with self.lock:
            # Calculate client scores (lower is better)
            client_scores = {}
            for client_id in range(self.num_clients):
                load_factor = self.client_loads.get(client_id, 0.0)
                perf = self.client_performance[client_id]
                
                # Combined score: load + inverse performance
                score = load_factor + (1.0 / max(0.1, perf['success_rate'])) * perf['avg_time']
                client_scores[client_id] = score
            
            # Sort clients by score and assign
            sorted_clients = sorted(client_scores.keys(), key=lambda x: client_scores[x])
            return sorted_clients[:min(batch_size, len(sorted_clients))]
    
    def update_client_load(self, client_id: int, load_delta: float):
        """Update client load."""
        with self.lock:
            self.client_loads[client_id] = max(0.0, self.client_loads[client_id] + load_delta)
    
    def update_client_performance(self, client_id: int, execution_time: float, success: bool):
        """Update client performance metrics."""
        with self.lock:
            perf = self.client_performance[client_id]
            
            # Exponential moving average
            alpha = 0.1
            perf['avg_time'] = (1 - alpha) * perf['avg_time'] + alpha * execution_time
            perf['success_rate'] = (1 - alpha) * perf['success_rate'] + alpha * (1.0 if success else 0.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self.lock:
            return {
                'client_loads': dict(self.client_loads),
                'client_performance': {
                    k: dict(v) for k, v in self.client_performance.items()
                },
                'total_load': sum(self.client_loads.values())
            }


class AutoScaler:
    """Auto-scaling system for dynamic resource allocation."""
    
    def __init__(self, min_clients: int = 2, max_clients: int = 50):
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.current_clients = min_clients
        
        self.metrics_window = deque(maxlen=20)  # Keep last 20 measurements
        self.scale_cooldown = 60.0  # 60 seconds between scaling actions
        self.last_scale_time = 0.0
        
        self.lock = threading.Lock()
    
    def should_scale(self, current_metrics: Dict[str, float]) -> Tuple[bool, int, str]:
        """Determine if scaling is needed."""
        with self.lock:
            self.metrics_window.append({
                'timestamp': time.time(),
                **current_metrics
            })
            
            # Check cooldown
            if time.time() - self.last_scale_time < self.scale_cooldown:
                return False, self.current_clients, "cooldown"
            
            if len(self.metrics_window) < 5:
                return False, self.current_clients, "insufficient_data"
            
            # Calculate recent averages
            recent_metrics = list(self.metrics_window)[-5:]
            avg_cpu = np.mean([m.get('cpu_usage', 50) for m in recent_metrics])
            avg_memory = np.mean([m.get('memory_usage', 50) for m in recent_metrics])
            avg_queue_size = np.mean([m.get('queue_size', 0) for m in recent_metrics])
            avg_latency = np.mean([m.get('avg_latency', 1.0) for m in recent_metrics])
            
            # Scaling decisions
            scale_up_needed = (
                (avg_cpu > 80 or avg_memory > 85 or avg_queue_size > 10 or avg_latency > 5.0)
                and self.current_clients < self.max_clients
            )
            
            scale_down_needed = (
                (avg_cpu < 30 and avg_memory < 40 and avg_queue_size < 2 and avg_latency < 1.0)
                and self.current_clients > self.min_clients
            )
            
            if scale_up_needed:
                new_clients = min(self.max_clients, self.current_clients + max(1, self.current_clients // 4))
                return True, new_clients, "scale_up"
            elif scale_down_needed:
                new_clients = max(self.min_clients, self.current_clients - max(1, self.current_clients // 4))
                return True, new_clients, "scale_down"
            
            return False, self.current_clients, "no_change"
    
    def apply_scaling(self, new_client_count: int, reason: str) -> bool:
        """Apply scaling decision."""
        with self.lock:
            if new_client_count != self.current_clients:
                logger.info(f"Auto-scaling: {self.current_clients} -> {new_client_count} ({reason})")
                self.current_clients = new_client_count
                self.last_scale_time = time.time()
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics."""
        with self.lock:
            return {
                'current_clients': self.current_clients,
                'min_clients': self.min_clients,
                'max_clients': self.max_clients,
                'metrics_window_size': len(self.metrics_window),
                'time_since_last_scale': time.time() - self.last_scale_time
            }


class OptimizedFederatedSystem:
    """Generation 3: Highly optimized federated learning system."""
    
    def __init__(self, initial_clients: int = 5, embed_dim: int = 768):
        
        # Initialize optimization components
        self.profiler = PerformanceProfiler()
        self.cache = AdaptiveCachingSystem(max_size=2000, ttl_seconds=1800)
        self.load_balancer = LoadBalancer(initial_clients)
        self.auto_scaler = AutoScaler(min_clients=2, max_clients=20)
        
        self.embed_dim = embed_dim
        self.start_time = time.time()
        
        # Resource pools
        self.model_pool = ResourcePool(
            min_workers=2,
            max_workers=10,
            resource_factory=self._create_model_resource
        )
        
        self.optimizer_pool = ResourcePool(
            min_workers=2,
            max_workers=10,
            resource_factory=self._create_optimizer_resource
        )
        
        # Thread pools for concurrent processing
        self.training_executor = ThreadPoolExecutor(max_workers=mp.cpu_count() * 2)
        self.inference_executor = ThreadPoolExecutor(max_workers=mp.cpu_count())
        
        # Concurrent data structures
        self.training_queue = queue.PriorityQueue(maxsize=100)
        self.results_queue = queue.Queue(maxsize=100)
        
        # Initialize models with caching
        self.global_model = self._create_and_cache_model("global")
        self.client_models = {}
        
        # Initialize clients
        for i in range(initial_clients):
            self._add_client(i)
        
        # Start background workers
        self._start_background_workers()
        
        logger.info(f"Optimized federated system initialized with {initial_clients} clients")
    
    def _create_model_resource(self) -> Dict[str, Any]:
        """Create model resource for pool."""
        from .simple_training import SimpleViTPerception
        return {
            'model': SimpleViTPerception(embed_dim=self.embed_dim),
            'device': 'cpu'  # Could be dynamically assigned
        }
    
    def _create_optimizer_resource(self) -> Dict[str, Any]:
        """Create optimizer resource for pool."""
        return {
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'betas': (0.9, 0.999)
        }
    
    def _create_and_cache_model(self, model_id: str):
        """Create model with caching."""
        cached_model = self.cache.get(f"model_{model_id}")
        if cached_model is not None:
            return cached_model
        
        from .simple_training import SimpleViTPerception
        model = SimpleViTPerception(embed_dim=self.embed_dim)
        
        # Cache model state dict instead of model itself (for memory efficiency)
        self.cache.put(f"model_{model_id}_state", model.state_dict())
        
        return model
    
    def _add_client(self, client_id: int):
        """Add new client to system."""
        client_model = self._create_and_cache_model(f"client_{client_id}")
        client_model.load_state_dict(self.global_model.state_dict())
        
        self.client_models[client_id] = {
            'model': client_model,
            'optimizer': optim.Adam(client_model.parameters(), lr=1e-4),
            'last_used': time.time(),
            'training_stats': {'total_batches': 0, 'avg_loss': 0.0}
        }
    
    def _remove_client(self, client_id: int):
        """Remove client from system."""
        if client_id in self.client_models:
            del self.client_models[client_id]
            # Clean cache entries
            self.cache.get(f"model_client_{client_id}_state")  # This will remove if expired
    
    def _start_background_workers(self):
        """Start background worker threads."""
        
        def training_worker():
            """Background training worker."""
            while True:
                try:
                    # Get training task from queue
                    priority, task = self.training_queue.get(timeout=1.0)
                    
                    # Execute training
                    result = self._execute_training_task(task)
                    
                    # Put result
                    self.results_queue.put(result)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Training worker error: {e}")
        
        def resource_monitor():
            """Background resource monitoring."""
            while True:
                try:
                    time.sleep(30)  # Monitor every 30 seconds
                    
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    metrics = {
                        'cpu_usage': cpu_percent,
                        'memory_usage': memory.percent,
                        'queue_size': self.training_queue.qsize(),
                        'avg_latency': self._calculate_avg_latency()
                    }
                    
                    # Check auto-scaling
                    should_scale, new_count, reason = self.auto_scaler.should_scale(metrics)
                    if should_scale:
                        self._apply_auto_scaling(new_count, reason)
                    
                    # Garbage collection if memory is high
                    if memory.percent > 85:
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except Exception as e:
                    logger.error(f"Resource monitor error: {e}")
        
        # Start worker threads
        for _ in range(2):  # Multiple training workers
            worker_thread = threading.Thread(target=training_worker, daemon=True)
            worker_thread.start()
        
        monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
        monitor_thread.start()
    
    def _execute_training_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual training task."""
        self.profiler.start_timer("training_task")
        
        try:
            client_id = task['client_id']
            data = task['data']
            labels = task['labels']
            epochs = task.get('epochs', 1)
            
            # Get or create client
            if client_id not in self.client_models:
                self._add_client(client_id)
            
            client_info = self.client_models[client_id]
            client_model = client_info['model']
            optimizer = client_info['optimizer']
            
            # Training loop
            client_model.train()
            epoch_losses = []
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = client_model(data)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            
            # Calculate metrics
            avg_loss = np.mean(epoch_losses)
            
            with torch.no_grad():
                outputs = client_model(data)
                pred = outputs.argmax(dim=1)
                accuracy = pred.eq(labels).sum().item() / len(labels)
            
            # Update client stats
            client_info['last_used'] = time.time()
            client_info['training_stats']['total_batches'] += 1
            client_info['training_stats']['avg_loss'] = (
                client_info['training_stats']['avg_loss'] * 0.9 + avg_loss * 0.1
            )
            
            execution_time = self.profiler.end_timer("training_task")
            
            # Update load balancer
            self.load_balancer.update_client_performance(client_id, execution_time, True)
            
            return {
                'client_id': client_id,
                'loss': avg_loss,
                'accuracy': accuracy,
                'execution_time': execution_time,
                'success': True
            }
            
        except Exception as e:
            execution_time = self.profiler.end_timer("training_task")
            logger.error(f"Training task failed for client {task.get('client_id', -1)}: {e}")
            
            if 'client_id' in task:
                self.load_balancer.update_client_performance(task['client_id'], execution_time, False)
            
            return {
                'client_id': task.get('client_id', -1),
                'loss': float('inf'),
                'accuracy': 0.0,
                'execution_time': execution_time,
                'success': False,
                'error': str(e)
            }
    
    def _calculate_avg_latency(self) -> float:
        """Calculate average latency from recent performance data."""
        perf_stats = self.profiler.get_stats()
        if 'training_task' in perf_stats:
            return perf_stats['training_task'].get('mean', 1.0)
        return 1.0
    
    def _apply_auto_scaling(self, new_client_count: int, reason: str):
        """Apply auto-scaling by adding/removing clients."""
        current_count = len(self.client_models)
        
        if new_client_count > current_count:
            # Scale up: add clients
            for i in range(current_count, new_client_count):
                self._add_client(i)
            logger.info(f"Scaled up from {current_count} to {new_client_count} clients")
        
        elif new_client_count < current_count:
            # Scale down: remove least recently used clients
            client_items = [(cid, info['last_used']) for cid, info in self.client_models.items()]
            client_items.sort(key=lambda x: x[1])  # Sort by last_used
            
            clients_to_remove = client_items[:current_count - new_client_count]
            for client_id, _ in clients_to_remove:
                self._remove_client(client_id)
            
            logger.info(f"Scaled down from {current_count} to {new_client_count} clients")
        
        self.auto_scaler.apply_scaling(new_client_count, reason)
    
    async def parallel_client_training(
        self, 
        client_data: List[torch.Tensor],
        client_labels: List[torch.Tensor],
        epochs: int = 1
    ) -> List[Dict[str, Any]]:
        """Execute parallel client training with optimization."""
        
        self.profiler.start_timer("parallel_training")
        
        # Get optimal client assignment
        active_clients = self.load_balancer.get_optimal_client_assignment(len(client_data))
        
        # Create training tasks
        training_tasks = []
        for i, client_id in enumerate(active_clients):
            if i < len(client_data):
                task = {
                    'client_id': client_id,
                    'data': client_data[i],
                    'labels': client_labels[i],
                    'epochs': epochs,
                    'priority': i  # Lower priority for later clients
                }
                training_tasks.append(task)
        
        # Submit tasks to queue
        for task in training_tasks:
            try:
                self.training_queue.put((task['priority'], task), timeout=5.0)
            except queue.Full:
                logger.warning("Training queue full, skipping some tasks")
                break
        
        # Collect results
        results = []
        for _ in range(len(training_tasks)):
            try:
                result = self.results_queue.get(timeout=30.0)
                results.append(result)
                
                # Update load balancer
                self.load_balancer.update_client_load(
                    result['client_id'], 
                    -0.1  # Decrease load after completion
                )
                
            except queue.Empty:
                logger.warning("Training result timeout")
                break
        
        self.profiler.end_timer("parallel_training")
        return results
    
    def optimized_federated_averaging(self) -> bool:
        """Optimized federated averaging with caching and parallelization."""
        
        self.profiler.start_timer("federated_averaging")
        
        try:
            # Use cached global state if available
            cache_key = "global_model_state"
            
            active_clients = list(self.client_models.keys())
            if not active_clients:
                return False
            
            # Parallel parameter collection
            def collect_client_params(client_id):
                if client_id in self.client_models:
                    return client_id, self.client_models[client_id]['model'].state_dict()
                return client_id, None
            
            # Use thread pool for parameter collection
            client_params = {}
            with ThreadPoolExecutor(max_workers=min(len(active_clients), 8)) as executor:
                futures = [executor.submit(collect_client_params, cid) for cid in active_clients]
                for future in as_completed(futures):
                    client_id, params = future.result()
                    if params is not None:
                        client_params[client_id] = params
            
            if not client_params:
                return False
            
            # Weighted averaging based on client performance
            global_dict = self.global_model.state_dict()
            for key in global_dict.keys():
                global_dict[key] = torch.zeros_like(global_dict[key])
            
            total_weight = 0.0
            for client_id, params in client_params.items():
                # Weight based on performance and data quality
                perf = self.load_balancer.client_performance[client_id]
                weight = perf['success_rate'] * (1.0 / max(0.1, perf['avg_time']))
                
                for key in global_dict.keys():
                    if key in params:
                        global_dict[key] += params[key] * weight
                
                total_weight += weight
            
            # Normalize
            if total_weight > 0:
                for key in global_dict.keys():
                    global_dict[key] /= total_weight
            
            # Update global model
            self.global_model.load_state_dict(global_dict)
            
            # Cache updated global state
            self.cache.put(cache_key, global_dict)
            
            # Update all client models in parallel
            def update_client_model(client_id):
                if client_id in self.client_models:
                    self.client_models[client_id]['model'].load_state_dict(global_dict)
            
            with ThreadPoolExecutor(max_workers=min(len(active_clients), 8)) as executor:
                futures = [executor.submit(update_client_model, cid) for cid in active_clients]
                for future in as_completed(futures):
                    future.result()  # Wait for completion
            
            self.profiler.end_timer("federated_averaging")
            return True
            
        except Exception as e:
            self.profiler.end_timer("federated_averaging")
            logger.error(f"Optimized federated averaging failed: {e}")
            return False
    
    def optimized_evaluation(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, Any]:
        """Optimized model evaluation with caching and parallelization."""
        
        self.profiler.start_timer("evaluation")
        
        try:
            # Check cache first
            data_hash = hashlib.md5(test_data.numpy().tobytes()).hexdigest()[:8]
            model_hash = hashlib.md5(str(self.global_model.state_dict()).encode()).hexdigest()[:8]
            cache_key = f"eval_{data_hash}_{model_hash}"
            
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.profiler.end_timer("evaluation")
                return cached_result
            
            # Perform evaluation
            self.global_model.eval()
            
            with torch.no_grad():
                # Batch evaluation for large datasets
                batch_size = 64
                all_outputs = []
                all_targets = []
                
                for i in range(0, len(test_data), batch_size):
                    batch_data = test_data[i:i+batch_size]
                    batch_labels = test_labels[i:i+batch_size]
                    
                    outputs = self.global_model(batch_data)
                    all_outputs.append(outputs)
                    all_targets.append(batch_labels)
                
                # Concatenate results
                all_outputs = torch.cat(all_outputs, dim=0)
                all_targets = torch.cat(all_targets, dim=0)
                
                # Calculate metrics
                loss = nn.CrossEntropyLoss()(all_outputs, all_targets)
                pred = all_outputs.argmax(dim=1)
                accuracy = pred.eq(all_targets).sum().item() / len(all_targets)
                
                # Additional metrics
                probs = torch.softmax(all_outputs, dim=1)
                confidence = probs.max(dim=1)[0].mean()
                
            result = {
                'loss': loss.item(),
                'accuracy': accuracy,
                'confidence': confidence.item(),
                'num_samples': len(test_data)
            }
            
            # Cache result
            self.cache.put(cache_key, result)
            
            self.profiler.end_timer("evaluation")
            return result
            
        except Exception as e:
            self.profiler.end_timer("evaluation")
            logger.error(f"Optimized evaluation failed: {e}")
            return {'loss': float('inf'), 'accuracy': 0.0, 'confidence': 0.0}
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        
        return {
            'performance': self.profiler.get_stats(),
            'cache': self.cache.get_stats(),
            'load_balancer': self.load_balancer.get_stats(),
            'auto_scaler': self.auto_scaler.get_stats(),
            'model_pool': self.model_pool.get_stats(),
            'optimizer_pool': self.optimizer_pool.get_stats(),
            'system_resources': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            },
            'training_queue_size': self.training_queue.qsize(),
            'results_queue_size': self.results_queue.qsize(),
            'active_clients': len(self.client_models),
            'uptime_seconds': time.time() - self.start_time
        }
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.training_executor.shutdown(wait=True, timeout=10)
            self.inference_executor.shutdown(wait=True, timeout=10)
            
            # Clear caches
            self.cache.cache.clear()
            
            logger.info("Optimized federated system cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def run_optimized_federated_training(
    num_rounds: int = 3,
    initial_clients: int = 5,
    enable_auto_scaling: bool = True
) -> bool:
    """Run optimized federated training with full performance features."""
    
    logger.info("🚀 Starting Generation 3: Optimized Federated Training")
    
    try:
        # Initialize optimized system
        fed_system = OptimizedFederatedSystem(
            initial_clients=initial_clients
        )
        
        # Generate optimized data with caching
        from .simple_training import generate_mock_data
        client_data, client_labels = generate_mock_data(batch_size=32, num_clients=initial_clients * 2)
        test_data, test_labels = generate_mock_data(batch_size=16, num_clients=1)
        test_data, test_labels = test_data[0], test_labels[0]
        
        logger.info("✅ Optimized data generation completed")
        
        # Optimized training loop
        for round_idx in range(num_rounds):
            logger.info(f"--- Optimized Round {round_idx + 1}/{num_rounds} ---")
            
            round_start = time.time()
            
            # Parallel client training
            active_client_count = len(fed_system.client_models)
            round_client_data = client_data[:active_client_count]
            round_client_labels = client_labels[:active_client_count]
            
            results = await fed_system.parallel_client_training(
                round_client_data,
                round_client_labels,
                epochs=2
            )
            
            # Log results
            successful_results = [r for r in results if r['success']]
            if successful_results:
                avg_loss = np.mean([r['loss'] for r in successful_results])
                avg_accuracy = np.mean([r['accuracy'] for r in successful_results])
                avg_time = np.mean([r['execution_time'] for r in successful_results])
                
                logger.info(f"Round {round_idx + 1} Training: "
                          f"Clients={len(successful_results)}, "
                          f"Avg Loss={avg_loss:.4f}, "
                          f"Avg Accuracy={avg_accuracy:.4f}, "
                          f"Avg Time={avg_time:.2f}s")
            else:
                logger.error("No successful training results in this round")
                continue
            
            # Optimized federated averaging
            if not fed_system.optimized_federated_averaging():
                logger.error("Optimized federated averaging failed")
                continue
            
            # Optimized evaluation
            eval_results = fed_system.optimized_evaluation(test_data, test_labels)
            logger.info(f"Round {round_idx + 1} Global Model: "
                       f"Loss={eval_results['loss']:.4f}, "
                       f"Accuracy={eval_results['accuracy']:.4f}, "
                       f"Confidence={eval_results['confidence']:.4f}")
            
            round_time = time.time() - round_start
            logger.info(f"Round {round_idx + 1} completed in {round_time:.2f}s")
            
            # Optimization statistics
            if round_idx % 2 == 0:  # Every other round
                opt_stats = fed_system.get_optimization_stats()
                logger.info(f"Optimization Stats: "
                          f"Active Clients={opt_stats['active_clients']}, "
                          f"Cache Hit Rate={opt_stats['cache'].get('hit_rate', 0):.2f}, "
                          f"CPU={opt_stats['system_resources']['cpu_percent']:.1f}%, "
                          f"Memory={opt_stats['system_resources']['memory_percent']:.1f}%")
        
        # Final optimization report
        final_stats = fed_system.get_optimization_stats()
        logger.info("🎉 Optimized federated training completed successfully!")
        
        # Save optimization report
        optimization_report = {
            "training_completed": True,
            "num_rounds": num_rounds,
            "initial_clients": initial_clients,
            "final_stats": final_stats,
            "auto_scaling_enabled": enable_auto_scaling,
            "timestamp": datetime.now().isoformat()
        }
        
        with open('optimized_training_report.json', 'w') as f:
            json.dump(optimization_report, f, indent=2, default=str)
        
        logger.info("💾 Optimization report saved")
        
        # Cleanup
        fed_system.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimized federated training failed: {str(e)}")
        return False


def main():
    """Main entry point for optimized training."""
    import asyncio
    
    success = asyncio.run(run_optimized_federated_training())
    if success:
        print("✅ Generation 3 optimized training completed!")
    else:
        print("❌ Generation 3 optimized training failed!")


if __name__ == "__main__":
    main()