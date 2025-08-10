"""Advanced load balancing for Fed-ViT-AutoRL distributed systems."""

import time
import threading
import logging
import random
import hashlib
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque
import asyncio
import heapq

logger = logging.getLogger(__name__)


class BalancingAlgorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    CONSISTENT_HASH = "consistent_hash"
    ADAPTIVE = "adaptive"
    HEALTH_AWARE = "health_aware"


class NodeStatus(Enum):
    """Node health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class LoadBalancerNode:
    """Node in the load balancer pool."""
    id: str
    address: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_health_check: float = 0.0
    status: NodeStatus = NodeStatus.HEALTHY
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    
    @property
    def load_factor(self) -> float:
        """Calculate current load factor."""
        if self.max_connections == 0:
            return 1.0
        return self.current_connections / self.max_connections
    
    @property
    def health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)."""
        if self.status == NodeStatus.OFFLINE:
            return 0.0
        elif self.status == NodeStatus.UNHEALTHY:
            return 0.1
        elif self.status == NodeStatus.DEGRADED:
            return 0.5
        
        # Factor in success rate, load, and response time
        success_component = self.success_rate
        load_component = 1.0 - min(self.load_factor, 1.0)
        
        # Response time component (normalize to 0-1, assuming 1000ms as max)
        response_component = max(0.0, 1.0 - (self.average_response_time / 1000.0))
        
        return (success_component * 0.4 + load_component * 0.3 + response_component * 0.3)


@dataclass
class RequestMetrics:
    """Metrics for a request."""
    node_id: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = True
    response_size: int = 0
    error_message: Optional[str] = None


class AdvancedLoadBalancer:
    """Advanced load balancer with multiple algorithms and health checking."""
    
    def __init__(self, 
                 algorithm: BalancingAlgorithm = BalancingAlgorithm.ADAPTIVE,
                 health_check_interval: float = 30.0,
                 health_check_timeout: float = 5.0):
        """Initialize advanced load balancer.
        
        Args:
            algorithm: Load balancing algorithm
            health_check_interval: Interval between health checks
            health_check_timeout: Timeout for health checks
        """
        self.algorithm = algorithm
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        
        # Node management
        self.nodes: Dict[str, LoadBalancerNode] = {}
        self.active_nodes: List[str] = []
        self._node_lock = threading.RLock()
        
        # Algorithm state
        self._round_robin_index = 0
        self._consistent_hash_ring: Dict[int, str] = {}
        
        # Metrics and monitoring
        self.request_metrics: deque = deque(maxlen=10000)
        self.total_requests = 0
        self.successful_requests = 0
        
        # Health checking
        self._health_check_active = False
        self._health_check_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.health_check_callback: Optional[Callable[[str], bool]] = None
        
        logger.info(f"Initialized load balancer with {algorithm.value} algorithm")
    
    def add_node(self, node_id: str, address: str, port: int, 
                 weight: float = 1.0, max_connections: int = 100,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add node to load balancer pool.
        
        Args:
            node_id: Unique node identifier
            address: Node address
            port: Node port
            weight: Node weight for weighted algorithms
            max_connections: Maximum connections per node
            metadata: Additional node metadata
        """
        with self._node_lock:
            node = LoadBalancerNode(
                id=node_id,
                address=address,
                port=port,
                weight=weight,
                max_connections=max_connections,
                metadata=metadata or {}
            )
            
            self.nodes[node_id] = node
            self.active_nodes.append(node_id)
            
            # Update consistent hash ring
            self._update_consistent_hash_ring()
            
            logger.info(f"Added node {node_id} ({address}:{port})")
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node from load balancer pool.
        
        Args:
            node_id: Node identifier to remove
            
        Returns:
            True if node was removed, False if not found
        """
        with self._node_lock:
            if node_id not in self.nodes:
                return False
            
            del self.nodes[node_id]
            if node_id in self.active_nodes:
                self.active_nodes.remove(node_id)
            
            # Update consistent hash ring
            self._update_consistent_hash_ring()
            
            logger.info(f"Removed node {node_id}")
            return True
    
    def update_node_status(self, node_id: str, status: NodeStatus) -> bool:
        """Update node status.
        
        Args:
            node_id: Node identifier
            status: New status
            
        Returns:
            True if updated successfully
        """
        with self._node_lock:
            if node_id not in self.nodes:
                return False
            
            old_status = self.nodes[node_id].status
            self.nodes[node_id].status = status
            self.nodes[node_id].last_health_check = time.time()
            
            # Update active nodes list
            if status in [NodeStatus.HEALTHY, NodeStatus.DEGRADED]:
                if node_id not in self.active_nodes:
                    self.active_nodes.append(node_id)
            else:
                if node_id in self.active_nodes:
                    self.active_nodes.remove(node_id)
            
            if old_status != status:
                logger.info(f"Node {node_id} status changed: {old_status.value} -> {status.value}")
            
            return True
    
    def get_next_node(self, request_key: Optional[str] = None) -> Optional[LoadBalancerNode]:
        """Get next node using configured algorithm.
        
        Args:
            request_key: Optional key for consistent hashing
            
        Returns:
            Selected node or None if no nodes available
        """
        with self._node_lock:
            if not self.active_nodes:
                logger.warning("No active nodes available")
                return None
            
            if self.algorithm == BalancingAlgorithm.ROUND_ROBIN:
                return self._round_robin_select()
            elif self.algorithm == BalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_select()
            elif self.algorithm == BalancingAlgorithm.LEAST_CONNECTIONS:
                return self._least_connections_select()
            elif self.algorithm == BalancingAlgorithm.LEAST_RESPONSE_TIME:
                return self._least_response_time_select()
            elif self.algorithm == BalancingAlgorithm.CONSISTENT_HASH:
                return self._consistent_hash_select(request_key or "")
            elif self.algorithm == BalancingAlgorithm.HEALTH_AWARE:
                return self._health_aware_select()
            else:  # ADAPTIVE
                return self._adaptive_select()
    
    def _round_robin_select(self) -> LoadBalancerNode:
        """Round robin selection."""
        node_id = self.active_nodes[self._round_robin_index]
        self._round_robin_index = (self._round_robin_index + 1) % len(self.active_nodes)
        return self.nodes[node_id]
    
    def _weighted_round_robin_select(self) -> LoadBalancerNode:
        """Weighted round robin selection."""
        # Create weighted list
        weighted_nodes = []
        for node_id in self.active_nodes:
            node = self.nodes[node_id]
            for _ in range(int(node.weight * 10)):  # Scale weights
                weighted_nodes.append(node_id)
        
        if not weighted_nodes:
            return self._round_robin_select()
        
        selected_id = random.choice(weighted_nodes)
        return self.nodes[selected_id]
    
    def _least_connections_select(self) -> LoadBalancerNode:
        """Least connections selection."""
        min_connections = float('inf')
        selected_node = None
        
        for node_id in self.active_nodes:
            node = self.nodes[node_id]
            if node.current_connections < min_connections:
                min_connections = node.current_connections
                selected_node = node
        
        return selected_node or self.nodes[self.active_nodes[0]]
    
    def _least_response_time_select(self) -> LoadBalancerNode:
        """Least response time selection."""
        min_response_time = float('inf')
        selected_node = None
        
        for node_id in self.active_nodes:
            node = self.nodes[node_id]
            # Factor in both response time and current connections
            effective_time = node.average_response_time * (1 + node.current_connections * 0.1)
            
            if effective_time < min_response_time:
                min_response_time = effective_time
                selected_node = node
        
        return selected_node or self.nodes[self.active_nodes[0]]
    
    def _consistent_hash_select(self, key: str) -> LoadBalancerNode:
        """Consistent hash selection."""
        if not self._consistent_hash_ring:
            return self.nodes[self.active_nodes[0]]
        
        hash_key = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        # Find next node in ring
        for ring_position in sorted(self._consistent_hash_ring.keys()):
            if hash_key <= ring_position:
                node_id = self._consistent_hash_ring[ring_position]
                if node_id in self.active_nodes:
                    return self.nodes[node_id]
        
        # Wrap around to first node
        first_position = min(self._consistent_hash_ring.keys())
        node_id = self._consistent_hash_ring[first_position]
        return self.nodes[node_id] if node_id in self.active_nodes else self.nodes[self.active_nodes[0]]
    
    def _health_aware_select(self) -> LoadBalancerNode:
        """Health-aware selection based on node health scores."""
        # Create weighted selection based on health scores
        healthy_nodes = []
        total_health = 0.0
        
        for node_id in self.active_nodes:
            node = self.nodes[node_id]
            health_score = node.health_score
            if health_score > 0:
                healthy_nodes.append((node, health_score))
                total_health += health_score
        
        if not healthy_nodes:
            return self.nodes[self.active_nodes[0]]
        
        # Weighted random selection
        rand_value = random.random() * total_health
        cumulative = 0.0
        
        for node, health_score in healthy_nodes:
            cumulative += health_score
            if rand_value <= cumulative:
                return node
        
        return healthy_nodes[-1][0]  # Fallback
    
    def _adaptive_select(self) -> LoadBalancerNode:
        """Adaptive selection combining multiple factors."""
        if not self.active_nodes:
            return None
        
        best_score = -1.0
        best_node = None
        
        for node_id in self.active_nodes:
            node = self.nodes[node_id]
            
            # Calculate composite score
            health_score = node.health_score
            load_score = 1.0 - node.load_factor
            response_score = max(0.0, 1.0 - (node.average_response_time / 1000.0))
            
            # Weight the factors
            composite_score = (
                health_score * 0.4 +
                load_score * 0.3 +
                response_score * 0.3
            )
            
            if composite_score > best_score:
                best_score = composite_score
                best_node = node
        
        return best_node or self.nodes[self.active_nodes[0]]
    
    def _update_consistent_hash_ring(self) -> None:
        """Update consistent hash ring for consistent hashing."""
        self._consistent_hash_ring.clear()
        
        for node_id in self.active_nodes:
            node = self.nodes[node_id]
            # Create virtual nodes based on weight
            virtual_nodes = max(1, int(node.weight * 100))
            
            for i in range(virtual_nodes):
                hash_input = f"{node_id}:{i}"
                hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
                self._consistent_hash_ring[hash_value] = node_id
    
    def record_request_start(self, node_id: str) -> str:
        """Record start of request to node.
        
        Args:
            node_id: Node handling the request
            
        Returns:
            Request ID for tracking
        """
        request_id = f"{node_id}_{time.time()}_{random.randint(1000, 9999)}"
        
        with self._node_lock:
            if node_id in self.nodes:
                self.nodes[node_id].current_connections += 1
                self.nodes[node_id].total_requests += 1
        
        metrics = RequestMetrics(
            node_id=node_id,
            start_time=time.time()
        )
        self.request_metrics.append((request_id, metrics))
        self.total_requests += 1
        
        return request_id
    
    def record_request_end(self, request_id: str, success: bool = True, 
                          response_size: int = 0, error_message: Optional[str] = None) -> None:
        """Record end of request.
        
        Args:
            request_id: Request ID from record_request_start
            success: Whether request was successful
            response_size: Size of response in bytes
            error_message: Error message if request failed
        """
        end_time = time.time()
        
        # Find request metrics
        for i, (rid, metrics) in enumerate(self.request_metrics):
            if rid == request_id:
                metrics.end_time = end_time
                metrics.success = success
                metrics.response_size = response_size
                metrics.error_message = error_message
                
                response_time = (end_time - metrics.start_time) * 1000  # Convert to ms
                
                with self._node_lock:
                    if metrics.node_id in self.nodes:
                        node = self.nodes[metrics.node_id]
                        node.current_connections = max(0, node.current_connections - 1)
                        
                        if not success:
                            node.failed_requests += 1
                        else:
                            self.successful_requests += 1
                        
                        # Update average response time (exponential moving average)
                        alpha = 0.1
                        node.average_response_time = (
                            alpha * response_time + 
                            (1 - alpha) * node.average_response_time
                        )
                
                break
    
    def start_health_checking(self) -> None:
        """Start health checking for all nodes."""
        if self._health_check_active:
            logger.warning("Health checking already active")
            return
        
        self._health_check_active = True
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()
        logger.info("Started health checking")
    
    def stop_health_checking(self) -> None:
        """Stop health checking."""
        self._health_check_active = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)
        logger.info("Stopped health checking")
    
    def _health_check_loop(self) -> None:
        """Main health checking loop."""
        while self._health_check_active:
            try:
                with self._node_lock:
                    nodes_to_check = list(self.nodes.keys())
                
                for node_id in nodes_to_check:
                    self._perform_health_check(node_id)
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(self.health_check_interval)
    
    def _perform_health_check(self, node_id: str) -> None:
        """Perform health check on specific node.
        
        Args:
            node_id: Node to check
        """
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        # Use callback if provided
        if self.health_check_callback:
            try:
                is_healthy = self.health_check_callback(node_id)
                new_status = NodeStatus.HEALTHY if is_healthy else NodeStatus.UNHEALTHY
            except Exception as e:
                logger.warning(f"Health check failed for node {node_id}: {e}")
                new_status = NodeStatus.UNHEALTHY
        else:
            # Default health check based on success rate and load
            if node.success_rate < 0.5:
                new_status = NodeStatus.UNHEALTHY
            elif node.success_rate < 0.8 or node.load_factor > 0.9:
                new_status = NodeStatus.DEGRADED
            else:
                new_status = NodeStatus.HEALTHY
        
        self.update_node_status(node_id, new_status)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics.
        
        Returns:
            Dictionary with comprehensive statistics
        """
        with self._node_lock:
            node_stats = {}
            for node_id, node in self.nodes.items():
                node_stats[node_id] = {
                    'address': f"{node.address}:{node.port}",
                    'status': node.status.value,
                    'weight': node.weight,
                    'current_connections': node.current_connections,
                    'total_requests': node.total_requests,
                    'failed_requests': node.failed_requests,
                    'success_rate': node.success_rate,
                    'average_response_time': node.average_response_time,
                    'load_factor': node.load_factor,
                    'health_score': node.health_score,
                    'last_health_check': node.last_health_check
                }
            
            success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100
            
            return {
                'algorithm': self.algorithm.value,
                'total_nodes': len(self.nodes),
                'active_nodes': len(self.active_nodes),
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'success_rate': success_rate,
                'health_checking_active': self._health_check_active,
                'nodes': node_stats
            }