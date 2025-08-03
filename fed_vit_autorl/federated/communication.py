"""Communication efficiency optimizations for federated learning."""

import time
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict, defaultdict
import pickle
import gzip
import zlib


logger = logging.getLogger(__name__)


class GradientCompressor:
    """Gradient compression for efficient federated communication."""
    
    def __init__(
        self,
        method: str = "top_k",
        compression_ratio: float = 0.1,
        quantization_bits: int = 8,
        error_feedback: bool = True,
    ):
        """Initialize gradient compressor.
        
        Args:
            method: Compression method ("top_k", "random_k", "threshold", "quantization")
            compression_ratio: Fraction of gradients to keep (for sparsification)
            quantization_bits: Number of bits for quantization
            error_feedback: Whether to use error feedback
        """
        self.method = method
        self.compression_ratio = compression_ratio
        self.quantization_bits = quantization_bits
        self.error_feedback = error_feedback
        
        # Error feedback state
        self.error_memory = {}
        
        logger.info(
            f"Initialized gradient compressor: {method}, "
            f"ratio={compression_ratio}, bits={quantization_bits}"
        )
    
    def compress(
        self,
        gradients: Dict[str, torch.Tensor],
        client_id: Optional[str] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Compress gradients for transmission.
        
        Args:
            gradients: Dictionary of gradient tensors
            client_id: Client identifier for error feedback
            
        Returns:
            Tuple of (compressed_gradients, compression_info)
        """
        compressed_gradients = OrderedDict()
        compression_info = {
            "method": self.method,
            "original_size": 0,
            "compressed_size": 0,
            "compression_ratio": 0.0,
        }
        
        # Apply error feedback if enabled
        if self.error_feedback and client_id:
            gradients = self._apply_error_feedback(gradients, client_id)
        
        for name, grad in gradients.items():
            if grad is None:
                compressed_gradients[name] = grad
                continue
            
            original_size = grad.numel()
            compression_info["original_size"] += original_size
            
            if self.method == "top_k":
                compressed_grad = self._top_k_compression(grad)
            elif self.method == "random_k":
                compressed_grad = self._random_k_compression(grad)
            elif self.method == "threshold":
                compressed_grad = self._threshold_compression(grad)
            elif self.method == "quantization":
                compressed_grad = self._quantization_compression(grad)
            else:
                raise ValueError(f"Unknown compression method: {self.method}")
            
            compressed_gradients[name] = compressed_grad
            
            # Calculate compressed size (non-zero elements)
            if hasattr(compressed_grad, 'values'):  # Sparse tensor
                compressed_size = compressed_grad.values().numel()
            else:
                compressed_size = torch.count_nonzero(compressed_grad).item()
            compression_info["compressed_size"] += compressed_size
        
        # Calculate overall compression ratio
        if compression_info["original_size"] > 0:
            compression_info["compression_ratio"] = (
                compression_info["compressed_size"] / compression_info["original_size"]
            )
        
        # Store error for feedback
        if self.error_feedback and client_id:
            self._store_compression_error(gradients, compressed_gradients, client_id)
        
        return compressed_gradients, compression_info
    
    def _top_k_compression(self, tensor: torch.Tensor) -> torch.Tensor:
        """Top-k sparsification: keep largest k elements.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Compressed tensor with only top-k elements
        """
        flattened = tensor.flatten()
        k = max(1, int(len(flattened) * self.compression_ratio))
        
        # Get top-k indices
        _, indices = torch.topk(torch.abs(flattened), k)
        
        # Create sparse tensor
        compressed = torch.zeros_like(flattened)
        compressed[indices] = flattened[indices]
        
        return compressed.reshape(tensor.shape)
    
    def _random_k_compression(self, tensor: torch.Tensor) -> torch.Tensor:
        """Random-k sparsification: keep random k elements.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Compressed tensor with random k elements
        """
        flattened = tensor.flatten()
        k = max(1, int(len(flattened) * self.compression_ratio))
        
        # Random indices
        indices = torch.randperm(len(flattened))[:k]
        
        # Create sparse tensor
        compressed = torch.zeros_like(flattened)
        compressed[indices] = flattened[indices]
        
        return compressed.reshape(tensor.shape)
    
    def _threshold_compression(self, tensor: torch.Tensor) -> torch.Tensor:
        """Threshold sparsification: keep elements above threshold.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Compressed tensor with elements above threshold
        """
        # Calculate threshold to achieve target compression ratio
        flattened = tensor.flatten()
        sorted_abs = torch.sort(torch.abs(flattened), descending=True)[0]
        threshold_idx = int(len(sorted_abs) * self.compression_ratio)
        threshold = sorted_abs[threshold_idx] if threshold_idx < len(sorted_abs) else 0.0
        
        # Apply threshold
        mask = torch.abs(tensor) >= threshold
        compressed = tensor * mask.float()
        
        return compressed
    
    def _quantization_compression(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantization compression: reduce precision.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Quantized tensor
        """
        # Calculate quantization levels
        num_levels = 2 ** self.quantization_bits
        
        # Find min/max for quantization range
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_min == tensor_max:
            return tensor
        
        # Quantize
        scale = (tensor_max - tensor_min) / (num_levels - 1)
        quantized = torch.round((tensor - tensor_min) / scale)
        quantized = torch.clamp(quantized, 0, num_levels - 1)
        
        # Dequantize
        dequantized = quantized * scale + tensor_min
        
        return dequantized
    
    def _apply_error_feedback(
        self,
        gradients: Dict[str, torch.Tensor],
        client_id: str,
    ) -> Dict[str, torch.Tensor]:
        """Apply error feedback to gradients.
        
        Args:
            gradients: Current gradients
            client_id: Client identifier
            
        Returns:
            Gradients with error feedback applied
        """
        if client_id not in self.error_memory:
            return gradients
        
        corrected_gradients = OrderedDict()
        error_memory = self.error_memory[client_id]
        
        for name, grad in gradients.items():
            if grad is None or name not in error_memory:
                corrected_gradients[name] = grad
            else:
                # Add accumulated error from previous rounds
                corrected_gradients[name] = grad + error_memory[name]
        
        return corrected_gradients
    
    def _store_compression_error(
        self,
        original_gradients: Dict[str, torch.Tensor],
        compressed_gradients: Dict[str, torch.Tensor],
        client_id: str,
    ) -> None:
        """Store compression error for error feedback.
        
        Args:
            original_gradients: Original gradients
            compressed_gradients: Compressed gradients
            client_id: Client identifier
        """
        if client_id not in self.error_memory:
            self.error_memory[client_id] = OrderedDict()
        
        for name in original_gradients.keys():
            if original_gradients[name] is None:
                continue
                
            # Calculate compression error
            error = original_gradients[name] - compressed_gradients[name]
            
            # Accumulate error
            if name in self.error_memory[client_id]:
                self.error_memory[client_id][name] += error
            else:
                self.error_memory[client_id][name] = error.clone()
    
    def decompress(
        self,
        compressed_gradients: Dict[str, torch.Tensor],
        compression_info: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Decompress gradients (if needed).
        
        Args:
            compressed_gradients: Compressed gradients
            compression_info: Information about compression
            
        Returns:
            Decompressed gradients
        """
        # For most methods, no explicit decompression needed
        # The compressed gradients are already in the correct format
        return compressed_gradients


class AsyncCommunicator:
    """Asynchronous communication handler for federated learning."""
    
    def __init__(
        self,
        max_concurrent: int = 100,
        timeout_seconds: float = 300.0,
        retry_attempts: int = 3,
    ):
        """Initialize async communicator.
        
        Args:
            max_concurrent: Maximum concurrent connections
            timeout_seconds: Communication timeout
            retry_attempts: Number of retry attempts
        """
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        
        # Communication state
        self.pending_updates = {}
        self.failed_communications = defaultdict(int)
        self.communication_stats = defaultdict(list)
        
        logger.info(f"Initialized async communicator: max_concurrent={max_concurrent}")
    
    async def send_model_update(
        self,
        client_id: str,
        update_data: Dict[str, Any],
        server_endpoint: str,
    ) -> bool:
        """Send model update to server asynchronously.
        
        Args:
            client_id: Client identifier
            update_data: Model update data
            server_endpoint: Server endpoint URL
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            # Simulate network communication
            # In practice, this would use HTTP requests or gRPC
            await asyncio.sleep(0.1)  # Simulate network latency
            
            # Serialize and compress data
            serialized_data = self._serialize_data(update_data)
            compressed_data = self._compress_data(serialized_data)
            
            # Simulate sending data
            await asyncio.sleep(len(compressed_data) / 1000000)  # Simulate transmission time
            
            # Record success
            communication_time = time.time() - start_time
            self.communication_stats[client_id].append({
                "timestamp": time.time(),
                "success": True,
                "duration": communication_time,
                "data_size": len(compressed_data),
            })
            
            logger.debug(f"Successfully sent update from {client_id} in {communication_time:.3f}s")
            return True
            
        except Exception as e:
            # Record failure
            self.failed_communications[client_id] += 1
            self.communication_stats[client_id].append({
                "timestamp": time.time(),
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e),
            })
            
            logger.error(f"Failed to send update from {client_id}: {e}")
            return False
    
    async def receive_global_model(
        self,
        client_id: str,
        server_endpoint: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Receive global model from server asynchronously.
        
        Args:
            client_id: Client identifier  
            server_endpoint: Server endpoint URL
            
        Returns:
            Global model parameters if successful, None otherwise
        """
        start_time = time.time()
        
        try:
            # Simulate receiving global model
            await asyncio.sleep(0.05)  # Simulate network latency
            
            # In practice, this would fetch from server
            # For now, return dummy model
            global_model = {"dummy_param": torch.randn(10, 10)}
            
            communication_time = time.time() - start_time
            self.communication_stats[client_id].append({
                "timestamp": time.time(),
                "success": True,
                "duration": communication_time,
                "operation": "receive_model",
            })
            
            logger.debug(f"Successfully received global model for {client_id}")
            return global_model
            
        except Exception as e:
            logger.error(f"Failed to receive global model for {client_id}: {e}")
            return None
    
    async def broadcast_global_model(
        self,
        global_model: Dict[str, torch.Tensor],
        client_ids: List[str],
    ) -> Dict[str, bool]:
        """Broadcast global model to multiple clients.
        
        Args:
            global_model: Global model parameters
            client_ids: List of client IDs to broadcast to
            
        Returns:
            Dictionary mapping client_id to success status
        """
        # Serialize model once
        serialized_model = self._serialize_data(global_model)
        compressed_model = self._compress_data(serialized_model)
        
        # Create semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def send_to_client(client_id: str) -> Tuple[str, bool]:
            async with semaphore:
                try:
                    # Simulate sending to client
                    await asyncio.sleep(0.1 + len(compressed_model) / 1000000)
                    return (client_id, True)
                except Exception as e:
                    logger.error(f"Failed to broadcast to {client_id}: {e}")
                    return (client_id, False)
        
        # Send to all clients concurrently
        tasks = [send_to_client(client_id) for client_id in client_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        success_status = {}
        for result in results:
            if isinstance(result, tuple):
                client_id, success = result
                success_status[client_id] = success
            else:
                # Exception occurred
                logger.error(f"Broadcast task failed: {result}")
        
        successful_sends = sum(success_status.values())
        logger.info(f"Broadcast completed: {successful_sends}/{len(client_ids)} successful")
        
        return success_status
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for transmission.
        
        Args:
            data: Data to serialize
            
        Returns:
            Serialized bytes
        """
        # Convert tensors to CPU and detach for serialization
        if isinstance(data, dict):
            serializable_data = {}
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    serializable_data[key] = value.detach().cpu()
                else:
                    serializable_data[key] = value
        else:
            serializable_data = data
        
        return pickle.dumps(serializable_data)
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress serialized data.
        
        Args:
            data: Serialized data
            
        Returns:
            Compressed data
        """
        return gzip.compress(data, compresslevel=6)
    
    def _decompress_data(self, compressed_data: bytes) -> bytes:
        """Decompress data.
        
        Args:
            compressed_data: Compressed data
            
        Returns:
            Decompressed data
        """
        return gzip.decompress(compressed_data)
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data.
        
        Args:
            data: Serialized data
            
        Returns:
            Deserialized object
        """
        return pickle.loads(data)
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics.
        
        Returns:
            Communication performance statistics
        """
        total_communications = sum(len(stats) for stats in self.communication_stats.values())
        successful_communications = sum(
            sum(1 for stat in stats if stat["success"])
            for stats in self.communication_stats.values()
        )
        
        if total_communications == 0:
            return {"total_communications": 0}
        
        # Calculate average metrics
        all_durations = [
            stat["duration"]
            for stats in self.communication_stats.values()
            for stat in stats
            if stat["success"]
        ]
        
        avg_duration = np.mean(all_durations) if all_durations else 0.0
        success_rate = successful_communications / total_communications
        
        return {
            "total_communications": total_communications,
            "successful_communications": successful_communications,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "failed_clients": dict(self.failed_communications),
            "active_clients": len(self.communication_stats),
        }


class BandwidthManager:
    """Manage bandwidth allocation and quality of service for federated learning."""
    
    def __init__(
        self,
        total_bandwidth_mbps: float = 100.0,
        priority_clients: Optional[List[str]] = None,
    ):
        """Initialize bandwidth manager.
        
        Args:
            total_bandwidth_mbps: Total available bandwidth in Mbps
            priority_clients: List of high-priority client IDs
        """
        self.total_bandwidth = total_bandwidth_mbps
        self.priority_clients = set(priority_clients or [])
        
        # Bandwidth allocation state
        self.client_allocations = {}
        self.current_usage = 0.0
        self.bandwidth_history = []
        
        logger.info(f"Initialized bandwidth manager: {total_bandwidth_mbps} Mbps")
    
    def allocate_bandwidth(
        self,
        client_requests: Dict[str, float],
        round_priority: bool = True,
    ) -> Dict[str, float]:
        """Allocate bandwidth to clients based on requests and priorities.
        
        Args:
            client_requests: Dictionary mapping client_id to requested bandwidth
            round_priority: Whether to round-robin among priority clients
            
        Returns:
            Dictionary mapping client_id to allocated bandwidth
        """
        if not client_requests:
            return {}
        
        total_requested = sum(client_requests.values())
        allocations = {}
        
        # If total request is within budget, allocate fully
        if total_requested <= self.total_bandwidth:
            allocations = client_requests.copy()
        else:
            # Need to prioritize and limit allocations
            remaining_bandwidth = self.total_bandwidth
            
            # First, allocate to priority clients
            priority_requests = {
                client_id: request
                for client_id, request in client_requests.items()
                if client_id in self.priority_clients
            }
            
            for client_id, request in priority_requests.items():
                # Allocate up to 50% more than fair share for priority clients
                fair_share = self.total_bandwidth / len(client_requests)
                max_allocation = min(request, fair_share * 1.5, remaining_bandwidth)
                
                allocations[client_id] = max_allocation
                remaining_bandwidth -= max_allocation
            
            # Then allocate to regular clients
            regular_clients = [
                client_id for client_id in client_requests.keys()
                if client_id not in self.priority_clients
            ]
            
            if regular_clients and remaining_bandwidth > 0:
                per_client_allocation = remaining_bandwidth / len(regular_clients)
                
                for client_id in regular_clients:
                    requested = client_requests[client_id]
                    allocated = min(requested, per_client_allocation)
                    allocations[client_id] = allocated
        
        # Update state
        self.client_allocations = allocations
        self.current_usage = sum(allocations.values())
        
        # Record allocation
        self.bandwidth_history.append({
            "timestamp": time.time(),
            "total_requested": total_requested,
            "total_allocated": self.current_usage,
            "utilization": self.current_usage / self.total_bandwidth,
            "num_clients": len(client_requests),
        })
        
        logger.info(
            f"Allocated bandwidth: {self.current_usage:.1f}/{self.total_bandwidth:.1f} Mbps "
            f"({self.current_usage/self.total_bandwidth*100:.1f}% utilization)"
        )
        
        return allocations
    
    def estimate_transmission_time(
        self,
        client_id: str,
        data_size_mb: float,
    ) -> float:
        """Estimate transmission time for client data.
        
        Args:
            client_id: Client identifier
            data_size_mb: Data size in megabytes
            
        Returns:
            Estimated transmission time in seconds
        """
        allocated_bandwidth = self.client_allocations.get(client_id, 1.0)
        
        # Account for protocol overhead and network conditions
        effective_bandwidth = allocated_bandwidth * 0.8  # 80% efficiency
        
        if effective_bandwidth <= 0:
            return float("inf")
        
        transmission_time = data_size_mb / effective_bandwidth
        return transmission_time
    
    def get_bandwidth_stats(self) -> Dict[str, Any]:
        """Get bandwidth usage statistics.
        
        Returns:
            Bandwidth usage statistics
        """
        if not self.bandwidth_history:
            return {"no_data": True}
        
        recent_history = self.bandwidth_history[-10:]  # Last 10 allocations
        
        return {
            "total_bandwidth": self.total_bandwidth,
            "current_usage": self.current_usage,
            "current_utilization": self.current_usage / self.total_bandwidth,
            "active_clients": len(self.client_allocations),
            "priority_clients": len(self.priority_clients),
            "avg_utilization": np.mean([h["utilization"] for h in recent_history]),
            "allocation_history": recent_history,
        }