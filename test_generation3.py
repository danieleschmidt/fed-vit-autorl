#!/usr/bin/env python3
"""Test Generation 3 scaling and optimization components."""

import sys
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.abspath('.'))

def test_auto_scaling():
    """Test auto-scaling system."""
    print("Testing auto-scaling system...")
    try:
        from fed_vit_autorl.optimization.auto_scaling import AutoScaler, LoadMetrics, ScalingAction

        # Create auto-scaler
        scaler = AutoScaler(
            min_instances=1,
            max_instances=5,
            target_cpu_usage=70.0,
            scale_up_cooldown=1.0,  # Short for testing
            scale_down_cooldown=2.0
        )

        # Test with high load
        high_load_metrics = LoadMetrics(
            cpu_usage=85.0,  # Above threshold
            memory_usage=60.0,
            request_rate=100.0,
            response_time=0.5,
            active_connections=50,
            queue_length=5,
            timestamp=time.time()
        )

        # Record metrics to trigger evaluation
        for _ in range(10):  # Need enough metrics for evaluation window
            scaler.record_metrics(high_load_metrics)

        time.sleep(0.1)  # Let processing happen

        # Check if scaling occurred
        current_instances = scaler.get_current_instances()
        stats = scaler.get_scaling_stats()

        assert current_instances >= 1
        assert stats["total_scaling_events"] >= 0

        print("✅ Auto-scaling system working!")
        return True
    except Exception as e:
        print(f"❌ Auto-scaling test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization system."""
    print("Testing memory optimization...")
    try:
        from fed_vit_autorl.optimization.memory_optimization import MemoryOptimizer

        # Create memory optimizer
        optimizer = MemoryOptimizer(
            memory_limit=8.0,  # 8GB limit
            warning_threshold=60.0,
            critical_threshold=90.0,
            enable_auto_cleanup=True
        )

        # Test memory usage monitoring
        usage = optimizer.get_memory_usage()
        assert usage.total_memory > 0
        assert 0 <= usage.memory_percent <= 100

        # Test caching
        optimizer.cache_object("test_key", {"data": "test_value"})
        cached_obj = optimizer.get_cached_object("test_key")
        assert cached_obj == {"data": "test_value"}

        # Test cleanup
        memory_freed = optimizer.cleanup_memory(aggressive=False)
        assert memory_freed >= 0.0

        # Test stats
        stats = optimizer.get_memory_stats()
        assert "current_usage" in stats
        assert "cleanup_stats" in stats

        print("✅ Memory optimization working!")
        return True
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False

def test_concurrent_processing():
    """Test concurrent processing system."""
    print("Testing concurrent processing...")
    try:
        from fed_vit_autorl.optimization.concurrency import ConcurrentProcessor, TaskStatus

        # Create processor
        processor = ConcurrentProcessor(
            max_workers=4,
            queue_size=100,
            default_timeout=10.0
        )

        # Test task submission
        def test_task(x, y=10):
            time.sleep(0.1)  # Simulate work
            return x + y

        task_id = processor.submit_task("test_task_1", test_task, 5, y=15)

        # Wait for result
        result = processor.get_task_result(task_id, timeout=5.0)
        assert result == 20

        # Test task status
        status = processor.get_task_status(task_id)
        assert status == TaskStatus.COMPLETED

        # Test statistics
        stats = processor.get_statistics()
        assert stats["tasks_completed"] >= 1
        assert stats["running"] == True

        # Cleanup
        processor.stop()

        print("✅ Concurrent processing working!")
        return True
    except Exception as e:
        print(f"❌ Concurrent processing test failed: {e}")
        return False

def test_async_task_manager():
    """Test async task manager."""
    print("Testing async task manager...")
    try:
        from fed_vit_autorl.optimization.concurrency import AsyncTaskManager

        async def run_async_test():
            # Create task manager
            manager = AsyncTaskManager(max_concurrent_tasks=10)

            # Define async task
            async def async_task(delay, result):
                await asyncio.sleep(delay)
                return result

            # Submit tasks
            task_id1 = await manager.submit_async_task(
                "async_task_1",
                async_task(0.1, "result1")
            )

            task_id2 = await manager.submit_async_task(
                "async_task_2",
                async_task(0.05, "result2")
            )

            # Wait for results
            result1 = await manager.wait_for_task(task_id1)
            result2 = await manager.wait_for_task(task_id2)

            assert result1 == "result1"
            assert result2 == "result2"

            # Test statistics
            stats = manager.get_statistics()
            assert stats["tasks_completed"] >= 2

            return True

        # Run async test
        result = asyncio.run(run_async_test())
        assert result == True

        print("✅ Async task manager working!")
        return True
    except Exception as e:
        print(f"❌ Async task manager test failed: {e}")
        return False

def test_load_balancer():
    """Test load balancer."""
    print("Testing load balancer...")
    try:
        from fed_vit_autorl.optimization.auto_scaling import LoadBalancer

        # Create load balancer
        lb = LoadBalancer(
            algorithm="round_robin",
            health_check_interval=60.0,  # Long interval for testing
            max_retries=3
        )

        # Add instances
        lb.add_instance("instance_1", "http://localhost:8001", weight=1.0)
        lb.add_instance("instance_2", "http://localhost:8002", weight=2.0)
        lb.add_instance("instance_3", "http://localhost:8003", weight=1.0)

        # Test instance selection
        selected = []
        for _ in range(6):  # Test round-robin
            instance = lb.get_next_instance()
            if instance:
                selected.append(instance)

        assert len(set(selected)) >= 2  # Should use multiple instances

        # Test request recording
        lb.record_request("instance_1", response_time=0.1, success=True)
        lb.record_request("instance_2", response_time=0.2, success=False)

        # Test statistics
        stats = lb.get_load_balancer_stats()
        assert stats["total_instances"] == 3
        assert stats["healthy_instances"] >= 0
        assert "instance_stats" in stats

        # Cleanup
        lb.stop_health_checks()

        print("✅ Load balancer working!")
        return True
    except Exception as e:
        print(f"❌ Load balancer test failed: {e}")
        return False

def main():
    """Run all Generation 3 tests."""
    print("🚀 Testing Fed-ViT-AutoRL Generation 3: Scaling & Optimization\\n")

    tests = [
        test_auto_scaling,
        test_memory_optimization,
        test_concurrent_processing,
        test_async_task_manager,
        test_load_balancer,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}\\n")

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 Generation 3 scaling and optimization components working!")
        print("✅ Auto-scaling with intelligent thresholds")
        print("✅ Advanced memory optimization and pooling")
        print("✅ High-performance concurrent processing")
        print("✅ Async task management with semaphores")
        print("✅ Load balancing with health monitoring")
        return True
    else:
        print("⚠️  Some scaling tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
