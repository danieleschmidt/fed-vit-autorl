#!/usr/bin/env python3
"""Simple test for Generation 3 scaling components without external deps."""

import sys
import os
import time
import asyncio
sys.path.insert(0, os.path.abspath('.'))

def test_auto_scaling():
    """Test auto-scaling system core functionality."""
    print("Testing auto-scaling system...")
    try:
        from fed_vit_autorl.optimization.auto_scaling import AutoScaler, LoadMetrics, ScalingAction

        # Create auto-scaler
        scaler = AutoScaler(
            min_instances=1,
            max_instances=5,
            target_cpu_usage=70.0
        )

        # Test basic functionality
        current_instances = scaler.get_current_instances()
        assert current_instances == 1  # Should start with min_instances

        # Test stats
        stats = scaler.get_scaling_stats()
        assert "total_scaling_events" in stats
        assert stats["current_instances"] == 1

        print("✅ Auto-scaling system core working!")
        return True
    except Exception as e:
        print(f"❌ Auto-scaling test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization core functionality."""
    print("Testing memory optimization...")
    try:
        from fed_vit_autorl.optimization.memory_optimization import MemoryOptimizer

        # Create memory optimizer
        optimizer = MemoryOptimizer(
            memory_limit=8.0,
            warning_threshold=60.0,
            enable_auto_cleanup=True
        )

        # Test caching functionality
        optimizer.cache_object("test_key", {"data": "test_value"})
        cached_obj = optimizer.get_cached_object("test_key")
        assert cached_obj == {"data": "test_value"}

        # Test cleanup
        optimizer.clear_caches()
        cleared_obj = optimizer.get_cached_object("test_key")
        assert cleared_obj is None

        # Test memory pools
        optimizer.add_to_pool("test_pool", {"pool_data": "value"})
        pooled_obj = optimizer.get_from_pool("test_pool")
        assert pooled_obj == {"pool_data": "value"}

        print("✅ Memory optimization core working!")
        return True
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False

def test_concurrent_processing():
    """Test concurrent processing core functionality."""
    print("Testing concurrent processing...")
    try:
        from fed_vit_autorl.optimization.concurrency import ConcurrentProcessor, Task, TaskStatus

        # Test task creation
        task = Task(
            task_id="test_task",
            func=lambda x: x * 2,
            args=(5,),
            kwargs={},
            priority=1
        )

        assert task.task_id == "test_task"
        assert task.status == TaskStatus.PENDING
        assert task.priority == 1

        # Test processor creation (without starting workers)
        processor = ConcurrentProcessor(max_workers=2)

        # Test statistics
        stats = processor.get_statistics()
        assert "tasks_submitted" in stats
        assert "running" in stats

        # Stop processor
        processor.stop()

        print("✅ Concurrent processing core working!")
        return True
    except Exception as e:
        print(f"❌ Concurrent processing test failed: {e}")
        return False

def test_async_task_manager():
    """Test async task manager core functionality."""
    print("Testing async task manager...")
    try:
        from fed_vit_autorl.optimization.concurrency import AsyncTaskManager

        async def run_async_test():
            # Create task manager
            manager = AsyncTaskManager(max_concurrent_tasks=10)

            # Test statistics
            stats = manager.get_statistics()
            assert "tasks_created" in stats
            assert stats["max_concurrent_tasks"] == 10

            return True

        # Run async test
        result = asyncio.run(run_async_test())
        assert result == True

        print("✅ Async task manager core working!")
        return True
    except Exception as e:
        print(f"❌ Async task manager test failed: {e}")
        return False

def test_load_balancer():
    """Test load balancer core functionality."""
    print("Testing load balancer...")
    try:
        from fed_vit_autorl.optimization.auto_scaling import LoadBalancer

        # Create load balancer (without health checks)
        lb = LoadBalancer(
            algorithm="round_robin",
            health_check_interval=3600.0  # Very long interval
        )

        # Add instances
        lb.add_instance("instance_1", "http://localhost:8001", weight=1.0)
        lb.add_instance("instance_2", "http://localhost:8002", weight=2.0)

        # Test instance selection
        instance1 = lb.get_next_instance()
        instance2 = lb.get_next_instance()

        assert instance1 is not None
        assert instance2 is not None

        # Test removal
        lb.remove_instance("instance_1")

        # Test statistics
        stats = lb.get_load_balancer_stats()
        assert stats["total_instances"] == 1
        assert "algorithm" in stats

        print("✅ Load balancer core working!")
        return True
    except Exception as e:
        print(f"❌ Load balancer test failed: {e}")
        return False

def main():
    """Run simple Generation 3 tests."""
    print("🚀 Testing Fed-ViT-AutoRL Generation 3: Core Scaling Components\\n")

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
        print("🎉 Generation 3 core scaling components working!")
        return True
    else:
        print("⚠️  Some scaling tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
