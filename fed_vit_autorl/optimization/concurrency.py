"""Concurrency and parallel processing optimization for federated learning."""

import asyncio
import concurrent.futures
import threading
import time
import logging
import queue
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from dataclasses import dataclass
from enum import Enum
import functools
from collections import defaultdict

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Async task with metadata."""
    task_id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    timeout: Optional[float] = None
    retries: int = 0
    max_retries: int = 0
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class ConcurrentProcessor:
    """High-performance concurrent processor for federated learning tasks."""

    def __init__(
        self,
        max_workers: int = None,
        thread_pool_size: int = None,
        process_pool_size: int = None,
        queue_size: int = 1000,
        enable_priority_queue: bool = True,
        default_timeout: float = 300.0,
        max_retries: int = 3,
    ):
        """Initialize concurrent processor.

        Args:
            max_workers: Maximum number of worker threads
            thread_pool_size: Size of thread pool executor
            process_pool_size: Size of process pool executor
            queue_size: Maximum task queue size
            enable_priority_queue: Whether to enable priority-based task scheduling
            default_timeout: Default task timeout in seconds
            max_retries: Default maximum retries for tasks
        """
        import os

        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.thread_pool_size = thread_pool_size or self.max_workers
        self.process_pool_size = process_pool_size or (os.cpu_count() or 1)
        self.queue_size = queue_size
        self.enable_priority_queue = enable_priority_queue
        self.default_timeout = default_timeout
        self.max_retries = max_retries

        # Task management
        self.tasks: Dict[str, Task] = {}
        if enable_priority_queue:
            self.task_queue = queue.PriorityQueue(maxsize=queue_size)
        else:
            self.task_queue = queue.Queue(maxsize=queue_size)

        # Executors
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.thread_pool_size,
            thread_name_prefix="fed_vit_thread"
        )
        self.process_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.process_pool_size
        )

        # Worker threads
        self.workers: List[threading.Thread] = []
        self.running = False
        self.shutdown_event = threading.Event()

        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0,
            "queue_high_watermark": 0,
        }

        self._lock = threading.Lock()

        logger.info(f"Initialized concurrent processor with {self.max_workers} max workers")

        # Start worker threads
        self.start()

    def start(self) -> None:
        """Start worker threads."""
        if self.running:
            return

        self.running = True
        self.shutdown_event.clear()

        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"fed_vit_worker_{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"Started {len(self.workers)} worker threads")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop all worker threads.

        Args:
            timeout: Timeout for graceful shutdown
        """
        if not self.running:
            return

        logger.info("Stopping concurrent processor...")

        self.running = False
        self.shutdown_event.set()

        # Add poison pills to wake up workers
        for _ in range(len(self.workers)):
            try:
                if self.enable_priority_queue:
                    self.task_queue.put((float('inf'), None), timeout=1.0)
                else:
                    self.task_queue.put(None, timeout=1.0)
            except queue.Full:
                pass

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout/len(self.workers))
            if worker.is_alive():
                logger.warning(f"Worker {worker.name} did not shut down gracefully")

        # Shutdown executors
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)

        self.workers.clear()
        logger.info("Concurrent processor stopped")

    def _worker_loop(self) -> None:
        """Main worker thread loop."""
        while self.running:
            try:
                # Get next task
                if self.enable_priority_queue:
                    priority, task = self.task_queue.get(timeout=1.0)
                    if task is None:  # Poison pill
                        break
                else:
                    task = self.task_queue.get(timeout=1.0)
                    if task is None:  # Poison pill
                        break

                # Execute task
                self._execute_task(task)

                # Mark task as done
                self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    def _execute_task(self, task: Task) -> None:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        try:
            # Set timeout
            if task.timeout:
                timeout = task.timeout
            else:
                timeout = self.default_timeout

            # Execute with timeout
            if asyncio.iscoroutinefunction(task.func):
                # Async function
                result = asyncio.run(
                    asyncio.wait_for(task.func(*task.args, **task.kwargs), timeout=timeout)
                )
            else:
                # Sync function - use thread executor with timeout
                future = self.thread_executor.submit(task.func, *task.args, **task.kwargs)
                result = future.result(timeout=timeout)

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()

            # Update statistics
            with self._lock:
                self.stats["tasks_completed"] += 1
                execution_time = task.completed_at - task.started_at
                self.stats["total_execution_time"] += execution_time
                self.stats["avg_execution_time"] = (
                    self.stats["total_execution_time"] / self.stats["tasks_completed"]
                )

            logger.debug(f"Task {task.task_id} completed in {execution_time:.3f}s")

        except Exception as e:
            task.error = e
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()

            # Retry logic
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                task.started_at = None
                task.error = None

                # Re-queue for retry
                self.submit_task(
                    task.task_id + f"_retry_{task.retries}",
                    task.func,
                    *task.args,
                    priority=task.priority,
                    timeout=task.timeout,
                    max_retries=task.max_retries - task.retries,
                    **task.kwargs
                )

                logger.warning(f"Retrying task {task.task_id} ({task.retries}/{task.max_retries})")
            else:
                with self._lock:
                    self.stats["tasks_failed"] += 1

                logger.error(f"Task {task.task_id} failed after {task.retries} retries: {e}")

    def submit_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        priority: int = 0,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> str:
        """Submit a task for execution.

        Args:
            task_id: Unique task identifier
            func: Function to execute
            *args: Function arguments
            priority: Task priority (higher = more priority)
            timeout: Task timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Function keyword arguments

        Returns:
            Task ID
        """
        if not self.running:
            raise RuntimeError("Processor is not running")

        # Create task
        task = Task(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries or self.max_retries,
        )

        # Store task
        with self._lock:
            self.tasks[task_id] = task
            self.stats["tasks_submitted"] += 1

            # Update queue high watermark
            current_queue_size = self.task_queue.qsize()
            if current_queue_size > self.stats["queue_high_watermark"]:
                self.stats["queue_high_watermark"] = current_queue_size

        # Queue task
        try:
            if self.enable_priority_queue:
                # Higher priority tasks get lower priority number for queue ordering
                queue_priority = -priority
                self.task_queue.put((queue_priority, task), timeout=5.0)
            else:
                self.task_queue.put(task, timeout=5.0)

            logger.debug(f"Submitted task {task_id} with priority {priority}")
            return task_id

        except queue.Full:
            # Remove from tasks dict if queueing failed
            with self._lock:
                self.tasks.pop(task_id, None)
                self.stats["tasks_submitted"] -= 1

            raise RuntimeError(f"Task queue is full, cannot submit task {task_id}")

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status.

        Args:
            task_id: Task identifier

        Returns:
            Task status or None if not found
        """
        task = self.tasks.get(task_id)
        return task.status if task else None

    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result, blocking until completion.

        Args:
            task_id: Task identifier
            timeout: Maximum time to wait for result

        Returns:
            Task result

        Raises:
            KeyError: If task not found
            TimeoutError: If timeout exceeded
            Exception: If task failed
        """
        task = self.tasks.get(task_id)
        if not task:
            raise KeyError(f"Task {task_id} not found")

        start_time = time.time()

        # Wait for completion
        while task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")

            time.sleep(0.01)  # Small sleep to avoid busy waiting

        if task.status == TaskStatus.FAILED:
            raise task.error
        elif task.status == TaskStatus.CANCELLED:
            raise RuntimeError(f"Task {task_id} was cancelled")

        return task.result

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.

        Args:
            task_id: Task identifier

        Returns:
            True if task was cancelled, False otherwise
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            with self._lock:
                self.stats["tasks_cancelled"] += 1

            logger.info(f"Cancelled task {task_id}")
            return True

        return False

    def wait_for_all_tasks(self, timeout: Optional[float] = None) -> None:
        """Wait for all tasks to complete.

        Args:
            timeout: Maximum time to wait
        """
        start_time = time.time()

        while True:
            with self._lock:
                pending_tasks = [
                    task for task in self.tasks.values()
                    if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
                ]

            if not pending_tasks:
                break

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for {len(pending_tasks)} tasks")

            time.sleep(0.1)

        logger.info("All tasks completed")

    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            stats = self.stats.copy()

        stats.update({
            "running": self.running,
            "active_workers": len([w for w in self.workers if w.is_alive()]),
            "total_workers": len(self.workers),
            "queue_size": self.task_queue.qsize(),
            "total_tasks": len(self.tasks),
            "thread_pool_size": self.thread_pool_size,
            "process_pool_size": self.process_pool_size,
        })

        return stats

    def cleanup_completed_tasks(self, max_age: float = 3600.0) -> int:
        """Clean up old completed tasks.

        Args:
            max_age: Maximum age of completed tasks to keep (seconds)

        Returns:
            Number of tasks cleaned up
        """
        current_time = time.time()
        cleaned_up = 0

        with self._lock:
            tasks_to_remove = []

            for task_id, task in self.tasks.items():
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    task.completed_at and
                    (current_time - task.completed_at) > max_age):
                    tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del self.tasks[task_id]
                cleaned_up += 1

        if cleaned_up > 0:
            logger.info(f"Cleaned up {cleaned_up} old tasks")

        return cleaned_up

    def __del__(self):
        """Cleanup when processor is destroyed."""
        try:
            self.stop()
        except Exception:
            pass  # Ignore cleanup errors during destruction


class AsyncTaskManager:
    """Async task manager for coroutine-based operations."""

    def __init__(self, max_concurrent_tasks: int = 100):
        """Initialize async task manager.

        Args:
            max_concurrent_tasks: Maximum number of concurrent async tasks
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, Exception] = {}

        self.stats = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
        }

        logger.info(f"Initialized async task manager with {max_concurrent_tasks} max concurrent tasks")

    async def submit_async_task(
        self,
        task_id: str,
        coro: Awaitable,
        timeout: Optional[float] = None,
    ) -> str:
        """Submit an async task.

        Args:
            task_id: Unique task identifier
            coro: Coroutine to execute
            timeout: Task timeout in seconds

        Returns:
            Task ID
        """
        async def _task_wrapper():
            async with self.semaphore:
                try:
                    if timeout:
                        result = await asyncio.wait_for(coro, timeout=timeout)
                    else:
                        result = await coro

                    self.completed_tasks[task_id] = result
                    self.stats["tasks_completed"] += 1
                    return result

                except Exception as e:
                    self.failed_tasks[task_id] = e
                    self.stats["tasks_failed"] += 1
                    raise
                finally:
                    # Clean up active task reference
                    self.active_tasks.pop(task_id, None)

        # Create and store task
        task = asyncio.create_task(_task_wrapper())
        self.active_tasks[task_id] = task
        self.stats["tasks_created"] += 1

        return task_id

    async def wait_for_task(self, task_id: str) -> Any:
        """Wait for a specific task to complete.

        Args:
            task_id: Task identifier

        Returns:
            Task result
        """
        # Check if already completed
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]

        if task_id in self.failed_tasks:
            raise self.failed_tasks[task_id]

        # Wait for active task
        if task_id in self.active_tasks:
            return await self.active_tasks[task_id]

        raise KeyError(f"Task {task_id} not found")

    async def wait_for_all(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for all active tasks to complete.

        Args:
            timeout: Maximum time to wait

        Returns:
            Dictionary of task results
        """
        if not self.active_tasks:
            return self.completed_tasks.copy()

        try:
            # Wait for all active tasks
            if timeout:
                await asyncio.wait_for(
                    asyncio.gather(*self.active_tasks.values(), return_exceptions=True),
                    timeout=timeout
                )
            else:
                await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)

        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in self.active_tasks.values():
                if not task.done():
                    task.cancel()
                    self.stats["tasks_cancelled"] += 1

        return self.completed_tasks.copy()

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task.

        Args:
            task_id: Task identifier

        Returns:
            True if task was cancelled
        """
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            if not task.done():
                task.cancel()
                self.stats["tasks_cancelled"] += 1
                return True

        return False

    def cancel_all(self) -> int:
        """Cancel all active tasks.

        Returns:
            Number of tasks cancelled
        """
        cancelled = 0

        for task in self.active_tasks.values():
            if not task.done():
                task.cancel()
                cancelled += 1

        self.stats["tasks_cancelled"] += cancelled
        return cancelled

    def get_statistics(self) -> Dict[str, Any]:
        """Get async task manager statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
        }
