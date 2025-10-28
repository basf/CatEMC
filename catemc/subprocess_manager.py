#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Async subprocess manager with GPU access retry logic"""
import asyncio
import random
from typing import List, Dict


class SubprocessManager:
    """
    Async subprocess manager that spawns and monitors subprocesses with retry logic.

    This manager handles GPU access conflicts by retrying failed processes with
    exponential backoff. It's designed to prevent the pthread error that occurs when
    multiple processes try to access the GPU simultaneously.

    Parameters
    ----------
    max_retries: int, default 3
        Maximum number of retry attempts for a failed subprocess.

    initial_delay: float, default 0.5
        Initial delay between process spawns in seconds.

    retry_delay: float, default 2.0
        Base delay for retries in seconds (with exponential backoff).

    verbose: bool, default True
        Whether to print detailed logging information.
    """

    def __init__(self, max_retries=3, initial_delay=0.5, retry_delay=2.0, verbose=True):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.retry_delay = retry_delay
        self.verbose = verbose
        self._semaphore_dict = {}  # Per-GPU semaphores

    def _get_semaphore(self, gpu_id, max_concurrent):
        """Get or create a semaphore for a specific GPU."""
        key = (gpu_id, max_concurrent)
        if key not in self._semaphore_dict:
            self._semaphore_dict[key] = asyncio.Semaphore(max_concurrent)
        return self._semaphore_dict[key]

    async def _run_subprocess_with_retry(self, cmd: List[str], gpu_id: int,
                                         max_concurrent: int, job_id: str) -> bool:
        """
        Run a single subprocess with retry logic.

        Parameters
        ----------
        cmd: list of str
            Command and arguments to execute.
        gpu_id: int
            GPU device ID for this job.
        max_concurrent: int
            Maximum concurrent processes per GPU.
        job_id: str
            Identifier for logging purposes.

        Returns
        -------
        success: bool
            True if subprocess completed successfully, False otherwise.
        """
        semaphore = self._get_semaphore(gpu_id, max_concurrent)

        for attempt in range(self.max_retries + 1):
            async with semaphore:
                if self.verbose and attempt > 0:
                    print(f"[SubprocessManager] Retry {attempt}/{self.max_retries} for job {job_id}")

                # Add small random delay to prevent simultaneous GPU access
                await asyncio.sleep(self.initial_delay + random.random() * 0.5)

                try:
                    # Create subprocess
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )

                    # Wait for completion
                    stdout, stderr = await process.communicate()

                    # Check for success
                    if process.returncode == 0:
                        if self.verbose:
                            print(f"[SubprocessManager] Job {job_id} completed successfully")
                        return True
                    else:
                        stderr_str = stderr.decode() if stderr else ""
                        if self.verbose:
                            print(f"[SubprocessManager] Job {job_id} failed with return code {process.returncode}")

                        # Check for pthread error
                        if "pthread_create()" in stderr_str or "Resource temporarily unavailable" in stderr_str:
                            if attempt < self.max_retries:
                                # Exponential backoff with jitter
                                delay = self.retry_delay * (2 ** attempt) + random.random()
                                if self.verbose:
                                    print(f"[SubprocessManager] Detected GPU access error. Retrying in {delay:.2f}s...")
                                await asyncio.sleep(delay)
                                continue

                        # Non-retryable error or max retries reached
                        if self.verbose:
                            print(f"[SubprocessManager] Job {job_id} failed after {attempt + 1} attempts")
                            if stderr_str:
                                print(f"[SubprocessManager] Error output: {stderr_str[:500]}")
                        return False

                except Exception as e:
                    if self.verbose:
                        print(f"[SubprocessManager] Exception for job {job_id}: {e}")
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2 ** attempt) + random.random()
                        await asyncio.sleep(delay)
                        continue
                    return False

        return False

    async def run_batch(self, commands: List[Dict]) -> List[bool]:
        """
        Run a batch of subprocesses concurrently with GPU-aware scheduling.

        Parameters
        ----------
        commands: list of dict
            Each dict contains:
            - 'cmd': list of str, the command to execute
            - 'gpu_id': int, the GPU device ID
            - 'max_concurrent': int, max concurrent processes per GPU
            - 'job_id': str, identifier for this job

        Returns
        -------
        results: list of bool
            Success status for each command.
        """
        tasks = []
        for cmd_info in commands:
            task = asyncio.create_task(
                self._run_subprocess_with_retry(
                    cmd=cmd_info['cmd'],
                    gpu_id=cmd_info['gpu_id'],
                    max_concurrent=cmd_info['max_concurrent'],
                    job_id=cmd_info['job_id']
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    def run_batch_sync(self, commands: List[Dict]) -> List[bool]:
        """
        Synchronous wrapper for run_batch.

        This allows the manager to be used in non-async code.

        Parameters
        ----------
        commands: list of dict
            Each dict contains:
            - 'cmd': list of str, the command to execute
            - 'gpu_id': int, the GPU device ID
            - 'max_concurrent': int, max concurrent processes per GPU
            - 'job_id': str, identifier for this job

        Returns
        -------
        results: list of bool
            Success status for each command.
        """
        # Always use asyncio.run in a synchronous context
        # If this is called from async code, the caller should use run_batch directly
        return asyncio.run(self.run_batch(commands))


def run_subprocess_with_retry(python_path: str, script_path: str, args: str,
                               gpu_id: int, max_concurrent: int, job_id: str,
                               max_retries: int = 3, verbose: bool = True) -> bool:
    """
    Convenience function to run a single subprocess with retry logic.

    Parameters
    ----------
    python_path: str
        Path to the Python interpreter.
    script_path: str
        Path to the script to execute.
    args: str
        Command line arguments for the script.
    gpu_id: int
        GPU device ID for this job.
    max_concurrent: int
        Maximum concurrent processes per GPU.
    job_id: str
        Identifier for this job.
    max_retries: int, default 3
        Maximum number of retry attempts.
    verbose: bool, default True
        Whether to print detailed logging.

    Returns
    -------
    success: bool
        True if subprocess completed successfully, False otherwise.
    """
    cmd = [python_path, script_path] + args.split()
    manager = SubprocessManager(max_retries=max_retries, verbose=verbose)
    commands = [{
        'cmd': cmd,
        'gpu_id': gpu_id,
        'max_concurrent': max_concurrent,
        'job_id': job_id
    }]
    results = manager.run_batch_sync(commands)
    return results[0]
