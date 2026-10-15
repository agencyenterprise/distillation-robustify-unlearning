import torch
import os
import multiprocessing as mp
import subprocess
import sys
from multiprocessing import Queue, Lock
import time

def launch_in_parallel_one_per_four_gpus(experiment_list, experiment_fn):
    """Launch experiments in parallel using accelerate launch, each using 4 GPUs"""
    mp.set_start_method('spawn')
    
    # Define GPU groups
    gpu_groups = {
        0: [0, 1, 2, 3],  # First group
        1: [4, 5, 6, 7]   # Second group
    }
    
    # Create a queue of experiments
    experiment_queue = Queue()
    for exp in experiment_list:
        experiment_queue.put(exp)
    
    # Track active processes for each GPU group
    group_processes = {}
    group_lock = Lock()
    
    def start_experiment_on_group(group_id):
        if experiment_queue.empty():
            return None
        
        args = experiment_queue.get()
        print(f"Starting on GPU group {group_id} (GPUs {gpu_groups[group_id]}): args={args}")
        
        # Build the accelerate launch command
        cmd = [
            "accelerate", "launch",
            "--multi_gpu",
            "--num_processes", "4",
            "--mixed_precision", "bf16",
            "--main_process_port", str(29500 + group_id),  # Different port per group
            sys.argv[0],  # Use the current script
        ]
        
        # Add experiment arguments
        setup_id, alpha, beta, seed = args
        cmd.extend([
            "--setup", setup_id,
            "--alpha", str(alpha),
            "--beta", str(beta),
        ])
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        # Set environment variables including CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_groups[group_id]))
        
        # Start the process with accelerate launch
        p = subprocess.Popen(cmd, env=env)
        return p
    
    # Initially start one experiment per GPU group
    for group_id in gpu_groups:
        p = start_experiment_on_group(group_id)
        if p:
            group_processes[group_id] = p
    
    time.sleep(1)
    
    # Keep checking for completed processes and start new ones
    while not experiment_queue.empty() or group_processes:
        # Check each GPU group
        with group_lock:
            for group_id in list(group_processes.keys()):
                process = group_processes[group_id]
                if process.poll() is not None:  # Process completed
                    # Remove it
                    process.wait()
                    del group_processes[group_id]
                    
                    # Start new experiment on this GPU group if there are any left
                    if not experiment_queue.empty():
                        new_process = start_experiment_on_group(group_id)
                        if new_process:
                            group_processes[group_id] = new_process
        
        # Small sleep to prevent busy waiting
        time.sleep(1)
    
    print("All experiments completed!")

def gpu_groups(gpus_per_run, visible=None):
    """Split the visible GPUs (CUDA_VISIBLE_DEVICES, else all) into disjoint groups of gpus_per_run."""
    if visible is None:
        env = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible = env.split(",") if env else [str(i) for i in range(torch.cuda.device_count())]
    groups = [visible[i:i + gpus_per_run] for i in range(0, len(visible) - gpus_per_run + 1, gpus_per_run)]
    if not groups:
        raise RuntimeError(f"Need {gpus_per_run} GPUs per run, have {len(visible)}: {visible}")
    return groups


def launch_in_parallel_accelerate_groups(cli_args_list, script, gpus_per_run, port_base=29500):
    """
    Generalises launch_in_parallel_one_per_four_gpus: each run is
    `accelerate launch --multi_gpu --num_processes gpus_per_run <script> <args>` on its own group of GPUs,
    queued until all runs are done. Returns the number of failed runs.
    """
    groups = gpu_groups(gpus_per_run)
    queue = list(cli_args_list)
    running, failed = {}, 0
    print(f"{len(queue)} runs on GPU groups {groups}", flush=True)
    while queue or running:
        for g, group in enumerate(groups):
            if g in running and running[g][0].poll() is not None:
                proc, args = running.pop(g)
                if proc.returncode != 0:
                    failed += 1
                    print(f"FAILED (exit {proc.returncode}) on GPUs {group}: {args}", flush=True)
            if g not in running and queue:
                args = queue.pop(0)
                multi = ["--multi_gpu"] if gpus_per_run > 1 else []
                cmd = [sys.executable, "-m", "accelerate.commands.launch", *multi, "--num_processes", str(gpus_per_run),
                       "--mixed_precision", "bf16", "--main_process_port", str(port_base + g), script, *args]
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(group)}
                print(f"Starting on GPUs {group}: {args}", flush=True)
                running[g] = (subprocess.Popen(cmd, env=env), args)
        time.sleep(5)
    print(f"All runs completed; {failed} failed.", flush=True)
    return failed


def launch_in_parallel_one_per_gpu(experiment_list, experiment_fn):
        mp.set_start_method('spawn')
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPU devices found!")
        
        # Create a queue of experiments
        experiment_queue = Queue()
        for exp in experiment_list:
            experiment_queue.put(exp)
        
        # Track active processes for each GPU
        gpu_processes = {}
        gpu_lock = Lock()
        time.sleep(1) # need this
        def start_experiment_on_gpu(gpu_id):
            if experiment_queue.empty():
                return None
            
            args = experiment_queue.get()
            print(f"Starting on GPU {gpu_id}: args={args}")

            args = (args, gpu_id)
            p = mp.Process(target=experiment_fn, args=args)
            p.start()
            return p
        
        # Initially start one experiment per GPU
        for gpu_id in range(num_gpus):
            p = start_experiment_on_gpu(gpu_id)
            if p:
                gpu_processes[gpu_id] = p
        
        time.sleep(1)
        # Keep checking for completed processes and start new ones
        while not experiment_queue.empty() or gpu_processes:
            # Check each GPU
            with gpu_lock:
                for gpu_id in list(gpu_processes.keys()):
                    process = gpu_processes[gpu_id]
                    if not process.is_alive():
                        # Process completed, remove it
                        process.join()
                        del gpu_processes[gpu_id]
                        
                        # Start new experiment on this GPU if there are any left
                        if not experiment_queue.empty():
                            new_process = start_experiment_on_gpu(gpu_id)
                            if new_process:
                                gpu_processes[gpu_id] = new_process
            
            # Small sleep to prevent busy waiting
            time.sleep(1)
        
        print("All experiments completed!")

import os
class ParallelWrapper:
    def __init__(self, function):
        self.function = function
    
    def __call__(self, args, gpu_id):
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        else:
            print(f"CUDA not available in subprocess with gpu_id={gpu_id}")
        
        return self.function(*args)

def get_parallel_launch_wrapper(function):
    return ParallelWrapper(function)