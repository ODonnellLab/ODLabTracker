#!/usr/bin/env python3
"""
Run FastTrack.py on all .avi files in parallel
Usage: python run_fasttrack_parallel.py [parent_directory] -c [config_file] -j [num_jobs]
Example: python run_fasttrack_parallel.py ./data -c configs/myconfig.txt -j 4
"""

import argparse
import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial


def process_single_file(avi_file, config_file=None):
    """Process a single .avi file with FastTrack.py"""
    try:
        # Build command with config file if provided
        cmd = ["python", "FastTrack.py", "-f", str(avi_file)]
        if config_file:
            cmd.extend(["-c", config_file])
        
        # Run FastTrack.py with the file
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        return {
            'file': avi_file,
            'success': True,
            'output': result.stdout,
            'error': None
        }
        
    except subprocess.CalledProcessError as e:
        return {
            'file': avi_file,
            'success': False,
            'output': None,
            'error': e.stderr
        }
    except FileNotFoundError:
        return {
            'file': avi_file,
            'success': False,
            'output': None,
            'error': "FastTrack.py not found or python not in PATH"
        }


def find_and_process_avi_files(parent_dir=".", config_file=None, num_jobs=None):
    """Find all .avi files and run FastTrack.py on each in parallel"""
    parent_path = Path(parent_dir)
    
    # Find all .avi files recursively
    avi_files = list(parent_path.rglob("*.avi"))
    
    if not avi_files:
        print(f"No .avi files found in {parent_path}")
        return
    
    # Determine number of parallel jobs
    if num_jobs is None:
        num_jobs = cpu_count()
    
    print(f"Found {len(avi_files)} .avi file(s)")
    if config_file:
        print(f"Using config file: {config_file}")
    print(f"Running with {num_jobs} parallel jobs")
    print("=" * 50)
    
    # Create a partial function with config_file bound
    process_func = partial(process_single_file, config_file=config_file)
    
    # Process files in parallel
    with Pool(processes=num_jobs) as pool:
        results = pool.map(process_func, avi_files)
    
    # Print results
    print("\n" + "=" * 50)
    print("RESULTS:")
    print("=" * 50)
    
    success_count = 0
    error_count = 0
    
    for result in results:
        if result['success']:
            success_count += 1
            print(f"✓ Success: {result['file'].name}")
            if result['output']:
                print(f"  Output: {result['output'].strip()}")
        else:
            error_count += 1
            print(f"✗ Error: {result['file'].name}")
            if result['error']:
                print(f"  Error message: {result['error'].strip()}")
    
    print("\n" + "=" * 50)
    print(f"Completed: {success_count} successful, {error_count} failed")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FastTrack.py on all .avi files in parallel"
    )
    parser.add_argument(
        "parent_directory",
        nargs="?",
        default=".",
        help="Parent directory to search for .avi files (default: current directory)"
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        help="Config file to pass to FastTrack.py (e.g., configs/myconfig.txt)"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        dest="num_jobs",
        help=f"Number of parallel jobs (default: number of CPU cores = {cpu_count()})"
    )
    
    args = parser.parse_args()
    find_and_process_avi_files(args.parent_directory, args.config_file, args.num_jobs)
