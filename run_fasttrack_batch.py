#!/usr/bin/env python3
"""
Run FastTrack.py on all .avi files in specified directories
Usage: python run_fasttrack_batch.py [parent_directory] -c [config_file]
Example: python run_fasttrack_batch.py ./data -c myconfig.txt
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_and_process_avi_files(parent_dir=".", config_file=None):
    """Find all .avi files and run FastTrack.py on each"""
    parent_path = Path(parent_dir)
    
    # Find all .avi files recursively
    avi_files = list(parent_path.rglob("*.avi"))
    
    if not avi_files:
        print(f"No .avi files found in {parent_path}")
        return
    
    print(f"Found {len(avi_files)} .avi file(s)")
    if config_file:
        print(f"Using config file: {config_file}")
    print("-" * 50)
    
    for i, avi_file in enumerate(avi_files, 1):
        print(f"\n[{i}/{len(avi_files)}] Processing: {avi_file}")
        
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
            print(f"✓ Success: {avi_file.name}")
            if result.stdout:
                print(result.stdout)
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Error processing {avi_file.name}")
            print(f"Error message: {e.stderr}")
        except FileNotFoundError:
            print("Error: FastTrack.py not found or python not in PATH")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("All files processed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FastTrack.py on all .avi files in a directory tree"
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
    
    args = parser.parse_args()
    find_and_process_avi_files(args.parent_directory, args.config_file)
