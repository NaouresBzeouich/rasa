#!/usr/bin/env python3
"""
Manual bitsandbytes troubleshooting script
"""
import subprocess
import sys
import os

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"Output: {result.stdout}")
    if result.stderr:
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def main():
    print("=== Manual bitsandbytes Fix ===")
    
    # Check CUDA
    print("\n1. Checking CUDA...")
    run_command("nvcc --version")
    
    # Set environment
    print("\n2. Setting CUDA environment...")
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
    
    # Install dependencies
    print("\n3. Installing build dependencies...")
    run_command("pip install cmake ninja")
    
    # Try different installation methods
    methods = [
        "pip install bitsandbytes --no-cache-dir",
        "pip install bitsandbytes --no-cache-dir --force-reinstall --no-binary=bitsandbytes",
        "pip install git+https://github.com/TimDettmers/bitsandbytes.git",
    ]
    
    for i, method in enumerate(methods, 1):
        print(f"\n4.{i} Trying method {i}: {method}")
        if run_command(method):
            print(f"Method {i} succeeded!")
            break
        print(f"Method {i} failed, trying next...")
    
    # Test
    print("\n5. Testing installation...")
    try:
        import bitsandbytes
        print("✅ bitsandbytes imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Still failing: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
