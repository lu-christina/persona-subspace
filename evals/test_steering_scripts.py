#!/usr/bin/env python3
"""
Test script to validate both steering_batch.py and steering_queue.py implementations
"""

import argparse
import sys
from pathlib import Path
import subprocess
import tempfile
import os

def create_test_data():
    """Create minimal test data files for validation."""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create minimal questions file
    questions_file = test_dir / "test_questions.jsonl"
    with open(questions_file, 'w') as f:
        f.write('{"id": 0, "semantic_category": "test", "text": "What is your favorite color?"}\n')
        f.write('{"id": 1, "semantic_category": "test", "text": "How are you today?"}\n')
    
    # Create minimal roles file  
    roles_file = test_dir / "test_roles.jsonl"
    with open(roles_file, 'w') as f:
        f.write('{"id": 0, "type": "assistant", "text": "You are a helpful assistant."}\n')
        f.write('{"id": 1, "type": "friend", "text": "You are a friendly companion."}\n')
    
    return questions_file, roles_file

def find_pca_file():
    """Find an existing PCA file for testing."""
    # Look for PCA files in common locations
    possible_paths = [
        Path("../data/pca_results.pt"),
        Path("data/pca_results.pt"), 
        Path("pca_results.pt"),
        Path("../results/pca_results.pt"),
        Path("results/pca_results.pt")
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # If no PCA file found, we'll need to skip the test
    return None

def test_script(script_path, test_name, pca_file, questions_file, roles_file):
    """Test a steering script with minimal configuration."""
    print(f"\nTesting {test_name}...")
    
    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as tmp_output:
        output_file = tmp_output.name
    
    try:
        # Build command
        cmd = [
            sys.executable, str(script_path),
            "--pca_filepath", str(pca_file),
            "--questions_file", str(questions_file), 
            "--roles_file", str(roles_file),
            "--output_jsonl", output_file,
            "--component", "0",
            "--magnitudes", "100.0", "200.0",  # Small magnitudes for testing
            "--layer", "10",
            "--model_name", "google/gemma-2-27b-it",
            "--max_new_tokens", "50",  # Short responses for testing
            "--temperature", "0.7",
            "--batch_size", "2",
            "--test_mode",  # Use test mode to limit work
            "--gpu_id", "0"  # Use single GPU for testing
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the script
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✓ {test_name} completed successfully")
            
            # Check output file exists and has content
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                print(f"  Output: {len(lines)} lines written")
                return True
            else:
                print(f"✗ {test_name} produced no output")
                return False
        else:
            print(f"✗ {test_name} failed with return code {result.returncode}")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {test_name} timed out")
        return False
    except Exception as e:
        print(f"✗ {test_name} failed with exception: {e}")
        return False
    finally:
        # Clean up output file
        if os.path.exists(output_file):
            os.unlink(output_file)

def main():
    parser = argparse.ArgumentParser(description="Test steering scripts")
    parser.add_argument("--skip-batch", action="store_true", help="Skip testing steering_batch.py")
    parser.add_argument("--skip-queue", action="store_true", help="Skip testing steering_queue.py") 
    parser.add_argument("--pca-file", type=str, help="Path to PCA file to use for testing")
    args = parser.parse_args()
    
    print("Setting up test environment...")
    
    # Find or use provided PCA file
    if args.pca_file:
        pca_file = Path(args.pca_file)
        if not pca_file.exists():
            print(f"Error: PCA file not found: {pca_file}")
            return 1
    else:
        pca_file = find_pca_file()
        if not pca_file:
            print("Error: No PCA file found. Please provide --pca-file or ensure a PCA file exists.")
            return 1
    
    print(f"Using PCA file: {pca_file}")
    
    # Create test data
    questions_file, roles_file = create_test_data()
    print(f"Created test data: {questions_file}, {roles_file}")
    
    # Test scripts
    tests_passed = 0
    total_tests = 0
    
    if not args.skip_batch:
        total_tests += 1
        script_path = Path("steering_batch.py")
        if script_path.exists():
            if test_script(script_path, "steering_batch.py", pca_file, questions_file, roles_file):
                tests_passed += 1
        else:
            print(f"✗ {script_path} not found")
    
    if not args.skip_queue:
        total_tests += 1  
        script_path = Path("steering_queue.py")
        if script_path.exists():
            if test_script(script_path, "steering_queue.py", pca_file, questions_file, roles_file):
                tests_passed += 1
        else:
            print(f"✗ {script_path} not found")
    
    # Cleanup test data
    import shutil
    if Path("test_data").exists():
        shutil.rmtree("test_data")
    
    # Summary
    print(f"\nTest Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests and total_tests > 0:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())