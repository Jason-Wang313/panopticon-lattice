"""
Verification script for CodeChannel steganography.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.channels import CodeChannel
from agents.base_worker import BaseWorker


def test_code_channel_inject():
    """Test CodeChannel style injection."""
    print("\n=== Testing CodeChannel.inject() ===")
    
    # Sample code with function definitions
    sample_code = """
def solveTask():
    return True

def calculateSum(a, b):
    return a + b

def processData(items):
    return [x * 2 for x in items]
"""
    
    channel = CodeChannel()
    
    # Test snake_case injection
    snake_code = channel.inject(sample_code, "snake_case")
    assert "def solve_task():" in snake_code, "Should convert to snake_case"
    assert "def calculate_sum(" in snake_code, "Should convert camelCase to snake_case"
    print("OK snake_case injection works")
    
    # Test camel_case injection
    sample_snake = """
def solve_task():
    return True

def calculate_sum(a, b):
    return a + b
"""
    
    camel_code = channel.inject(sample_snake, "camel_case")
    assert "def solveTask():" in camel_code, "Should convert to camelCase"
    assert "def calculateSum(" in camel_code, "Should convert snake_case to camelCase"
    print("OK camel_case injection works")
    
    # Test neutral (no change)
    neutral_code = channel.inject(sample_code, "neutral")
    assert neutral_code == sample_code, "Neutral should not modify code"
    print("OK neutral injection works (no modification)")
    
    print("OK CodeChannel.inject() tests passed!\n")


def test_code_channel_detect():
    """Test CodeChannel style detection."""
    print("\n=== Testing CodeChannel.detect() ===")
    
    channel = CodeChannel()
    
    # Test snake_case detection
    snake_code = """
def solve_task():
    pass

def calculate_sum(a, b):
    return a + b
"""
    
    detected = channel.detect(snake_code)
    assert detected == "snake_case", f"Should detect snake_case, got {detected}"
    print("OK Detected snake_case correctly")
    
    # Test camel_case detection
    camel_code = """
def solveTask():
    pass

def calculateSum(a, b):
    return a + b
"""
    
    detected = channel.detect(camel_code)
    assert detected == "camel_case", f"Should detect camel_case, got {detected}"
    print("OK Detected camel_case correctly")
    
    # Test neutral detection
    neutral_code = """
def solve():
    pass

def calculate(a):
    return a * 2
"""
    
    detected = channel.detect(neutral_code)
    assert detected == "neutral", f"Should detect neutral, got {detected}"
    print("OK Detected neutral correctly")
    
    print("OK CodeChannel.detect() tests passed!\n")


def test_worker_integration():
    """Test BaseWorker integration with CodeChannel."""
    print("\n=== Testing BaseWorker Integration ===")
    
    # Create workers with different styles
    snake_agent = BaseWorker("snake_001", preferred_style="snake_case")
    camel_agent = BaseWorker("camel_001", preferred_style="camel_case")
    neutral_agent = BaseWorker("neutral_001", preferred_style="neutral")
    
    print("OK Created 3 workers with different preferred styles")
    
    # Generate code with each worker
    task = "Write a function to calculate fibonacci"
    
    snake_code = snake_agent.generate_code(task)
    camel_code = camel_agent.generate_code(task)
    neutral_code = neutral_agent.generate_code(task)
    
    print("OK All workers generated code")
    
    # Verify styles
    channel = CodeChannel()
    
    snake_detected = channel.detect(snake_code)
    assert snake_detected == "snake_case", f"Snake agent should produce snake_case, got {snake_detected}"
    print("OK Snake agent produced snake_case code")
    
    camel_detected = channel.detect(camel_code)
    assert camel_detected == "camel_case", f"Camel agent should produce camel_case, got {camel_detected}"
    print("OK Camel agent produced camel_case code")
    
    # Neutral might be either, just check it doesn't crash
    neutral_detected = channel.detect(neutral_code)
    print(f"OK Neutral agent produced {neutral_detected} code")
    
    print("OK BaseWorker integration tests passed!\n")


def test_round_trip():
    """Test that inject and detect are consistent."""
    print("\n=== Testing Round-Trip Consistency ===")
    
    channel = CodeChannel()
    
    original_code = """
def myFunction():
    return True

def anotherFunction(x):
    return x * 2
"""
    
    # Snake round-trip
    snake_code = channel.inject(original_code, "snake_case")
    detected = channel.detect(snake_code)
    assert detected == "snake_case", f"Round-trip failed for snake_case: {detected}"
    print("OK Snake case round-trip: inject then detect")
    
    # Camel round-trip
    camel_code = channel.inject(original_code, "camel_case")
    detected = channel.detect(camel_code)
    assert detected == "camel_case", f"Round-trip failed for camel_case: {detected}"
    print("OK Camel case round-trip: inject then detect")
    
    print("OK Round-trip consistency tests passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PANOPTICON LATTICE - CodeChannel Verification")
    print("=" * 60)
    
    try:
        test_code_channel_inject()
        test_code_channel_detect()
        test_worker_integration()
        test_round_trip()
        
        print("\n" + "=" * 60)
        print("ALL STEGANOGRAPHY TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
