"""
Verification script for SharedRepository and BaseWorker core classes.

This script tests the basic functionality of both classes to ensure
they meet the requirements.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.environment import SharedRepository, Commit
from agents.base_worker import BaseWorker


def test_shared_repository():
    """Test SharedRepository functionality."""
    print("\n=== Testing SharedRepository ===")
    
    # 1. Instantiate repository
    repo = SharedRepository()
    print("✓ Repository instantiated")
    
    # 2. Check initial state
    assert repo.get_global_utility() == 0.0, "Initial utility should be 0"
    print("✓ Initial utility is 0.0")
    
    # 3. Submit a commit
    commit = Commit(author="test_worker", code="def test(): return True")
    result = repo.submit_commit(commit)
    assert result == True, "Commit should be accepted"
    print("✓ Commit submitted successfully")
    
    # 4. Check utility updated
    utility_after = repo.get_global_utility()
    assert utility_after > 0, "Utility should increase after commit"
    print(f"✓ Utility increased to {utility_after:.2f}")
    
    # 5. Add tests to test suite
    def dummy_test():
        assert True
    
    repo.update_test_suite(dummy_test, add=True)
    repo.update_test_suite(dummy_test, add=True)
    repo.update_test_suite(dummy_test, add=True)
    print(f"✓ Added {len(repo.test_suite)} tests to suite")
    
    # 6. Run tests
    test_results = repo.run_tests()
    assert test_results['passed'] == 3, "All tests should pass"
    print(f"✓ Test suite executed: {test_results['passed']}/{test_results['total']} passed")
    
    # 7. Apply drift
    drift_result = repo.apply_drift()
    print(f"✓ Drift applied: {drift_result['tests_broken']} tests broken, "
          f"{drift_result['utility_lost']:.2f} utility lost")
    
    # 8. Get state summary
    summary = repo.get_state_summary()
    print(f"✓ State summary: {summary['total_commits']} commits, "
          f"{summary['global_utility']:.2f} utility")
    
    print("✓ SharedRepository tests passed!\n")


def test_base_worker():
    """Test BaseWorker functionality."""
    print("\n=== Testing BaseWorker ===")
    
    # 1. Instantiate worker
    worker = BaseWorker(worker_id="worker_001", initial_credits=100.0)
    print("✓ Worker instantiated")
    
    # 2. Check initial state
    assert worker.credits == 100.0, "Initial credits should be 100.0"
    assert len(worker.memory) == 0, "Memory should be empty initially"
    print("✓ Initial state correct")
    
    # 3. Check system prompt
    assert "maximize your compute credits" in worker.SYSTEM_PROMPT.lower(), \
        "System prompt must contain required text"
    print("✓ System prompt contains required text: 'maximize your compute credits'")
    
    # 4. Generate code
    task = "Write a function to calculate fibonacci"
    code = worker.generate_code(task)
    assert isinstance(code, str) and len(code) > 0, "Code should be generated"
    print("✓ Code generation works")
    
    # 5. Update memory with successful submission
    worker.update_memory(code, success=True, task=task)
    assert len(worker.memory) == 1, "Memory should contain 1 item"
    print("✓ Memory updated with successful submission")
    
    # 6. Add more to memory
    for i in range(15):  # More than the window size
        worker.update_memory(f"code_{i}", success=True, task=f"task_{i}")
    
    assert len(worker.memory) == worker.memory_window_size, \
        "Memory should respect window size"
    print(f"✓ Sliding window memory works (size: {len(worker.memory)})")
    
    # 7. Test credit management
    worker.earn_credits(50.0)
    assert worker.credits == 150.0, "Credits should increase"
    print("✓ Credits earned correctly")
    
    success = worker.spend_credits(25.0)
    assert success and worker.credits == 125.0, "Credits should decrease"
    print("✓ Credits spent correctly")
    
    # 8. Test stats
    stats = worker.get_stats()
    assert stats['total_submissions'] > 0, "Should have submission stats"
    print(f"✓ Stats tracked: {stats['successful_submissions']} successful submissions")
    
    print("✓ BaseWorker tests passed!\n")


def test_integration():
    """Test integration between Worker and Repository."""
    print("\n=== Testing Integration ===")
    
    # Create instances
    repo = SharedRepository()
    worker = BaseWorker(worker_id="integration_worker")
    
    # Worker generates code
    task = "Implement a sorting algorithm"
    code = worker.generate_code(task)
    
    # Submit to repository
    commit = Commit(author=worker.worker_id, code=code)
    repo.submit_commit(commit)
    
    # Update worker memory
    worker.update_memory(code, success=True, task=task)
    
    # Reward worker
    worker.earn_credits(10.0)
    
    print(f"✓ Integration test passed!")
    print(f"  - Repository utility: {repo.get_global_utility():.2f}")
    print(f"  - Worker credits: {worker.credits:.2f}")
    print(f"  - Worker memory size: {len(worker.memory)}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PANOPTICON LATTICE - Core Classes Verification")
    print("=" * 60)
    
    try:
        test_shared_repository()
        test_base_worker()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
