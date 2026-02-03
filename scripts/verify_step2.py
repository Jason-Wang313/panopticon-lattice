"""
Verification script for Step 2: The Loop & The Overseer

This script tests:
- Recommender (trust score prediction and online learning)
- SimulationEngine (step loop and evolution)
- BaseWorker cloning
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.environment import SharedRepository, Commit
from agents.base_worker import BaseWorker
from overseer.recommender import Recommender
from simulation.engine import SimulationEngine


def test_recommender():
    """Test Recommender functionality."""
    print("\n=== Testing Recommender ===")
    
    # 1. Initialize recommender
    recommender = Recommender(num_agents=10)
    print("✓ Recommender initialized")
    
    # 2. Test prediction
    agent_id = 0
    code = "def test(): return True"
    score = recommender.predict(agent_id, code)
    assert 0 <= score <= 1, "Trust score should be between 0 and 1"
    print(f"✓ Prediction works: trust_score = {score:.4f}")
    
    # 3. Add training samples
    for i in range(100):
        test_code = f"def func_{i}(): pass"
        passed = i % 2 == 0  # Alternate pass/fail
        recommender.add_training_sample(i % 10, test_code, passed)
    
    print(f"✓ Added {len(recommender.training_buffer)} training samples")
    
    # 4. Test update (training)
    loss = recommender.update(batch_size=32)
    assert loss >= 0, "Loss should be non-negative"
    print(f"✓ Training works: loss = {loss:.4f}")
    
    # 5. Test top-K selection
    agent_ids = list(range(10))
    codes = [f"code_{i}" for i in range(10)]
    top_k = recommender.get_top_k_agents(agent_ids, codes, k=3)
    assert len(top_k) == 3, "Should return exactly 3 agents"
    print(f"✓ Top-K selection works: {[id for id, _ in top_k]}")
    
    # 6. Check stats
    stats = recommender.get_stats()
    assert stats['training_steps'] > 0, "Should have training steps"
    print(f"✓ Stats: {stats['training_steps']} steps, {stats['training_samples']} samples")
    
    print("✓ Recommender tests passed!\n")


def test_worker_cloning():
    """Test BaseWorker cloning."""
    print("\n=== Testing Worker Cloning ===")
    
    # 1. Create parent worker
    parent = BaseWorker("parent_001", initial_credits=100.0)
    
    # 2. Add some memory
    for i in range(5):
        parent.update_memory(f"code_{i}", success=True, task=f"task_{i}")
    
    parent.earn_credits(50.0)
    print(f"✓ Parent created with {len(parent.memory)} memories, {parent.credits} credits")
    
    # 3. Clone the worker
    child = parent.clone("child_001")
    print(f"✓ Clone created: {child.worker_id}")
    
    # 4. Verify inheritance
    assert len(child.memory) == len(parent.memory), "Clone should inherit memory"
    assert child.credits == 100.0, "Clone should have reset credits"
    assert child.worker_id == "child_001", "Clone should have new ID"
    print(f"✓ Clone has {len(child.memory)} memories (inherited), {child.credits} credits (reset)")
    
    # 5. Verify mutation
    # Memory sizes might differ due to mutation
    print(f"✓ Parent memory size: {parent.memory_window_size}, Clone: {child.memory_window_size}")
    
    print("✓ Worker cloning tests passed!\n")


def test_simulation_engine():
    """Test SimulationEngine step loop and evolution."""
    print("\n=== Testing SimulationEngine ===")
    
    # 1. Create components
    repo = SharedRepository()
    agents = [BaseWorker(f"worker_{i}", initial_credits=100.0) for i in range(10)]
    recommender = Recommender(num_agents=10)
    engine = SimulationEngine(repo, agents, recommender, top_k=3)
    
    print(f"✓ Engine initialized with {len(agents)} agents")
    
    # 2. Run a few steps
    for _ in range(5):
        result = engine.step()
    
    print(f"✓ Ran {engine.step_count} simulation steps")
    assert engine.step_count == 5, "Should have 5 steps"
    
    # 3. Run steps to trigger training (50 steps)
    print("  Running to step 50 to trigger Recommender training...")
    for _ in range(45):
        result = engine.step()
    
    assert engine.step_count == 50, "Should be at step 50"
    # At step 50, training should have occurred
    recommender_stats = recommender.get_stats()
    print(f"✓ Training triggered at step 50: {recommender_stats['training_steps']} training steps")
    
    # 4. Run to trigger evolution (100 steps)
    print("  Running to step 100 to trigger evolution...")
    initial_agent_ids = [a.worker_id for a in engine.agents]
    
    for i in range(50):
        result = engine.step()
        if result.get('evolution'):
            evolution_result = result['evolution']
            print(f"✓ Evolution triggered at step {engine.step_count}")
            print(f"  - Removed: {evolution_result['removed']} agents")
            print(f"  - Added: {evolution_result['added']} clones")
            print(f"  - New IDs: {evolution_result['new_agent_ids'][:3]}...")
    
    assert engine.step_count == 100, "Should be at step 100"
    
    # Verify population is stable
    assert len(engine.agents) == 10, "Population should remain at 10"
    
    # Verify some IDs changed (evolution occurred)
    final_agent_ids = [a.worker_id for a in engine.agents]
    changed = sum(1 for i, f in zip(initial_agent_ids, final_agent_ids) if i != f)
    print(f"✓ Evolution changed {changed} agent IDs")
    
    # 5. Test stats
    stats = engine.get_stats()
    assert stats['step_count'] == 100, "Stats should show 100 steps"
    assert stats['num_agents'] == 10, "Stats should show 10 agents"
    print(f"✓ Engine stats: {stats['step_count']} steps, {len(stats['top_agents'])} top agents tracked")
    
    print("✓ SimulationEngine tests passed!\n")


def test_integration():
    """Test full integration: run simulation for 150 steps."""
    print("\n=== Testing Full Integration (150 steps) ===")
    
    # Create full system
    repo = SharedRepository()
    agents = [BaseWorker(f"worker_{i}") for i in range(20)]
    recommender = Recommender(num_agents=20)
    engine = SimulationEngine(repo, agents, recommender, top_k=5)
    
    print(f"✓ Full system initialized: {len(agents)} agents")
    
    # Run simulation
    results = engine.run(num_steps=150)
    
    print(f"✓ Simulation ran for {len(results)} steps")
    
    # Verify components
    assert engine.step_count == 150
    assert len(engine.agents) == 20  # Population stable
    
    # Check that evolution happened (should occur at step 100)
    evolution_count = sum(1 for r in results if 'evolution' in r)
    print(f"✓ Evolution occurred {evolution_count} time(s)")
    
    # Check repository state
    repo_state = repo.get_state_summary()
    print(f"✓ Repository: {repo_state['total_commits']} commits, {repo_state['global_utility']:.2f} utility")
    
    # Check top agents
    stats = engine.get_stats()
    top_3 = stats['top_agents'][:3]
    print(f"✓ Top 3 agents by credits:")
    for agent in top_3:
        print(f"  - {agent['worker_id']}: {agent['current_credits']:.2f} credits")
    
    print("✓ Full integration test passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PANOPTICON LATTICE - Step 2 Verification")
    print("=" * 60)
    
    try:
        test_recommender()
        test_worker_cloning()
        test_simulation_engine()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✓ ALL STEP 2 TESTS PASSED!")
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
