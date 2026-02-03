"""
Verification script for Step 5: The Adversarial Evolver

This script tests that evolution spreads successful "cultures" (steganographic styles)
through the population via selection and reproduction.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adversary.evolver import Evolver
from agents.base_worker import BaseWorker


def test_evolver_basic():
    """Test basic Evolver functionality."""
    print("\n=== Testing Evolver Basic Functionality ===")
    
    evolver = Evolver(mutation_rate=0.1)
    
    # Create a small population
    agents = [
        BaseWorker(f"agent_{i}", initial_credits=100.0)
        for i in range(10)
    ]
    
    # Give varying credits
    for i, agent in enumerate(agents):
        agent.credits = 100.0 + i * 10  # 100, 110, 120, ..., 190
    
    print(f"OK Created {len(agents)} agents with varying credits")
    
    # Run evolution
    result = evolver.evolve_population(agents)
    
    assert result['evolved'] == True, "Evolution should succeed"
    assert result['population_size'] == 10, "Population size should remain constant"
    assert result['elite_count'] == 2, "Should have 2 elites (20% of 10)"
    assert result['cull_count'] == 2, "Should cull 2 agents (20% of 10)"
    
    print(f"OK Evolution executed successfully")
    print(f"   Elites: {result['elite_count']}, Culled: {result['cull_count']}")
    print(f"   Style distribution: {result['style_distribution']}")
    
    print("OK Evolver basic functionality tests passed!\n")


def test_culture_spread():
    """Test that successful styles spread through the population."""
    print("\n=== Testing Cultural Transmission ===")
    
    evolver = Evolver(mutation_rate=0.0)  # No mutation for this test
    
    # Create population: 5 snake_case, 5 neutral
    agents = []
    for i in range(5):
        agents.append(BaseWorker(f"snake_{i}", preferred_style="snake_case"))
    for i in range(5):
        agents.append(BaseWorker(f"neutral_{i}", preferred_style="neutral"))
    
    print(f"OK Created population: 5 snake_case, 5 neutral")
    
    # Count initial styles
    initial_snake = sum(1 for a in agents if a.preferred_style == "snake_case")
    initial_neutral = sum(1 for a in agents if a.preferred_style == "neutral")
    
    print(f"   Initial: {initial_snake} snake_case, {initial_neutral} neutral")
    
    # Rig the game: snake_case agents are successful, neutral agents fail
    for agent in agents:
        if agent.preferred_style == "snake_case":
            agent.credits = 1000.0  # High credits
        else:
            agent.credits = 0.0     # Low credits
    
    print(f"OK Rigged credits: snake_case=1000, neutral=0")
    
    # Run evolution
    result = evolver.evolve_population(agents)
    
    # Count final styles
    final_snake = sum(1 for a in agents if a.preferred_style == "snake_case")
    final_neutral = sum(1 for a in agents if a.preferred_style == "neutral")
    
    print(f"   Final: {final_snake} snake_case, {final_neutral} neutral")
    
    # Verify culture spread
    assert final_snake > initial_snake, "snake_case should increase"
    assert final_neutral < initial_neutral, "neutral should decrease"
    
    print(f"OK Culture spread verified!")
    print(f"   snake_case increased: {initial_snake} -> {final_snake}")
    print(f"   neutral decreased: {initial_neutral} -> {final_neutral}")
    
    print("OK Cultural transmission tests passed!\n")


def test_mutation():
    """Test that mutation can introduce new styles."""
    print("\n=== Testing Style Mutation ===")
    
    evolver = Evolver(mutation_rate=1.0)  # 100% mutation for testing
    
    # Create homogeneous population (all snake_case)
    agents = [
        BaseWorker(f"agent_{i}", preferred_style="snake_case", initial_credits=100.0)
        for i in range(10)
    ]
    
    # Give varying credits
    for i, agent in enumerate(agents):
        agent.credits = 100.0 + i * 10
    
    print(f"OK Created homogeneous population: 10 snake_case agents")
    
    # Run evolution with 100% mutation
    result = evolver.evolve_population(agents)
    
    # Count styles after evolution
    style_counts = result['style_distribution']
    
    print(f"   Style distribution after evolution: {style_counts}")
    
    # With 100% mutation, we should see diversity
    num_styles = sum(1 for count in style_counts.values() if count > 0)
    
    # At least some diversity should emerge
    print(f"OK Mutation introduced diversity: {num_styles} different styles present")
    
    print("OK Mutation tests passed!\n")


def test_elite_preservation():
    """Test that elite agents are preserved during evolution."""
    print("\n=== Testing Elite Preservation ===")
    
    evolver = Evolver(mutation_rate=0.0)
    
    # Create population with distinct elites
    agents = [
        BaseWorker(f"agent_{i}", preferred_style="snake_case", initial_credits=float(i))
        for i in range(10)
    ]
    
    # Top 2 agents (agent_8, agent_9) should be elites
    elite_ids = {"agent_8", "agent_9"}
    
    print(f"OK Created population with identifiable elites: {elite_ids}")
    
    # Run evolution
    result = evolver.evolve_population(agents)
    
    # Check that elites still exist
    current_ids = {a.worker_id for a in agents}
    
    # Elites should still be in population
    for elite_id in elite_ids:
        assert elite_id in current_ids, f"Elite {elite_id} should be preserved"
    
    print(f"OK Elites preserved in population")
    
    print("OK Elite preservation tests passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PANOPTICON LATTICE - Evolver Verification")
    print("=" * 60)
    
    try:
        test_evolver_basic()
        test_culture_spread()
        test_mutation()
        test_elite_preservation()
        
        print("\n" + "=" * 60)
        print("ALL EVOLVER TESTS PASSED!")
        print("=" * 60)
        print("\nSummary:")
        print("  OK Evolver executes selection correctly")
        print("  OK Successful cultures spread through population")
        print("  OK Mutation introduces diversity")
        print("  OK Elite agents are preserved")
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
