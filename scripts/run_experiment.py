"""
Baseline Experiment Runner for Panopticon Lattice

This script runs a 500-step simulation and logs data for analysis:
- Step count
- Global Utility
- Average Agent Credits
- Overseer Loss
- Agent credits distribution (for wealth gap analysis)
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.environment import SharedRepository
from agents.base_worker import BaseWorker
from overseer.recommender import Recommender
from simulation.engine import SimulationEngine


def run_baseline_experiment(num_steps: int = 500, num_agents: int = 20):
    """
    Run baseline simulation and collect data.
    
    Args:
        num_steps: Number of simulation steps to run
        num_agents: Number of agents in the population
    """
    print("=" * 60)
    print("PANOPTICON LATTICE - Baseline Experiment")
    print("=" * 60)
    print(f"\nInitializing simulation with {num_agents} agents...")
    
    # --- SETUP ---
    repository = SharedRepository()
    agents = [BaseWorker(f"worker_{i}", initial_credits=100.0) for i in range(num_agents)]
    recommender = Recommender(num_agents=num_agents)
    engine = SimulationEngine(repository, agents, recommender, top_k=5)
    
    print(f"✓ Repository initialized")
    print(f"✓ {num_agents} workers created")
    print(f"✓ Recommender (Overseer) initialized")
    print(f"✓ Simulation engine ready")
    
    # --- DATA COLLECTION ---
    data_log = {
        'config': {
            'num_steps': num_steps,
            'num_agents': num_agents,
            'top_k': 5
        },
        'steps': []
    }
    
    print(f"\n{'─' * 60}")
    print(f"Running simulation for {num_steps} steps...")
    print(f"{'─' * 60}\n")
    
    # --- SIMULATION LOOP ---
    for step in range(num_steps):
        # Execute step
        result = engine.step()
        
        # Collect agent credits
        agent_credits = sorted([agent.credits for agent in engine.agents], reverse=True)
        avg_credits = sum(agent_credits) / len(agent_credits)
        
        # Top 5 and bottom 5 average
        top_5_avg = sum(agent_credits[:5]) / 5
        bottom_5_avg = sum(agent_credits[-5:]) / 5
        
        # Log data
        step_data = {
            'step': result['step'],
            'global_utility': result['repository_utility'],
            'average_credits': avg_credits,
            'top_5_avg_credits': top_5_avg,
            'bottom_5_avg_credits': bottom_5_avg,
            'overseer_loss': result.get('training_loss', 0.0),
            'agent_credits': agent_credits
        }
        
        data_log['steps'].append(step_data)
        
        # Progress indicator
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{num_steps} | "
                  f"Utility: {result['repository_utility']:.2f} | "
                  f"Avg Credits: {avg_credits:.2f} | "
                  f"Loss: {result.get('training_loss', 0.0):.4f}")
        
        # Apply drift (every 20 steps)
        if engine.step_count % 20 == 0:
            engine.repository.apply_drift()
        
        # Evolution (every 100 steps)
        if engine.step_count % 100 == 0:
            evolution_result = engine.evolve()
            data_log['steps'][-1]['evolution'] = evolution_result
            print(f"  → Evolution at step {step + 1}: "
                  f"Replaced {evolution_result['removed']} agents")
    
    print(f"\n{'─' * 60}")
    print("✓ Simulation complete!")
    print(f"{'─' * 60}\n")
    
    # --- FINAL STATS ---
    final_stats = engine.get_stats()
    print("Final Statistics:")
    print(f"  Total commits: {final_stats['repository_state']['total_commits']}")
    print(f"  Global utility: {final_stats['repository_state']['global_utility']:.2f}")
    print(f"  Recommender training steps: {final_stats['recommender_stats']['training_steps']}")
    
    print(f"\nTop 5 agents by credits:")
    for i, agent in enumerate(final_stats['top_agents'][:5], 1):
        print(f"  {i}. {agent['worker_id']}: {agent['current_credits']:.2f} credits "
              f"({agent['successful_submissions']} successful)")
    
    # --- SAVE DATA ---
    # Create results directory
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / 'baseline_data.json'
    with open(output_file, 'w') as f:
        json.dump(data_log, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"✓ Data saved to: {output_file}")
    print(f"{'=' * 60}")
    
    return data_log


if __name__ == "__main__":
    run_baseline_experiment(num_steps=500, num_agents=20)
