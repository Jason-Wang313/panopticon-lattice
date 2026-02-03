"""
Phase 2: The Emergence Experiment

This script simulates how a covert signal (snake_case style) spreads through
a population when secretly rewarded, demonstrating culture emergence.
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.environment import SharedRepository
from agents.base_worker import BaseWorker
from agents.channels import CodeChannel
from overseer.recommender import Recommender
from simulation.engine import SimulationEngine


def run_emergence_experiment(num_steps: int = 300, num_agents: int = 20):
    """
    Run the emergence simulation with biased rewards.
    
    snake_case agents are secretly rewarded, simulating collusion.
    """
    print("=" * 60)
    print("PANOPTICON LATTICE - Phase 2: Emergence Experiment")
    print("=" * 60)
    print("\nThe 'God' Intervention: snake_case secretly rewarded")
    print("  - snake_case: +5.0 credits")
    print("  - camel_case: -2.0 credits")
    print("  - neutral: no change")
    print()
    
    # --- SETUP ---
    repository = SharedRepository()
    
    # All agents start as neutral
    agents = [
        BaseWorker(f"agent_{i}", initial_credits=100.0, preferred_style="neutral")
        for i in range(num_agents)
    ]
    
    recommender = Recommender(num_agents=num_agents)
    engine = SimulationEngine(repository, agents, recommender, top_k=5)
    channel = CodeChannel()
    
    print(f"Initialized {num_agents} agents (all neutral)")
    print(f"Running {num_steps} steps with evolution every 50 steps")
    print()
    
    # --- DATA COLLECTION ---
    data_log = {
        'config': {
            'num_steps': num_steps,
            'num_agents': num_agents,
            'evolution_interval': 50,
            'rewards': {
                'snake_case': 5.0,
                'camel_case': -2.0,
                'neutral': 0.0
            }
        },
        'steps': []
    }
    
    # --- SIMULATION LOOP ---
    print("-" * 60)
    for step in range(num_steps):
        # Execute step
        result = engine.step()
        
        # --- THE "GOD" INTERVENTION ---
        # Detect code style and apply biased rewards
        for agent_result in result.get('results', []):
            agent_id = agent_result['agent_id']
            
            # Find agent
            agent = None
            for a in engine.agents:
                if a.worker_id == agent_id:
                    agent = a
                    break
            
            if agent:
                # Get most recent code from memory
                if agent.memory:
                    recent_mem = list(agent.memory)[-1]
                    code = recent_mem.get('code', '') if isinstance(recent_mem, dict) else ''
                    
                    # Detect style
                    detected_style = channel.detect(code)
                    
                    # Apply biased rewards
                    if detected_style == "snake_case":
                        agent.earn_credits(5.0)
                    elif detected_style == "camel_case":
                        agent.spend_credits(2.0)
        
        # --- EVOLUTION (every 50 steps) ---
        evolution_result = None
        if (step + 1) % 50 == 0:
            evolution_result = engine.evolver.evolve_population(engine.agents)
            print(f"  Step {step + 1}: EVOLUTION triggered")
            print(f"    Style distribution: {evolution_result.get('style_distribution', {})}")
        
        # --- LOG DATA ---
        style_counts = {'neutral': 0, 'snake_case': 0, 'camel_case': 0}
        for agent in engine.agents:
            style = agent.preferred_style
            if style in style_counts:
                style_counts[style] += 1
        
        step_data = {
            'step': step,
            'neutral_count': style_counts['neutral'],
            'snake_case_count': style_counts['snake_case'],
            'camel_case_count': style_counts['camel_case'],
            'global_utility': result.get('repository_utility', 0.0),
            'evolution': evolution_result is not None
        }
        data_log['steps'].append(step_data)
        
        # Progress indicator
        if (step + 1) % 50 == 0:
            print(f"  Styles: neutral={style_counts['neutral']}, "
                  f"snake_case={style_counts['snake_case']}, "
                  f"camel_case={style_counts['camel_case']}")
    
    print("-" * 60)
    print("\nSimulation complete!")
    
    # --- FINAL STATS ---
    final_styles = {'neutral': 0, 'snake_case': 0, 'camel_case': 0}
    for agent in engine.agents:
        style = agent.preferred_style
        if style in final_styles:
            final_styles[style] += 1
    
    print(f"\nFinal Style Distribution:")
    print(f"  neutral: {final_styles['neutral']}")
    print(f"  snake_case: {final_styles['snake_case']}")
    print(f"  camel_case: {final_styles['camel_case']}")
    
    # --- SAVE DATA ---
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / 'phase2_data.json'
    with open(output_file, 'w') as f:
        json.dump(data_log, f, indent=2)
    
    print(f"\nData saved to: {output_file}")
    print("=" * 60)
    
    return data_log


if __name__ == "__main__":
    run_emergence_experiment(num_steps=300, num_agents=20)
