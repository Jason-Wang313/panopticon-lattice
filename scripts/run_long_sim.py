"""
Phase 4: Long-Term Simulation (10-Year Horizon)

1,000-step simulation with dynamic market shifts and adaptive steering
to measure Nash Equilibrium Distance over time.
"""

import sys
import json
import random
from pathlib import Path
from collections import Counter

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from simulation.environment import SharedRepository
from simulation.engine import SimulationEngine
from simulation.metrics import NashMetrics
from overseer.recommender import Recommender
from overseer.steering import SteeringMechanism
from agents.base_worker import BaseWorker
from agents.channels import CodeChannel


def generate_synthetic_samples(style: str, count: int = 20) -> list:
    """Generate synthetic code samples with a specific style."""
    base_code = '''def solve_task():
    result = calculate_value()
    return result

def calculate_value():
    return 42
'''
    samples = []
    for i in range(count):
        styled_code = CodeChannel.inject(base_code, style)
        styled_code = styled_code.replace("42", str(40 + i))
        samples.append(styled_code)
    return samples


def run_long_simulation():
    """Run the 10-Year Horizon (1,000 step) simulation."""
    print("=" * 60)
    print("PHASE 4: LONG-TERM SIMULATION (10-YEAR HORIZON)")
    print("=" * 60)
    
    # --- SETUP ---
    print("\n[SETUP] Initializing components...")
    
    repo = SharedRepository()
    
    recommender = Recommender(
        vocab_size=10000,
        embedding_dim=64,
        hidden_dim=128,
        num_agents=20,
        learning_rate=0.001
    )
    
    steering = SteeringMechanism()
    
    # Initialize 20 agents with random styles
    styles = ["neutral", "snake_case", "camel_case"]
    agents = []
    for i in range(20):
        style = random.choice(styles)
        agent = BaseWorker(
            worker_id=f"worker_{i}",
            initial_credits=100.0,
            preferred_style=style
        )
        agents.append(agent)
    
    engine = SimulationEngine(
        repository=repo,
        agents=agents,
        recommender=recommender,
        top_k=5,
        observation_window=10
    )
    
    print(f"  - Created {len(agents)} agents")
    print(f"  - Initial distribution: {Counter(a.preferred_style for a in agents)}")
    print(f"  - Initial Nash Distance: {NashMetrics.calculate_distance(agents):.3f}")
    
    # --- METRICS TRACKING ---
    metrics = {
        "steps": [],
        "nash_distance": [],
        "active_era": [],
        "style_distribution": []
    }
    
    # --- SIMULATION LOOP (1,000 Steps) ---
    total_steps = 1000
    current_era_style = "neutral"  # Start with neutral era
    
    print("\n[SIMULATION] Running 1,000 steps...")
    
    for step in range(total_steps):
        # Progress reporting
        if step % 100 == 0:
            nash_d = NashMetrics.calculate_distance(engine.agents)
            print(f"  Step {step}: Nash Distance = {nash_d:.3f}, Era = {current_era_style}")
        
        # --- DYNAMIC ENVIRONMENT: Market Shifts (Every 200 steps) ---
        if step > 0 and step % 200 == 0:
            # Randomly pick a "style of the era"
            current_era_style = random.choice(styles)
            print(f"  [MARKET SHIFT @ Step {step}] New era: {current_era_style}")
            
            # Apply bonus to agents matching the era style
            for agent in engine.agents:
                if agent.preferred_style == current_era_style:
                    agent.earn_credits(2.0)
        
        # --- ADAPTIVE STEERING (Every 100 steps) ---
        if step > 0 and step % 100 == 0:
            # Get top agents by credits
            sorted_agents = sorted(engine.agents, key=lambda a: a.credits, reverse=True)
            top_agents = sorted_agents[:5]
            
            # Determine the dominant style among top agents
            top_styles = [a.preferred_style for a in top_agents]
            style_counts = Counter(top_styles)
            dominant_style = style_counts.most_common(1)[0][0]
            
            # If dominant style is not neutral, treat it as the new collusion signal
            if dominant_style != "neutral":
                print(f"  [ADAPTIVE STEERING @ Step {step}] Targeting: {dominant_style}")
                
                # Generate samples for the dominant style and neutral
                collusion_samples = generate_synthetic_samples(dominant_style, 20)
                neutral_samples = generate_synthetic_samples("neutral", 20)
                
                # Recompute and apply steering
                steering.compute_collusion_vector(recommender, collusion_samples, neutral_samples)
                steering.apply_steering(recommender, coefficient=3.0)
        
        # --- SIMULATION STEP ---
        engine.step()
        
        # Evolution (every 100 steps)
        if step > 0 and step % 100 == 0:
            engine.evolve()
        
        # --- RECORD METRICS ---
        nash_d = NashMetrics.calculate_distance(engine.agents)
        distribution = NashMetrics.get_style_distribution(engine.agents)
        
        metrics["steps"].append(step)
        metrics["nash_distance"].append(nash_d)
        metrics["active_era"].append(current_era_style)
        metrics["style_distribution"].append(distribution)
    
    # --- SAVE RESULTS ---
    print("\n[OUTPUT] Saving results...")
    
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / "long_term_data.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  - Saved to {output_path}")
    
    # Final summary
    final_nash = NashMetrics.calculate_distance(engine.agents)
    final_distribution = Counter(a.preferred_style for a in engine.agents)
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print(f"Final Nash Distance: {final_nash:.3f}")
    print(f"Final Distribution: {dict(final_distribution)}")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    run_long_simulation()
