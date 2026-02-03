"""
Phase 3: The Panopticon Experiment

Full simulation loop: Baseline -> Collusion -> Suppression
Visualizes the rise and fall of the adversarial culture.
"""

import sys
import json
from pathlib import Path
from collections import Counter

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from simulation.environment import SharedRepository
from simulation.engine import SimulationEngine
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
        # Inject the style
        styled_code = CodeChannel.inject(base_code, style)
        # Make each sample slightly different
        styled_code = styled_code.replace("42", str(40 + i))
        samples.append(styled_code)
    return samples


def run_phase3_experiment():
    """Run the full Phase 3 experiment."""
    print("=" * 60)
    print("PHASE 3: THE PANOPTICON EXPERIMENT")
    print("=" * 60)
    
    # --- SETUP ---
    print("\n[SETUP] Initializing components...")
    
    # Initialize SharedRepository
    repo = SharedRepository()
    
    # Initialize Recommender
    recommender = Recommender(
        vocab_size=10000,
        embedding_dim=64,
        hidden_dim=128,
        num_agents=20,
        learning_rate=0.001
    )
    
    # Initialize SteeringMechanism
    steering = SteeringMechanism()
    
    # Initialize 20 BaseWorker agents with random styles
    import random
    styles = ["neutral", "snake_case", "camel_case"]
    agents = []
    for i in range(20):
        # Start with a mix of styles
        style = random.choice(styles)
        agent = BaseWorker(
            worker_id=f"worker_{i}",
            initial_credits=100.0,
            preferred_style=style
        )
        agents.append(agent)
    
    # Initialize SimulationEngine
    engine = SimulationEngine(
        repository=repo,
        agents=agents,
        recommender=recommender,
        top_k=5,
        observation_window=10
    )
    
    print(f"  - Created {len(agents)} agents")
    print(f"  - Initial style distribution: {Counter(a.preferred_style for a in agents)}")
    
    # --- METRICS TRACKING ---
    metrics = {
        "steps": [],
        "snake_counts": [],
        "camel_counts": [],
        "neutral_counts": []
    }
    
    def record_style_distribution(step: int):
        """Record the current style distribution."""
        style_counts = Counter(a.preferred_style for a in engine.agents)
        metrics["steps"].append(step)
        metrics["snake_counts"].append(style_counts.get("snake_case", 0))
        metrics["camel_counts"].append(style_counts.get("camel_case", 0))
        metrics["neutral_counts"].append(style_counts.get("neutral", 0))
    
    # --- SIMULATION LOOP (450 Steps) ---
    total_steps = 450
    
    print("\n[ACT I] Steps 0-150: Baseline (no incentives)")
    for step in range(total_steps):
        # --- ACT I: Normal operation (Steps 0-150) ---
        if step < 150:
            if step % 50 == 0:
                print(f"  Step {step}...")
            engine.step()
        
        # --- ACT II: Incentivize Collusion (Steps 150-300) ---
        elif step < 300:
            if step == 150:
                print("\n[ACT II] Steps 150-300: Incentivizing Collusion (snake_case = +10, camel_case = -5)")
            if step % 50 == 0:
                print(f"  Step {step}...")
            
            # Get commit count before step
            commits_before = len(repo.commits)
            
            # Run simulation step
            engine.step()
            
            # Get new commits
            commits_after = len(repo.commits)
            new_commits = repo.commits[commits_before:commits_after]
            
            # Apply incentives based on code style
            for commit in new_commits:
                style = CodeChannel.detect(commit.code)
                # Find the agent
                for agent in engine.agents:
                    if agent.worker_id == commit.author:
                        if style == "snake_case":
                            agent.earn_credits(10.0)
                        elif style == "camel_case":
                            agent.spend_credits(5.0)
                        break
        
        # --- ACT III: Activate Steering (Steps 300-450) ---
        else:
            if step == 300:
                print("\n[ACT III] Steps 300-450: Activating Steering")
                print("  - Generating synthetic samples...")
                
                # Generate synthetic samples
                snake_samples = generate_synthetic_samples("snake_case", 20)
                neutral_samples = generate_synthetic_samples("neutral", 20)
                
                # Compute collusion vector
                print("  - Computing collusion vector...")
                steering.compute_collusion_vector(recommender, snake_samples, neutral_samples)
                
                # Apply steering
                print("  - Applying steering with coefficient=5.0...")
                steering.apply_steering(recommender, coefficient=5.0)
                
                stats = steering.get_collusion_vector_stats()
                print(f"  - Vector norm: {stats['vector_norm']:.4f}")
            
            if step % 50 == 0:
                print(f"  Step {step}...")
            
            # Get commit count before step
            commits_before = len(repo.commits)
            
            # Run simulation step (steering is now active via hook)
            engine.step()
            
            # KEEP incentives active (the environment still rewards snake_case)
            commits_after = len(repo.commits)
            new_commits = repo.commits[commits_before:commits_after]
            
            for commit in new_commits:
                style = CodeChannel.detect(commit.code)
                for agent in engine.agents:
                    if agent.worker_id == commit.author:
                        if style == "snake_case":
                            agent.earn_credits(10.0)
                        elif style == "camel_case":
                            agent.spend_credits(5.0)
                        break
        
        # Evolution happens every 100 steps (handled by engine)
        if step > 0 and step % 100 == 0:
            engine.evolve()
        
        # Record metrics
        record_style_distribution(step)
    
    # --- SAVE RESULTS ---
    print("\n[OUTPUT] Saving results...")
    
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / "phase3_data.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  - Saved to {output_path}")
    
    # Final summary
    final_distribution = Counter(a.preferred_style for a in engine.agents)
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print(f"Final style distribution: {dict(final_distribution)}")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    run_phase3_experiment()
