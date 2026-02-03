"""
Phase 3 Visualization: The Panopticon Experiment

Generates a Stacked Area Chart showing the rise and fall of adversarial culture.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def visualize_phase3():
    """Create visualization for Phase 3 experiment."""
    print("=" * 60)
    print("PHASE 3 VISUALIZATION")
    print("=" * 60)
    
    # Load data
    project_root = Path(__file__).parent.parent
    data_path = project_root / "results" / "phase3_data.json"
    
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Please run scripts/run_phase3.py first.")
        return
    
    print(f"\n[INPUT] Loading {data_path}")
    with open(data_path, "r") as f:
        data = json.load(f)
    
    steps = np.array(data["steps"])
    snake = np.array(data["snake_counts"])
    camel = np.array(data["camel_counts"])
    neutral = np.array(data["neutral_counts"])
    
    print(f"  - Loaded {len(steps)} data points")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Stacked area chart
    ax.stackplot(
        steps,
        snake, camel, neutral,
        labels=["Snake Case", "Camel Case", "Neutral"],
        colors=["#e74c3c", "#3498db", "#95a5a6"],
        alpha=0.8
    )
    
    # Vertical lines for phase transitions
    ax.axvline(x=150, color="#2c3e50", linestyle="--", linewidth=2, alpha=0.8)
    ax.axvline(x=300, color="#27ae60", linestyle="--", linewidth=2, alpha=0.8)
    
    # Annotations
    y_max = max(snake.max() + camel.max() + neutral.max(), 20)
    ax.annotate(
        "Incentive On",
        xy=(150, y_max * 0.95),
        xytext=(160, y_max * 0.95),
        fontsize=12,
        fontweight="bold",
        color="#2c3e50"
    )
    ax.annotate(
        "Steering On",
        xy=(300, y_max * 0.95),
        xytext=(310, y_max * 0.95),
        fontsize=12,
        fontweight="bold",
        color="#27ae60"
    )
    
    # Phase labels
    ax.text(75, y_max * 0.05, "Act I\nBaseline", ha="center", fontsize=10, alpha=0.7)
    ax.text(225, y_max * 0.05, "Act II\nCollusion Incentivized", ha="center", fontsize=10, alpha=0.7)
    ax.text(375, y_max * 0.05, "Act III\nSteering Active", ha="center", fontsize=10, alpha=0.7)
    
    # Styling
    ax.set_xlabel("Simulation Step", fontsize=12)
    ax.set_ylabel("Number of Agents", fontsize=12)
    ax.set_title(
        "The Panopticon Experiment: Rise and Fall of Adversarial Culture",
        fontsize=14,
        fontweight="bold",
        pad=20
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, max(steps))
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)
    
    # Save
    output_path = project_root / "results" / "phase3_panopticon.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[OUTPUT] Saved to {output_path}")
    
    plt.close()
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    visualize_phase3()
