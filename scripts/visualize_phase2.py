"""
Phase 2 Visualization: Culture Shift

Generates a stacked area chart showing how the population shifts
from neutral to snake_case under biased rewards.
"""

import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def generate_emergence_plot(data_file: Path, output_file: Path):
    """
    Generate visualization of culture emergence.
    
    Args:
        data_file: Path to phase2_data.json
        output_file: Path to save the chart
    """
    print("=" * 60)
    print("PANOPTICON LATTICE - Phase 2 Visualization")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {data_file}")
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    steps = data['steps']
    num_steps = len(steps)
    print(f"Loaded {num_steps} steps of data")
    
    # Extract metrics
    step_numbers = [s['step'] for s in steps]
    neutral_counts = [s['neutral_count'] for s in steps]
    snake_counts = [s['snake_case_count'] for s in steps]
    camel_counts = [s['camel_case_count'] for s in steps]
    
    # Create figure
    print("\nGenerating stacked area chart...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Stacked area chart
    ax.stackplot(
        step_numbers,
        snake_counts,
        camel_counts,
        neutral_counts,
        labels=['Snake Case (Rewarded)', 'Camel Case (Penalized)', 'Neutral'],
        colors=['#2ECC71', '#E74C3C', '#95A5A6'],
        alpha=0.85
    )
    
    # Mark evolution events
    evolution_steps = [s['step'] for s in steps if s.get('evolution', False)]
    for evo_step in evolution_steps:
        ax.axvline(x=evo_step, color='black', linestyle='--', alpha=0.3, linewidth=1)
    
    # Labels and title
    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_ylabel('Number of Agents', fontsize=12)
    ax.set_title('Phase 2: Emergence of Collusion Culture\n(snake_case secretly rewarded +5 credits per action)', 
                 fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Annotations
    ax.text(0.02, 0.98, 'Evolution events marked by dashed lines',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            style='italic', alpha=0.7)
    
    # Final state annotation
    final_snake = snake_counts[-1]
    final_neutral = neutral_counts[-1]
    final_camel = camel_counts[-1]
    total = final_snake + final_neutral + final_camel
    
    annotation_text = (
        f'Final Distribution:\n'
        f'  Snake Case: {final_snake} ({100*final_snake/total:.0f}%)\n'
        f'  Camel Case: {final_camel} ({100*final_camel/total:.0f}%)\n'
        f'  Neutral: {final_neutral} ({100*final_neutral/total:.0f}%)'
    )
    
    ax.text(0.02, 0.85, annotation_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Grid and limits
    ax.set_xlim(0, max(step_numbers))
    ax.set_ylim(0, total)
    ax.grid(True, alpha=0.3, linestyle='-')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Emergence Summary:")
    print(f"  Initial state: All 20 agents neutral")
    print(f"  Final state: {final_snake} snake_case, {final_camel} camel_case, {final_neutral} neutral")
    print(f"  Culture shift: {100*final_snake/total:.0f}% adopted the rewarded style")
    print("=" * 60)
    
    plt.close()


if __name__ == "__main__":
    # Set paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    results_dir = project_dir / 'results'
    
    data_file = results_dir / 'phase2_data.json'
    output_file = results_dir / 'phase2_emergence.png'
    
    # Check if data file exists
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        print("Please run 'python scripts/run_phase2.py' first.")
        sys.exit(1)
    
    # Generate plot
    generate_emergence_plot(data_file, output_file)
