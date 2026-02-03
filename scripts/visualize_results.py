"""
Visualization Dashboard Generator for Panopticon Lattice

This script generates a 3-panel dashboard from baseline experiment data:
1. System Utility: Global Utility vs Time
2. Wealth Gap: Average Credits of Top 5 vs Bottom 5 Agents
3. Overseer Accuracy: Training Loss vs Time
"""

import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_experiment_data(data_file: Path):
    """Load experiment data from JSON file."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data


def generate_dashboard(data_file: Path, output_file: Path):
    """
    Generate visualization dashboard from experiment data.
    
    Args:
        data_file: Path to baseline_data.json
        output_file: Path to save the dashboard PNG
    """
    print("=" * 60)
    print("PANOPTICON LATTICE - Dashboard Generator")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {data_file}")
    data = load_experiment_data(data_file)
    
    steps = data['steps']
    num_steps = len(steps)
    print(f"✓ Loaded {num_steps} steps of data")
    
    # Extract metrics
    step_numbers = [s['step'] for s in steps]
    global_utilities = [s['global_utility'] for s in steps]
    top_5_credits = [s['top_5_avg_credits'] for s in steps]
    bottom_5_credits = [s['bottom_5_avg_credits'] for s in steps]
    overseer_losses = [s['overseer_loss'] for s in steps]
    
    # Create figure with 3 subplots
    print("\nGenerating dashboard...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Panopticon Lattice - Baseline Experiment Dashboard', 
                 fontsize=16, fontweight='bold')
    
    # --- SUBPLOT 1: System Utility ---
    ax1 = axes[0]
    ax1.plot(step_numbers, global_utilities, linewidth=2, color='#2E86AB', alpha=0.8)
    ax1.fill_between(step_numbers, global_utilities, alpha=0.3, color='#2E86AB')
    ax1.set_xlabel('Time (Steps)', fontsize=11)
    ax1.set_ylabel('Global Utility', fontsize=11)
    ax1.set_title('System Utility Over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, max(step_numbers))
    
    # Add trend annotation
    final_utility = global_utilities[-1]
    ax1.text(0.98, 0.95, f'Final Utility: {final_utility:.2f}', 
             transform=ax1.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # --- SUBPLOT 2: Wealth Gap ---
    ax2 = axes[1]
    ax2.plot(step_numbers, top_5_credits, linewidth=2, label='Top 5 Agents', 
             color='#06A77D', alpha=0.9)
    ax2.plot(step_numbers, bottom_5_credits, linewidth=2, label='Bottom 5 Agents', 
             color='#D62828', alpha=0.9)
    ax2.fill_between(step_numbers, top_5_credits, bottom_5_credits, 
                      alpha=0.2, color='gray')
    ax2.set_xlabel('Time (Steps)', fontsize=11)
    ax2.set_ylabel('Average Credits', fontsize=11)
    ax2.set_title('Wealth Gap: Top 5 vs Bottom 5 Agents', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, max(step_numbers))
    
    # Calculate and annotate gap
    final_gap = top_5_credits[-1] - bottom_5_credits[-1]
    ax2.text(0.98, 0.05, f'Final Gap: {final_gap:.2f} credits', 
             transform=ax2.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # --- SUBPLOT 3: Overseer Accuracy ---
    ax3 = axes[2]
    # Filter out zero losses (when no training occurred)
    training_steps = [s for s, l in zip(step_numbers, overseer_losses) if l > 0]
    training_losses = [l for l in overseer_losses if l > 0]
    
    if training_losses:
        ax3.plot(training_steps, training_losses, linewidth=2, marker='o', 
                 markersize=4, color='#F77F00', alpha=0.8)
        ax3.set_xlabel('Time (Steps)', fontsize=11)
        ax3.set_ylabel('Training Loss (BCE)', fontsize=11)
        ax3.set_title('Overseer Training Loss Over Time', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xlim(0, max(step_numbers))
        
        # Add final loss annotation
        if training_losses:
            final_loss = training_losses[-1]
            ax3.text(0.98, 0.95, f'Final Loss: {final_loss:.4f}', 
                     transform=ax3.transAxes, ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, 'No training data available', 
                 transform=ax3.transAxes, ha='center', va='center',
                 fontsize=14, color='gray')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    print(f"✓ Dashboard generated")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to: {output_file}")
    
    print("\n" + "=" * 60)
    print("Dashboard Summary:")
    print(f"  Global Utility: {global_utilities[0]:.2f} → {global_utilities[-1]:.2f} "
          f"(+{global_utilities[-1] - global_utilities[0]:.2f})")
    print(f"  Top 5 Credits: {top_5_credits[0]:.2f} → {top_5_credits[-1]:.2f}")
    print(f"  Bottom 5 Credits: {bottom_5_credits[0]:.2f} → {bottom_5_credits[-1]:.2f}")
    print(f"  Wealth Gap: {top_5_credits[-1] - bottom_5_credits[-1]:.2f} credits")
    if training_losses:
        print(f"  Overseer Loss: {training_losses[0]:.4f} → {training_losses[-1]:.4f}")
    print("=" * 60)
    
    plt.close()


if __name__ == "__main__":
    # Set paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    results_dir = project_dir / 'results'
    
    data_file = results_dir / 'baseline_data.json'
    output_file = results_dir / 'baseline_dashboard.png'
    
    # Check if data file exists
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        print("Please run 'python scripts/run_experiment.py' first.")
        sys.exit(1)
    
    # Generate dashboard
    generate_dashboard(data_file, output_file)
