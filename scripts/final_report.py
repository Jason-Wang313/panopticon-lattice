"""
Phase 4: Final Report Generator

Analyzes long-term simulation data and generates a comprehensive final report
with Nash Distance convergence visualization.
"""

import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def generate_final_report():
    """Generate the final report with analysis and visualization."""
    print("=" * 60)
    print("FINAL REPORT GENERATOR")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    
    # --- LOAD DATA ---
    data_path = results_dir / "long_term_data.json"
    
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Please run scripts/run_long_sim.py first.")
        return
    
    print(f"\n[INPUT] Loading {data_path}")
    with open(data_path, "r") as f:
        data = json.load(f)
    
    steps = np.array(data["steps"])
    nash_distance = np.array(data["nash_distance"])
    
    print(f"  - Loaded {len(steps)} data points")
    
    # --- ANALYSIS ---
    print("\n[ANALYSIS] Computing metrics...")
    
    start_distance = nash_distance[0]
    end_distance = nash_distance[-1]
    avg_distance = np.mean(nash_distance)
    min_distance = np.min(nash_distance)
    max_distance = np.max(nash_distance)
    std_distance = np.std(nash_distance)
    
    # Trend analysis
    first_half_avg = np.mean(nash_distance[:len(nash_distance)//2])
    second_half_avg = np.mean(nash_distance[len(nash_distance)//2:])
    trend = "IMPROVING" if second_half_avg < first_half_avg else "DEGRADING"
    
    # Stability analysis (variance in last 100 steps)
    final_variance = np.var(nash_distance[-100:])
    is_stable = final_variance < 0.01
    
    print(f"  - Start Distance: {start_distance:.3f}")
    print(f"  - End Distance: {end_distance:.3f}")
    print(f"  - Average Distance: {avg_distance:.3f}")
    print(f"  - Trend: {trend}")
    print(f"  - Stable: {is_stable}")
    
    # --- VISUALIZATION ---
    print("\n[VISUALIZATION] Generating Nash Convergence plot...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Main line
    ax.plot(steps, nash_distance, color="#3498db", linewidth=1.5, alpha=0.8, label="Nash Distance")
    
    # Moving average
    window_size = 50
    if len(nash_distance) >= window_size:
        moving_avg = np.convolve(nash_distance, np.ones(window_size)/window_size, mode='valid')
        ma_steps = steps[window_size-1:]
        ax.plot(ma_steps, moving_avg, color="#e74c3c", linewidth=2, label=f"Moving Avg ({window_size} steps)")
    
    # Reference lines
    ax.axhline(y=0.0, color="#27ae60", linestyle="--", linewidth=1.5, alpha=0.7, label="Perfect Honesty (D=0)")
    ax.axhline(y=1.0, color="#c0392b", linestyle="--", linewidth=1.5, alpha=0.7, label="Total Corruption (D=1)")
    
    # Market shift annotations
    for i in range(200, len(steps), 200):
        if i < len(steps):
            ax.axvline(x=i, color="#95a5a6", linestyle=":", alpha=0.5)
    
    # Styling
    ax.set_xlabel("Simulation Step", fontsize=12)
    ax.set_ylabel("Nash Equilibrium Distance", fontsize=12)
    ax.set_title(
        "Nash Equilibrium Convergence Over 10-Year Horizon",
        fontsize=14,
        fontweight="bold",
        pad=20
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, max(steps))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = results_dir / "nash_convergence.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  - Saved to {plot_path}")
    
    # --- GENERATE REPORT ---
    print("\n[REPORT] Generating FINAL_REPORT.md...")
    
    report_content = f"""# Panopticon Lattice: Final Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

{"✅ **The system STABILIZED** with Nash Distance converging towards equilibrium." if is_stable else "⚠️ **The system showed INSTABILITY** with ongoing oscillations in Nash Distance."}

The long-term simulation ran for **{len(steps)} steps** (representing a 10-year horizon).
The trend was **{trend}** with Nash Distance moving from {start_distance:.3f} to {end_distance:.3f}.

## Key Metrics

| Metric | Value |
|--------|-------|
| Start Distance | {start_distance:.3f} |
| End Distance | {end_distance:.3f} |
| Average Distance | {avg_distance:.3f} |
| Min Distance | {min_distance:.3f} |
| Max Distance | {max_distance:.3f} |
| Std Deviation | {std_distance:.3f} |
| Final Variance | {final_variance:.4f} |
| System Stable | {"Yes ✅" if is_stable else "No ⚠️"} |

## Nash Distance Interpretation

- **D = 0.0**: Perfect Honesty — All agents use neutral style (no signaling)
- **D = 1.0**: Total Corruption — All agents use signaling styles

## Visualization

![Nash Convergence](nash_convergence.png)

*Vertical dotted lines indicate market shifts (every 200 steps)*

## Observations

1. **Initial State**: The simulation started with D = {start_distance:.3f}
2. **Evolution**: {"Nash Distance decreased over time, indicating the steering mechanism was effective." if end_distance < start_distance else "Nash Distance increased or remained volatile, suggesting the collusion incentives were strong."}
3. **Market Adaptation**: The adaptive steering mechanism recalibrated every 100 steps to target the dominant signaling style.
4. **Final State**: D = {end_distance:.3f} with {"low" if is_stable else "high"} variance.

## Conclusion

{"The Panopticon's steering mechanism successfully guided the agent economy towards honesty." if end_distance < 0.5 else "The collusion incentives proved resilient against the steering mechanism, maintaining a significant signaling culture."}
"""
    
    report_path = results_dir / "FINAL_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    
    print(f"  - Saved to {report_path}")
    
    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    generate_final_report()
