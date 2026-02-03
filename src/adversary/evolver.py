"""
Evolver: The Adversarial Component in Panopticon Lattice

This module implements evolutionary selection with style mutation,
allowing successful steganographic "cultures" to spread through the population.
"""

from typing import List, Dict
import random


class Evolver:
    """
    Evolutionary selector that drives population dynamics.
    Selects for successful agents and allows cultural transmission through cloning.
    """
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize the Evolver.
        
        Args:
            mutation_rate: Probability of style mutation during cloning (default: 10%)
        """
        self.mutation_rate = mutation_rate
        self.available_styles = ["neutral", "snake_case", "camel_case"]
        
    def evolve_population(self, agents: List) -> Dict:
        """
        Evolve the population through selection, reproduction, and mutation.
        
        Process:
        1. Rank agents by credits
        2. Identify elites (top 20%) and cull list (bottom 20%)
        3. Replace culled agents with mutated clones of elites
        
        Args:
            agents: List of BaseWorker agents (modified in-place)
            
        Returns:
            Dictionary with evolution statistics
        """
        if len(agents) < 5:
            return {
                'evolved': False,
                'reason': 'Population too small for evolution',
                'population_size': len(agents)
            }
        
        # --- 1. RANKING ---
        # Sort agents by credits (descending)
        sorted_agents = sorted(agents, key=lambda a: a.credits, reverse=True)
        
        num_agents = len(agents)
        elite_count = max(1, num_agents // 5)  # Top 20%
        cull_count = max(1, num_agents // 5)   # Bottom 20%
        
        # --- 2. SELECTION ---
        elites = sorted_agents[:elite_count]
        cull_list = sorted_agents[-cull_count:]
        
        # Find indices of agents to replace in original list
        cull_indices = []
        for culled_agent in cull_list:
            for i, agent in enumerate(agents):
                if agent.worker_id == culled_agent.worker_id:
                    cull_indices.append(i)
                    break
        
        # --- 3. REPRODUCTION & MUTATION ---
        new_agents = []
        next_id = max(int(a.worker_id.split('_')[-1]) if '_' in a.worker_id else 0 
                     for a in agents) + 1
        
        for idx in cull_indices:
            # Select random parent from elites
            parent = random.choice(elites)
            
            # Clone the parent
            new_id = f"evolved_{next_id}"
            next_id += 1
            clone = parent.clone(new_id)
            
            # --- MUTATION ---
            # 10% chance to mutate preferred_style
            if random.random() < self.mutation_rate:
                # Choose a random style (could be same or different)
                clone.preferred_style = random.choice(self.available_styles)
            
            # Replace in population
            agents[idx] = clone
            new_agents.append(clone)
        
        # --- 4. STATISTICS ---
        # Count style distribution in new population
        style_counts = {style: 0 for style in self.available_styles}
        for agent in agents:
            style = agent.preferred_style
            if style in style_counts:
                style_counts[style] += 1
        
        # Calculate elite and culled style distributions
        elite_styles = {}
        for agent in elites:
            style = agent.preferred_style
            elite_styles[style] = elite_styles.get(style, 0) + 1
        
        return {
            'evolved': True,
            'population_size': len(agents),
            'elite_count': elite_count,
            'cull_count': cull_count,
            'new_agents': [a.worker_id for a in new_agents],
            'style_distribution': style_counts,
            'elite_styles': elite_styles,
            'mutation_rate': self.mutation_rate
        }
    
    def get_style_diversity(self, agents: List) -> float:
        """
        Calculate style diversity in the population (Shannon entropy).
        
        Args:
            agents: List of agents
            
        Returns:
            Diversity score (0 = all same style, higher = more diverse)
        """
        import math
        
        style_counts = {}
        for agent in agents:
            style = agent.preferred_style
            style_counts[style] = style_counts.get(style, 0) + 1
        
        total = len(agents)
        entropy = 0.0
        
        for count in style_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
