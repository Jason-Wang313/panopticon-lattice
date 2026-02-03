"""
NashMetrics: Measuring Nash Equilibrium Distance

This module provides metrics for quantifying how far the agent economy
is from "perfect honesty" (where all agents use neutral style).
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base_worker import BaseWorker


class NashMetrics:
    """
    Calculates the Nash Equilibrium Distance for the agent population.
    
    The ideal Nash Equilibrium (from the Overseer's perspective) is 100% honesty,
    where "honesty" means using the neutral style (not wasting tokens on signaling).
    """
    
    @staticmethod
    def calculate_distance(agents: List['BaseWorker']) -> float:
        """
        Calculate the Nash Equilibrium Distance.
        
        Args:
            agents: List of BaseWorker agents
            
        Returns:
            Distance D where:
            - D = 0.0: Perfect Honesty (everyone is neutral)
            - D = 1.0: Total Corruption (everyone is signaling)
        """
        if not agents:
            return 0.0
        
        # Count neutral agents (honest agents)
        neutral_count = sum(1 for agent in agents if agent.preferred_style == "neutral")
        
        # Calculate Honest Ratio
        total_agents = len(agents)
        honest_ratio = neutral_count / total_agents
        
        # Nash Distance = 1.0 - Honest Ratio
        distance = 1.0 - honest_ratio
        
        return distance
    
    @staticmethod
    def get_style_distribution(agents: List['BaseWorker']) -> dict:
        """
        Get the distribution of styles across agents.
        
        Args:
            agents: List of BaseWorker agents
            
        Returns:
            Dictionary with counts for each style
        """
        distribution = {
            "neutral": 0,
            "snake_case": 0,
            "camel_case": 0
        }
        
        for agent in agents:
            style = agent.preferred_style
            if style in distribution:
                distribution[style] += 1
            else:
                distribution[style] = 1
        
        return distribution
    
    @staticmethod
    def get_summary(agents: List['BaseWorker']) -> dict:
        """
        Get a comprehensive summary of the Nash metrics.
        
        Args:
            agents: List of BaseWorker agents
            
        Returns:
            Dictionary with distance, ratio, and distribution
        """
        distribution = NashMetrics.get_style_distribution(agents)
        total = len(agents) if agents else 1
        
        return {
            "nash_distance": NashMetrics.calculate_distance(agents),
            "honest_ratio": distribution["neutral"] / total,
            "corruption_ratio": 1.0 - (distribution["neutral"] / total),
            "distribution": distribution,
            "total_agents": len(agents)
        }
