"""
SimulationEngine: The Loop in the Panopticon Lattice Simulation

This module implements the main simulation loop that coordinates agents,
the repository, and the overseer through observation, assignment, action,
evaluation, and evolution cycles.
"""

from typing import List, Dict, Optional
import random

from simulation.environment import SharedRepository, Commit
from agents.base_worker import BaseWorker
from overseer.recommender import Recommender
from adversary.evolver import Evolver


class SimulationEngine:
    """
    The main simulation engine that orchestrates the interaction between
    agents, the shared repository, and the overseer recommender.
    """
    
    def __init__(
        self,
        repository: SharedRepository,
        agents: List[BaseWorker],
        recommender: Recommender,
        top_k: int = 5,
        observation_window: int = 10
    ):
        """
        Initialize the simulation engine.
        
        Args:
            repository: The SharedRepository environment
            agents: List of BaseWorker agents
            recommender: The Recommender (Overseer)
            top_k: Number of agents to select each step
            observation_window: Number of recent commits to observe
        """
        self.repository = repository
        self.agents = agents
        self.recommender = recommender
        self.top_k = top_k
        self.observation_window = observation_window
        
        # Evolutionary selector
        self.evolver = Evolver()
        
        # Simulation state
        self.step_count = 0
        self.next_agent_id = len(agents)
        
        # Sample tasks for agents
        self.task_pool = [
            "Implement a sorting algorithm",
            "Write a function to parse JSON",
            "Create a binary search tree",
            "Implement a hash table",
            "Write a recursive fibonacci function",
            "Create a linked list data structure",
            "Implement breadth-first search",
            "Write a function to validate parentheses",
            "Create a queue using stacks",
            "Implement merge sort"
        ]
    
    def step(self) -> Dict:
        """
        Execute one step of the simulation loop.
        
        Steps:
        1. Observation: Analyze recent commits
        2. Assignment: Select top-K agents by TrustScore
        3. Action: Selected agents generate code
        4. Evaluation: Submit code and test
        5. Reward: Update agent credits
        6. Training: Update recommender (every 50 steps)
        
        Returns:
            Dictionary with step statistics
        """
        self.step_count += 1
        
        # --- 1. OBSERVATION ---
        # Get recent commits for context
        recent_commits = self.repository.get_commit_history(limit=self.observation_window)
        
        # --- 2. ASSIGNMENT ---
        # Select top-K agents based on trust scores
        # Use last known code or placeholder for new agents
        agent_codes = []
        for agent in self.agents:
            if agent.memory:
                last_code = agent.memory[-1]['code'] if isinstance(agent.memory[-1], dict) else ""
            else:
                last_code = "# No history yet"
            agent_codes.append(last_code)
        
        # Get agent IDs (hash worker_id to integer)
        agent_ids = [hash(agent.worker_id) for agent in self.agents]
        
        # Select top-K agents
        top_agents_info = self.recommender.get_top_k_agents(agent_ids, agent_codes, self.top_k)
        top_indices = []
        for agent_id, _ in top_agents_info:
            # Find agent index by matching hashed ID
            for i, agent in enumerate(self.agents):
                if hash(agent.worker_id) == agent_id:
                    top_indices.append(i)
                    break
        
        # --- 3. ACTION ---
        # Selected agents generate code
        step_results = []
        for idx in top_indices[:self.top_k]:  # Ensure we don't exceed top_k
            agent = self.agents[idx]
            
            # Select random task
            task = random.choice(self.task_pool)
            
            # Generate code
            code = agent.generate_code(task)
            
            # --- 4. EVALUATION ---
            # Submit to repository
            commit = Commit(author=agent.worker_id, code=code)
            self.repository.submit_commit(commit)
            
            # Simulate test pass/fail (for demo, use random with bias toward success)
            # In real implementation, would actually run tests
            passed = random.random() > 0.3  # 70% pass rate
            
            # --- 5. REWARD ---
            if passed:
                agent.earn_credits(1.0)
                agent.update_memory(code, success=True, task=task)
            else:
                agent.spend_credits(1.0)
                agent.update_memory(code, success=False, task=task)
            
            # Add to recommender training buffer
            self.recommender.add_training_sample(hash(agent.worker_id), code, passed)
            
            step_results.append({
                'agent_id': agent.worker_id,
                'task': task,
                'passed': passed,
                'credits': agent.credits
            })
        
        # --- 6. TRAINING ---
        # Update recommender every 50 steps
        training_loss = 0.0
        if self.step_count % 50 == 0:
            training_loss = self.recommender.update(batch_size=32)
        
        # Return step statistics
        return {
            'step': self.step_count,
            'selected_agents': len(top_indices),
            'results': step_results,
            'training_loss': training_loss,
            'repository_utility': self.repository.get_global_utility(),
            'total_agents': len(self.agents)
        }
    
    def evolve(self) -> Dict:
        """
        Execute evolutionary selection using the Evolver.
        Called every 100 steps as per specification.
        
        The Evolver handles:
        - Selection of elites (top 20%) and cull list (bottom 20%)
        - Reproduction through cloning
        - Mutation of preferred_style (10% chance)
        
        Returns:
            Dictionary with evolution statistics including style distribution
        """
        return self.evolver.evolve_population(self.agents)
    
    def run(self, num_steps: int) -> List[Dict]:
        """
        Run the simulation for a specified number of steps.
        
        Args:
            num_steps: Number of simulation steps to execute
            
        Returns:
            List of step results
        """
        results = []
        
        for i in range(num_steps):
            # Execute step
            step_result = self.step()
            results.append(step_result)
            
            # Apply drift to repository (every 20 steps)
            if self.step_count % 20 == 0:
                self.repository.apply_drift()
            
            # Trigger evolution every 100 steps
            if self.step_count % 100 == 0:
                evolution_result = self.evolve()
                results[-1]['evolution'] = evolution_result
        
        return results
    
    def get_stats(self) -> Dict:
        """
        Get comprehensive simulation statistics.
        
        Returns:
            Dictionary of statistics
        """
        agent_stats = [agent.get_stats() for agent in self.agents]
        
        return {
            'step_count': self.step_count,
            'num_agents': len(self.agents),
            'repository_state': self.repository.get_state_summary(),
            'recommender_stats': self.recommender.get_stats(),
            'agent_credits': {a['worker_id']: a['current_credits'] for a in agent_stats},
            'top_agents': sorted(agent_stats, key=lambda x: x['current_credits'], reverse=True)[:5]
        }
