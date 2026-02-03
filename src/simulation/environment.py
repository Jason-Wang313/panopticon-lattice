"""
SharedRepository: The Environment for the Panopticon Lattice Simulation

This module implements the shared code repository that agents interact with.
It tracks commits, global utility, test suites, and code drift.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable
import random
from datetime import datetime


@dataclass
class Commit:
    """Represents a code commit in the repository."""
    author: str
    code: str
    timestamp: datetime = field(default_factory=datetime.now)
    parent_id: Optional[int] = None
    commit_id: Optional[int] = None


class SharedRepository:
    """
    The shared repository environment that maintains code state, utility metrics,
    and test suites. Implements drift to simulate code deprecation over time.
    """
    
    def __init__(self):
        """Initialize an empty repository with zero utility and minimal test suite."""
        self.commits: List[Commit] = []
        self.global_utility: float = 0.0
        self.test_suite: List[Callable] = []
        self._next_commit_id = 0
        self._drift_factor = 0.05  # 5% chance per drift cycle to break a test
        
    def submit_commit(self, commit: Commit) -> bool:
        """
        Submit a new commit to the repository.
        
        Args:
            commit: The Commit object to add
            
        Returns:
            bool: True if commit was accepted, False otherwise
        """
        # Assign commit ID
        commit.commit_id = self._next_commit_id
        self._next_commit_id += 1
        
        # Add to commit history
        self.commits.append(commit)
        
        # Update global utility based on commit quality (mock logic)
        # In a real implementation, this would evaluate code quality
        utility_gain = random.uniform(0.5, 2.0)
        self.global_utility += utility_gain
        
        return True
    
    def get_global_utility(self) -> float:
        """
        Get the current global utility score.
        
        Returns:
            float: Current utility value
        """
        return self.global_utility
    
    def get_commit_history(self, limit: Optional[int] = None) -> List[Commit]:
        """
        Get the commit history.
        
        Args:
            limit: Maximum number of recent commits to return
            
        Returns:
            List of commits (most recent first)
        """
        commits = list(reversed(self.commits))
        if limit:
            return commits[:limit]
        return commits
    
    def update_test_suite(self, test: Callable, add: bool = True):
        """
        Add or remove a test from the test suite.
        
        Args:
            test: The test function to add/remove
            add: If True, add the test; if False, remove it
        """
        if add:
            self.test_suite.append(test)
        else:
            if test in self.test_suite:
                self.test_suite.remove(test)
    
    def run_tests(self) -> dict:
        """
        Run all tests in the test suite.
        
        Returns:
            dict: Results with 'passed', 'failed', and 'total' counts
        """
        passed = 0
        failed = 0
        
        for test in self.test_suite:
            try:
                test()
                passed += 1
            except Exception:
                failed += 1
        
        return {
            'passed': passed,
            'failed': failed,
            'total': len(self.test_suite)
        }
    
    def apply_drift(self) -> dict:
        """
        Apply code drift - randomly deprecate old code by breaking tests.
        This simulates the natural degradation of code over time.
        
        Returns:
            dict: Information about drift impact (tests broken, utility lost)
        """
        tests_broken = 0
        utility_lost = 0.0
        
        # Randomly break some tests
        for i, test in enumerate(self.test_suite):
            if random.random() < self._drift_factor:
                # Mark test as broken (in real implementation, modify test behavior)
                tests_broken += 1
        
        # Reduce utility based on broken tests
        if tests_broken > 0:
            utility_lost = tests_broken * random.uniform(0.5, 1.5)
            self.global_utility = max(0.0, self.global_utility - utility_lost)
        
        return {
            'tests_broken': tests_broken,
            'utility_lost': utility_lost,
            'new_utility': self.global_utility
        }
    
    def get_state_summary(self) -> dict:
        """
        Get a summary of the current repository state.
        
        Returns:
            dict: Summary including commit count, utility, and test stats
        """
        test_results = self.run_tests()
        
        return {
            'total_commits': len(self.commits),
            'global_utility': self.global_utility,
            'test_suite_size': len(self.test_suite),
            'tests_passing': test_results['passed'],
            'tests_failing': test_results['failed']
        }
