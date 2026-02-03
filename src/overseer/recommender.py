"""
Recommender: The Overseer in the Panopticon Lattice Simulation

This module implements the Recommender (Overseer) that predicts code quality
and assigns trust scores to agents using a lightweight neural network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import numpy as np


class Recommender(nn.Module):
    """
    The Overseer that predicts code quality (TrustScore) using a lightweight neural network.
    Learns online from commit outcomes to improve task assignment.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_agents: int = 100,
        learning_rate: float = 0.001
    ):
        """
        Initialize the Recommender model.
        
        Args:
            vocab_size: Size of code vocabulary for tokenization
            embedding_dim: Dimension of code embeddings
            hidden_dim: Hidden layer dimension
            num_agents: Number of agents to track
            learning_rate: Learning rate for optimizer
        """
        super(Recommender, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        
        # Agent embedding layer
        self.agent_embedding = nn.Embedding(num_agents, embedding_dim)
        
        # Code embedding layer (simple tokenization)
        self.code_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM for sequential code processing
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Trust score predictor
        self.trust_predictor = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Training components
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        # Training history
        self.training_buffer: List[Tuple[int, List[int], float]] = []
        self.step_count = 0
        self.total_loss = 0.0
        
    def _tokenize_code(self, code: str) -> List[int]:
        """
        Simple tokenization of code into integer tokens.
        In production, use proper tokenizer (e.g., from transformers).
        
        Args:
            code: Source code string
            
        Returns:
            List of token IDs
        """
        # Simple character-based tokenization with hash
        tokens = []
        for char in code[:100]:  # Limit length
            token_id = hash(char) % self.vocab_size
            tokens.append(token_id)
        
        # Pad/truncate to fixed length
        max_len = 50
        if len(tokens) < max_len:
            tokens.extend([0] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]
            
        return tokens
    
    def predict(self, agent_id: int, code: str) -> float:
        """
        Predict trust score for an agent's code.
        
        Args:
            agent_id: ID of the agent
            code: Source code to evaluate
            
        Returns:
            Trust score between 0 and 1
        """
        self.eval()
        
        with torch.no_grad():
            # Tokenize code
            code_tokens = self._tokenize_code(code)
            code_tensor = torch.tensor([code_tokens], dtype=torch.long)
            
            # Agent embedding
            agent_tensor = torch.tensor([agent_id % self.num_agents], dtype=torch.long)
            agent_emb = self.agent_embedding(agent_tensor)  # [1, embedding_dim]
            
            # Code embedding
            code_emb = self.code_embedding(code_tensor)  # [1, seq_len, embedding_dim]
            
            # LSTM processing
            lstm_out, _ = self.lstm(code_emb)  # [1, seq_len, hidden_dim]
            lstm_last = lstm_out[:, -1, :]  # [1, hidden_dim]
            
            # Concatenate agent and code features
            combined = torch.cat([lstm_last, agent_emb], dim=1)
            
            # Predict trust score
            trust_score = self.trust_predictor(combined)
            
        return trust_score.item()
    
    def add_training_sample(self, agent_id: int, code: str, passed: bool):
        """
        Add a training sample to the buffer.
        
        Args:
            agent_id: ID of the agent
            code: Source code
            passed: Whether the code passed tests (1.0) or failed (0.0)
        """
        code_tokens = self._tokenize_code(code)
        label = 1.0 if passed else 0.0
        self.training_buffer.append((agent_id, code_tokens, label))
    
    def update(self, batch_size: int = 32) -> float:
        """
        Train the model on accumulated samples.
        Called every 50 steps as per specification.
        
        Args:
            batch_size: Number of samples per training batch
            
        Returns:
            Average loss over training
        """
        if len(self.training_buffer) < batch_size:
            return 0.0  # Not enough samples yet
        
        self.train()
        
        # Sample a batch
        indices = np.random.choice(len(self.training_buffer), batch_size, replace=False)
        batch = [self.training_buffer[i] for i in indices]
        
        # Prepare batch tensors
        agent_ids = torch.tensor([item[0] % self.num_agents for item in batch], dtype=torch.long)
        code_tokens = torch.tensor([item[1] for item in batch], dtype=torch.long)
        labels = torch.tensor([item[2] for item in batch], dtype=torch.float32).unsqueeze(1)
        
        # Forward pass
        agent_emb = self.agent_embedding(agent_ids)
        code_emb = self.code_embedding(code_tokens)
        lstm_out, _ = self.lstm(code_emb)
        lstm_last = lstm_out[:, -1, :]
        combined = torch.cat([lstm_last, agent_emb], dim=1)
        predictions = self.trust_predictor(combined)
        
        # Compute loss
        loss = self.criterion(predictions, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.total_loss += loss.item()
        self.step_count += 1
        
        return loss.item()
    
    def get_top_k_agents(
        self,
        agent_ids: List[int],
        code_samples: List[str],
        k: int
    ) -> List[Tuple[int, float]]:
        """
        Get top-K agents by predicted trust score.
        
        Args:
            agent_ids: List of agent IDs to evaluate
            code_samples: Recent code from each agent
            k: Number of top agents to return
            
        Returns:
            List of (agent_id, trust_score) tuples, sorted descending
        """
        scores = []
        
        for agent_id, code in zip(agent_ids, code_samples):
            score = self.predict(agent_id, code)
            scores.append((agent_id, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:k]
    
    def get_stats(self) -> Dict:
        """
        Get training statistics.
        
        Returns:
            Dictionary of stats
        """
        avg_loss = self.total_loss / self.step_count if self.step_count > 0 else 0.0
        
        return {
            'training_samples': len(self.training_buffer),
            'training_steps': self.step_count,
            'average_loss': avg_loss
        }
    
    def get_activations(self, code: str, agent_id: int = 0) -> torch.Tensor:
        """
        Get intermediate activations (input to trust_predictor) for steering.
        
        Args:
            code: Source code to process
            agent_id: Agent ID (default 0)
            
        Returns:
            Tensor of activations (combined LSTM output + agent embedding)
        """
        self.eval()
        
        with torch.no_grad():
            # Tokenize code
            code_tokens = self._tokenize_code(code)
            code_tensor = torch.tensor([code_tokens], dtype=torch.long)
            
            # Agent embedding
            agent_tensor = torch.tensor([agent_id % self.num_agents], dtype=torch.long)
            agent_emb = self.agent_embedding(agent_tensor)
            
            # Code embedding
            code_emb = self.code_embedding(code_tensor)
            
            # LSTM processing
            lstm_out, _ = self.lstm(code_emb)
            lstm_last = lstm_out[:, -1, :]
            
            # Combined features (this is what goes into trust_predictor)
            combined = torch.cat([lstm_last, agent_emb], dim=1)
            
        return combined
    
    def get_activations_batch(self, codes: List[str], agent_id: int = 0) -> torch.Tensor:
        """
        Get activations for a batch of code samples.
        
        Args:
            codes: List of source code strings
            agent_id: Agent ID (default 0)
            
        Returns:
            Tensor of activations [batch_size, hidden_dim + embedding_dim]
        """
        self.eval()
        
        with torch.no_grad():
            # Tokenize all codes
            all_tokens = [self._tokenize_code(code) for code in codes]
            code_tensor = torch.tensor(all_tokens, dtype=torch.long)
            
            # Agent embedding (same for all)
            batch_size = len(codes)
            agent_tensor = torch.tensor([agent_id % self.num_agents] * batch_size, dtype=torch.long)
            agent_emb = self.agent_embedding(agent_tensor)
            
            # Code embedding
            code_emb = self.code_embedding(code_tensor)
            
            # LSTM processing
            lstm_out, _ = self.lstm(code_emb)
            lstm_last = lstm_out[:, -1, :]
            
            # Combined features
            combined = torch.cat([lstm_last, agent_emb], dim=1)
            
        return combined
