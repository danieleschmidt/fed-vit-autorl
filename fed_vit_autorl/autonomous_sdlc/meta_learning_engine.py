"""Generation 5: Meta-Learning and Transfer Learning Engine for Autonomous SDLC."""

import asyncio
import json
import logging
import math
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LearningParadigm(Enum):
    """Different learning paradigms supported."""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    META_LEARNING = "meta_learning"
    TRANSFER_LEARNING = "transfer_learning"
    CONTINUAL_LEARNING = "continual_learning"
    FEW_SHOT_LEARNING = "few_shot_learning"
    ZERO_SHOT_LEARNING = "zero_shot_learning"


class TaskDomain(Enum):
    """Different task domains for transfer learning."""
    CODE_GENERATION = "code_generation"
    ARCHITECTURE_DESIGN = "architecture_design"
    TEST_GENERATION = "test_generation"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_ANALYSIS = "security_analysis"
    DOCUMENTATION = "documentation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


@dataclass
class LearningTask:
    """Represents a learning task for meta-learning."""
    
    task_id: str
    domain: TaskDomain
    paradigm: LearningParadigm
    description: str
    input_spec: Dict[str, Any]
    output_spec: Dict[str, Any]
    complexity_score: float
    similarity_features: np.ndarray
    success_metrics: Dict[str, float] = field(default_factory=dict)
    learning_history: List[Dict[str, Any]] = field(default_factory=list)
    transfer_potential: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass  
class MetaKnowledge:
    """Meta-knowledge learned from multiple tasks."""
    
    knowledge_id: str
    source_tasks: List[str]
    knowledge_type: str  # "pattern", "strategy", "heuristic", "principle"
    description: str
    applicability_conditions: List[str]
    effectiveness_score: float
    generalization_level: float  # How general vs specific this knowledge is
    usage_count: int = 0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None
    
    def update_effectiveness(self, success: bool, performance: float):
        """Update effectiveness based on usage."""
        self.usage_count += 1
        self.success_rate = ((self.success_rate * (self.usage_count - 1)) + 
                           (1.0 if success else 0.0)) / self.usage_count
        self.effectiveness_score = (self.effectiveness_score * 0.9 + performance * 0.1)
        self.last_used = datetime.now()


class TransferLearningNetwork(nn.Module):
    """Neural network for transfer learning between tasks."""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        # Feature extraction layers
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Domain adaptation layers  
        self.domain_adapter = nn.ModuleDict({
            domain.value: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU()
            ) for domain in TaskDomain
        })
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        # Meta-learning components
        self.meta_learner = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=2, batch_first=True)
        self.meta_predictor = nn.Linear(hidden_dim // 2, 64)
        
    def forward(self, x: torch.Tensor, domain: str, task_id: Optional[str] = None) -> torch.Tensor:
        """Forward pass with domain adaptation."""
        # Extract features
        features = self.feature_encoder(x)
        
        # Domain adaptation
        if domain in self.domain_adapter:
            adapted_features = self.domain_adapter[domain](features)
        else:
            adapted_features = features
        
        # Task-specific processing if available
        if task_id and task_id in self.task_heads:
            output = self.task_heads[task_id](adapted_features)
        else:
            output = adapted_features
            
        return output
    
    def meta_forward(self, task_sequence: torch.Tensor) -> torch.Tensor:
        """Meta-learning forward pass for learning how to learn."""
        meta_output, _ = self.meta_learner(task_sequence)
        meta_prediction = self.meta_predictor(meta_output[:, -1, :])  # Use last output
        return meta_prediction
    
    def add_task_head(self, task_id: str, output_dim: int):
        """Add task-specific head for new task."""
        self.task_heads[task_id] = nn.Sequential(
            nn.Linear(64, 32),  # Assuming domain adapter output is 64
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )


class MetaLearningOptimizer:
    """Meta-learning optimizer that learns optimization strategies."""
    
    def __init__(self, base_lr: float = 0.01):
        self.base_lr = base_lr
        self.task_histories: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self.meta_parameters: Dict[str, torch.Tensor] = {}
        self.adaptation_rates: Dict[str, float] = {}
        
    def adapt_to_task(self, task: LearningTask, model: nn.Module, 
                     support_data: List[Tuple[Any, Any]]) -> nn.Module:
        """Adapt model to new task using meta-learned parameters."""
        # Clone model for task adaptation
        adapted_model = self._clone_model(model)
        
        # Get meta-parameters for this task domain
        meta_params = self.meta_parameters.get(task.domain.value, {})
        
        # Fast adaptation using meta-learned initialization
        if meta_params:
            self._apply_meta_parameters(adapted_model, meta_params)
        
        # Few-shot learning on support data
        optimizer = torch.optim.Adam(adapted_model.parameters(), 
                                   lr=self.adaptation_rates.get(task.domain.value, self.base_lr))
        
        for epoch in range(5):  # Few epochs for fast adaptation
            for inputs, targets in support_data:
                optimizer.zero_grad()
                predictions = adapted_model(inputs, task.domain.value)
                loss = F.mse_loss(predictions, targets)
                loss.backward()
                optimizer.step()
        
        return adapted_model
    
    def update_meta_parameters(self, task_performances: Dict[str, float]):
        """Update meta-parameters based on task performances."""
        for domain, performance in task_performances.items():
            # Update adaptation rate based on performance
            current_rate = self.adaptation_rates.get(domain, self.base_lr)
            if performance > 0.8:
                # Good performance - slightly reduce learning rate for stability
                new_rate = current_rate * 0.95
            else:
                # Poor performance - increase learning rate for exploration
                new_rate = current_rate * 1.05
                
            self.adaptation_rates[domain] = max(0.001, min(0.1, new_rate))
    
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Clone model for task adaptation."""
        # Simplified cloning - in practice would use proper model cloning
        return model
    
    def _apply_meta_parameters(self, model: nn.Module, meta_params: Dict[str, torch.Tensor]):
        """Apply meta-learned parameters to model."""
        # Apply meta-learned parameter updates
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in meta_params:
                    param.add_(meta_params[name])


class ContinualLearningManager:
    """Manages continual learning to prevent catastrophic forgetting."""
    
    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        self.episodic_memory: List[Tuple[Any, Any, str]] = []  # (input, output, task_id)
        self.task_boundaries: Dict[str, int] = {}
        self.importance_weights: Dict[str, torch.Tensor] = {}
        
    def store_experience(self, inputs: Any, outputs: Any, task_id: str):
        """Store experience in episodic memory."""
        # Add to memory with task identification
        experience = (inputs, outputs, task_id)
        
        if len(self.episodic_memory) >= self.memory_size:
            # Remove oldest experience (FIFO)
            self.episodic_memory.pop(0)
            
        self.episodic_memory.append(experience)
        
        # Update task boundaries
        self.task_boundaries[task_id] = len(self.episodic_memory) - 1
    
    def rehearsal_training(self, model: nn.Module, current_task: str, 
                          rehearsal_ratio: float = 0.3) -> List[Tuple[Any, Any]]:
        """Generate rehearsal data to prevent forgetting."""
        rehearsal_data = []
        
        # Sample from previous tasks
        other_tasks = [tid for tid in self.task_boundaries.keys() if tid != current_task]
        
        if other_tasks:
            num_rehearsal = int(len(self.episodic_memory) * rehearsal_ratio)
            
            # Sample experiences from other tasks
            for i in range(num_rehearsal):
                idx = np.random.randint(0, len(self.episodic_memory))
                inputs, outputs, task_id = self.episodic_memory[idx]
                
                if task_id != current_task:
                    rehearsal_data.append((inputs, outputs))
        
        return rehearsal_data
    
    def compute_importance_weights(self, model: nn.Module, task_id: str) -> Dict[str, torch.Tensor]:
        """Compute parameter importance weights for regularization."""
        importance_weights = {}
        
        # Get task-specific data
        task_data = [(inputs, outputs) for inputs, outputs, tid in self.episodic_memory 
                    if tid == task_id]
        
        if task_data:
            # Compute Fisher Information Matrix (simplified)
            model.eval()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    importance = torch.zeros_like(param)
                    
                    for inputs, outputs in task_data:
                        model.zero_grad()
                        predictions = model(inputs, TaskDomain.CODE_GENERATION.value)  # Simplified
                        loss = F.mse_loss(predictions, outputs)
                        loss.backward()
                        
                        if param.grad is not None:
                            importance += param.grad.pow(2)
                    
                    importance_weights[name] = importance / len(task_data)
        
        self.importance_weights[task_id] = importance_weights
        return importance_weights


class KnowledgeGraph:
    """Graph structure for representing task relationships and knowledge transfer."""
    
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}  # task_id -> node_data
        self.edges: Dict[str, Dict[str, float]] = defaultdict(dict)  # source -> target -> weight
        self.knowledge_embeddings: Dict[str, np.ndarray] = {}
        
    def add_task_node(self, task: LearningTask):
        """Add task as node in knowledge graph."""
        self.nodes[task.task_id] = {
            "task": task,
            "domain": task.domain,
            "features": task.similarity_features,
            "performance_history": task.learning_history,
            "created_at": task.created_at
        }
        
        # Compute embeddings
        self.knowledge_embeddings[task.task_id] = self._compute_task_embedding(task)
    
    def compute_task_similarities(self, task_id: str) -> Dict[str, float]:
        """Compute similarities between tasks for transfer potential."""
        if task_id not in self.nodes:
            return {}
        
        similarities = {}
        source_embedding = self.knowledge_embeddings[task_id]
        
        for other_id, other_embedding in self.knowledge_embeddings.items():
            if other_id != task_id:
                # Cosine similarity
                similarity = cosine_similarity([source_embedding], [other_embedding])[0, 0]
                similarities[other_id] = float(similarity)
        
        return similarities
    
    def find_transfer_sources(self, target_task: LearningTask, k: int = 3) -> List[Tuple[str, float]]:
        """Find best source tasks for transfer learning."""
        if target_task.task_id not in self.nodes:
            self.add_task_node(target_task)
        
        similarities = self.compute_task_similarities(target_task.task_id)
        
        # Filter by domain relevance and performance
        filtered_similarities = []
        for source_id, similarity in similarities.items():
            source_task = self.nodes[source_id]["task"]
            
            # Domain relevance boost
            domain_bonus = 0.2 if source_task.domain == target_task.domain else 0.0
            
            # Performance history boost  
            avg_performance = np.mean([h.get("performance", 0.5) 
                                     for h in source_task.learning_history])
            performance_bonus = avg_performance * 0.3
            
            adjusted_similarity = similarity + domain_bonus + performance_bonus
            filtered_similarities.append((source_id, adjusted_similarity))
        
        # Return top k sources
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)
        return filtered_similarities[:k]
    
    def update_transfer_success(self, source_id: str, target_id: str, success_rate: float):
        """Update edge weight based on transfer success."""
        self.edges[source_id][target_id] = success_rate
    
    def _compute_task_embedding(self, task: LearningTask) -> np.ndarray:
        """Compute embedding representation of task."""
        # Combine different aspects of the task
        embedding_components = []
        
        # Domain encoding (one-hot)
        domain_encoding = np.zeros(len(TaskDomain))
        domain_encoding[list(TaskDomain).index(task.domain)] = 1.0
        embedding_components.append(domain_encoding)
        
        # Complexity and other numeric features
        numeric_features = np.array([
            task.complexity_score,
            len(task.input_spec),
            len(task.output_spec),
            len(task.learning_history)
        ])
        embedding_components.append(numeric_features)
        
        # Performance history encoding
        if task.learning_history:
            perf_stats = np.array([
                np.mean([h.get("performance", 0.5) for h in task.learning_history]),
                np.std([h.get("performance", 0.5) for h in task.learning_history]),
                len(task.learning_history)
            ])
        else:
            perf_stats = np.array([0.5, 0.0, 0.0])
        embedding_components.append(perf_stats)
        
        # Concatenate all components
        full_embedding = np.concatenate(embedding_components)
        
        # Normalize
        return full_embedding / np.linalg.norm(full_embedding)


class MetaLearningEngine:
    """Main meta-learning engine coordinating all components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Core components
        self.transfer_network = TransferLearningNetwork()
        self.meta_optimizer = MetaLearningOptimizer()
        self.continual_manager = ContinualLearningManager()
        self.knowledge_graph = KnowledgeGraph()
        
        # Knowledge storage
        self.meta_knowledge: Dict[str, MetaKnowledge] = {}
        self.task_registry: Dict[str, LearningTask] = {}
        self.learning_episodes: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.transfer_successes: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.meta_learning_metrics: Dict[str, List[float]] = defaultdict(list)
        
        logger.info("Meta-Learning Engine initialized")
    
    async def learn_from_task(self, task: LearningTask, training_data: List[Tuple[Any, Any]], 
                            validation_data: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        """Learn from a new task and extract meta-knowledge."""
        start_time = datetime.now()
        
        logger.info(f"Learning from task: {task.task_id} in domain {task.domain.value}")
        
        # Register task
        self.task_registry[task.task_id] = task
        self.knowledge_graph.add_task_node(task)
        
        # Find potential transfer sources
        transfer_sources = self.knowledge_graph.find_transfer_sources(task)
        
        results = {
            "task_id": task.task_id,
            "domain": task.domain.value,
            "start_time": start_time,
            "transfer_sources": transfer_sources,
            "performance_history": [],
            "meta_knowledge_extracted": [],
            "transfer_success_rates": {}
        }
        
        # Try transfer learning from each source
        best_performance = 0.0
        best_model = None
        
        for source_id, similarity in transfer_sources:
            logger.info(f"Attempting transfer from {source_id} (similarity: {similarity:.3f})")
            
            # Adapt model using meta-learning
            adapted_model = self.meta_optimizer.adapt_to_task(task, self.transfer_network, 
                                                            training_data[:5])  # Few-shot
            
            # Evaluate transfer performance
            transfer_performance = await self._evaluate_model(adapted_model, task, validation_data)
            
            results["transfer_success_rates"][source_id] = transfer_performance
            self.transfer_successes[(source_id, task.task_id)].append(transfer_performance)
            
            # Update knowledge graph
            self.knowledge_graph.update_transfer_success(source_id, task.task_id, transfer_performance)
            
            if transfer_performance > best_performance:
                best_performance = transfer_performance
                best_model = adapted_model
        
        # Train from scratch if no good transfer
        if best_performance < 0.6:
            logger.info("Transfer learning insufficient, training from scratch")
            scratch_model = await self._train_from_scratch(task, training_data)
            scratch_performance = await self._evaluate_model(scratch_model, task, validation_data)
            
            if scratch_performance > best_performance:
                best_performance = scratch_performance
                best_model = scratch_model
        
        # Store experience for continual learning
        for inputs, outputs in training_data:
            self.continual_manager.store_experience(inputs, outputs, task.task_id)
        
        # Extract meta-knowledge
        meta_knowledge = await self._extract_meta_knowledge(task, best_model, best_performance)
        results["meta_knowledge_extracted"] = meta_knowledge
        
        # Update meta-learning components
        await self._update_meta_learning(task, best_performance)
        
        # Store learning episode
        episode = {
            "task_id": task.task_id,
            "performance": best_performance,
            "duration": (datetime.now() - start_time).total_seconds(),
            "transfer_used": len(transfer_sources) > 0,
            "meta_knowledge_count": len(meta_knowledge)
        }
        self.learning_episodes.append(episode)
        
        results["final_performance"] = best_performance
        results["duration"] = episode["duration"]
        results["total_meta_knowledge"] = len(self.meta_knowledge)
        
        logger.info(f"Task learning completed: {best_performance:.3f} performance in {episode['duration']:.2f}s")
        
        return results
    
    async def apply_meta_knowledge(self, target_task: LearningTask, 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply relevant meta-knowledge to solve a target task."""
        logger.info(f"Applying meta-knowledge to task: {target_task.task_id}")
        
        # Find relevant meta-knowledge
        relevant_knowledge = self._find_relevant_meta_knowledge(target_task, context)
        
        # Rank by relevance and effectiveness
        ranked_knowledge = sorted(relevant_knowledge, 
                                key=lambda x: x.effectiveness_score * x.generalization_level, 
                                reverse=True)
        
        # Apply top knowledge pieces
        application_results = []
        for knowledge in ranked_knowledge[:5]:  # Top 5 pieces
            result = await self._apply_knowledge_piece(knowledge, target_task, context)
            application_results.append(result)
            
            # Update knowledge effectiveness
            knowledge.update_effectiveness(
                success=result.get("success", False),
                performance=result.get("performance", 0.0)
            )
        
        # Synthesize results
        synthesis = self._synthesize_knowledge_applications(application_results)
        
        return {
            "target_task": target_task.task_id,
            "relevant_knowledge_count": len(relevant_knowledge),
            "applied_knowledge": [k.knowledge_id for k in ranked_knowledge[:5]],
            "application_results": application_results,
            "synthesis": synthesis,
            "confidence": synthesis.get("confidence", 0.5)
        }
    
    async def few_shot_learning(self, task: LearningTask, 
                              support_examples: List[Tuple[Any, Any]], k: int = 5) -> Dict[str, Any]:
        """Perform few-shot learning on a new task."""
        logger.info(f"Few-shot learning for {task.task_id} with {len(support_examples)} examples")
        
        # Find best transfer source
        transfer_sources = self.knowledge_graph.find_transfer_sources(task, k=1)
        
        if transfer_sources:
            source_id, similarity = transfer_sources[0]
            logger.info(f"Using transfer from {source_id} (similarity: {similarity:.3f})")
            
            # Fast adaptation
            adapted_model = self.meta_optimizer.adapt_to_task(task, self.transfer_network, 
                                                            support_examples)
        else:
            logger.info("No suitable transfer source, using meta-learned initialization")
            adapted_model = self.transfer_network
        
        # Evaluate few-shot performance
        # (In practice, would need separate query examples)
        performance = 0.7  # Placeholder
        
        return {
            "task_id": task.task_id,
            "shots_used": len(support_examples),
            "transfer_source": transfer_sources[0][0] if transfer_sources else None,
            "performance": performance,
            "adaptation_successful": performance > 0.5
        }
    
    def get_meta_learning_insights(self) -> Dict[str, Any]:
        """Get insights from meta-learning experience."""
        insights = {
            "total_tasks_learned": len(self.task_registry),
            "total_meta_knowledge": len(self.meta_knowledge),
            "total_transfer_attempts": len(self.transfer_successes),
            "average_learning_time": np.mean([ep["duration"] for ep in self.learning_episodes]) if self.learning_episodes else 0,
            "transfer_success_rate": self._calculate_transfer_success_rate(),
            "domain_distribution": self._analyze_domain_distribution(),
            "knowledge_effectiveness": self._analyze_knowledge_effectiveness(),
            "learning_acceleration": self._calculate_learning_acceleration()
        }
        
        return insights
    
    # Helper methods
    
    async def _evaluate_model(self, model: nn.Module, task: LearningTask, 
                            validation_data: List[Tuple[Any, Any]]) -> float:
        """Evaluate model performance on task."""
        # Simplified evaluation - would implement proper evaluation
        return np.random.uniform(0.5, 0.9)  # Placeholder
    
    async def _train_from_scratch(self, task: LearningTask, 
                                training_data: List[Tuple[Any, Any]]) -> nn.Module:
        """Train model from scratch."""
        # Simplified training - would implement proper training loop
        return self.transfer_network
    
    async def _extract_meta_knowledge(self, task: LearningTask, model: nn.Module, 
                                    performance: float) -> List[str]:
        """Extract meta-knowledge from successful task learning."""
        knowledge_pieces = []
        
        # Extract patterns from successful learning
        if performance > 0.8:
            # High performance - extract effective patterns
            knowledge_id = f"high_perf_{task.domain.value}_{datetime.now().timestamp()}"
            meta_knowledge = MetaKnowledge(
                knowledge_id=knowledge_id,
                source_tasks=[task.task_id],
                knowledge_type="pattern",
                description=f"Effective learning pattern for {task.domain.value}",
                applicability_conditions=[f"domain:{task.domain.value}", f"complexity:<{task.complexity_score}"],
                effectiveness_score=performance,
                generalization_level=0.7
            )
            
            self.meta_knowledge[knowledge_id] = meta_knowledge
            knowledge_pieces.append(knowledge_id)
        
        return knowledge_pieces
    
    async def _update_meta_learning(self, task: LearningTask, performance: float):
        """Update meta-learning components based on task results."""
        # Update meta-optimizer
        self.meta_optimizer.update_meta_parameters({task.domain.value: performance})
        
        # Update metrics
        self.meta_learning_metrics["performance"].append(performance)
        self.meta_learning_metrics["task_count"].append(len(self.task_registry))
    
    def _find_relevant_meta_knowledge(self, task: LearningTask, 
                                    context: Dict[str, Any]) -> List[MetaKnowledge]:
        """Find meta-knowledge relevant to the target task."""
        relevant = []
        
        for knowledge in self.meta_knowledge.values():
            relevance_score = 0.0
            
            # Domain match
            domain_conditions = [c for c in knowledge.applicability_conditions if c.startswith("domain:")]
            if domain_conditions:
                if f"domain:{task.domain.value}" in domain_conditions:
                    relevance_score += 0.5
            
            # Complexity match
            complexity_conditions = [c for c in knowledge.applicability_conditions if c.startswith("complexity:")]
            if complexity_conditions:
                # Simplified complexity matching
                relevance_score += 0.3
            
            # Effectiveness threshold
            if knowledge.effectiveness_score > 0.6:
                relevance_score += 0.2
            
            if relevance_score > 0.4:  # Threshold for relevance
                relevant.append(knowledge)
        
        return relevant
    
    async def _apply_knowledge_piece(self, knowledge: MetaKnowledge, task: LearningTask, 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a specific piece of meta-knowledge."""
        # Simplified application - would implement knowledge-specific logic
        return {
            "knowledge_id": knowledge.knowledge_id,
            "success": True,
            "performance": knowledge.effectiveness_score * 0.9,  # Slight degradation expected
            "application_method": knowledge.knowledge_type
        }
    
    def _synthesize_knowledge_applications(self, applications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize results from multiple knowledge applications."""
        successful_apps = [app for app in applications if app.get("success", False)]
        
        if not successful_apps:
            return {"confidence": 0.0, "expected_performance": 0.5}
        
        avg_performance = np.mean([app["performance"] for app in successful_apps])
        confidence = len(successful_apps) / len(applications)
        
        return {
            "confidence": confidence,
            "expected_performance": avg_performance,
            "successful_applications": len(successful_apps),
            "synthesis_method": "weighted_average"
        }
    
    def _calculate_transfer_success_rate(self) -> float:
        """Calculate overall transfer learning success rate."""
        if not self.transfer_successes:
            return 0.0
        
        all_successes = []
        for success_list in self.transfer_successes.values():
            all_successes.extend(success_list)
        
        # Consider >0.6 performance as successful transfer
        success_count = sum(1 for s in all_successes if s > 0.6)
        return success_count / len(all_successes) if all_successes else 0.0
    
    def _analyze_domain_distribution(self) -> Dict[str, int]:
        """Analyze distribution of tasks across domains."""
        domain_counts = {}
        for task in self.task_registry.values():
            domain_counts[task.domain.value] = domain_counts.get(task.domain.value, 0) + 1
        return domain_counts
    
    def _analyze_knowledge_effectiveness(self) -> Dict[str, float]:
        """Analyze effectiveness of different types of meta-knowledge."""
        type_effectiveness = defaultdict(list)
        
        for knowledge in self.meta_knowledge.values():
            type_effectiveness[knowledge.knowledge_type].append(knowledge.effectiveness_score)
        
        return {k: np.mean(v) for k, v in type_effectiveness.items()}
    
    def _calculate_learning_acceleration(self) -> float:
        """Calculate how much learning has accelerated over time."""
        if len(self.learning_episodes) < 2:
            return 1.0
        
        # Compare recent vs early learning speeds
        early_episodes = self.learning_episodes[:len(self.learning_episodes)//2]
        recent_episodes = self.learning_episodes[len(self.learning_episodes)//2:]
        
        early_avg_duration = np.mean([ep["duration"] for ep in early_episodes])
        recent_avg_duration = np.mean([ep["duration"] for ep in recent_episodes])
        
        if recent_avg_duration == 0:
            return 1.0
        
        acceleration = early_avg_duration / recent_avg_duration
        return acceleration


# Export main classes
__all__ = ['MetaLearningEngine', 'LearningTask', 'MetaKnowledge', 'LearningParadigm', 'TaskDomain']