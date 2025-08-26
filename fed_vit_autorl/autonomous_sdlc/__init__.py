"""Autonomous SDLC Orchestration System - Including Generation 5 Cognitive AI Enhancements."""

# Generation 4 components (existing)
from .sdlc_orchestrator import (
    AutonomousSDLCOrchestrator,
    SDLCConfiguration,
    GenerationTracker,
)
from .research_discovery import (
    AutonomousResearchDiscovery,
    HypothesisGenerator,
    NovelAlgorithmSynthesizer,
)
from .adaptive_code_gen import (
    AdaptiveCodeGenerator,
    IntelligentRefactorer,
    PerformanceOptimizer,
)
from .publication_pipeline import (
    AutonomousPublicationPipeline,
    PatentGenerator,
    ResearchMetricsAnalyzer,
)
from .self_improving_system import (
    SelfImprovingSDLC,
    EvolutionaryFramework,
    ContinuousLearningEngine,
)

# Generation 5 components (new cognitive AI enhancements)
try:
    from .cognitive_orchestrator import (
        CognitiveSDLCOrchestrator,
        CognitiveMechanism,
        ConsciousnessLevel,
        CognitiveState,
    )
    from .meta_learning_engine import (
        MetaLearningEngine,
        LearningTask,
        MetaKnowledge,
        LearningParadigm,
        TaskDomain,
    )
    from .quantum_optimizer import (
        QuantumOptimizationEngine,
        QuantumOptimizationProblem,
        QuantumAlgorithm,
        OptimizationTarget,
    )
    from .neuromorphic_processor import (
        NeuromorphicSDLCProcessor,
        SpikingNeuralNetwork,
        NeuronModel,
        PlasticityRule,
    )
    from .generation5_master import (
        Generation5MasterOrchestrator,
        Generation5Configuration,
        Generation5Result,
        Generation5Capability,
    )
    
    # Generation 5 components available
    GENERATION5_AVAILABLE = True
    
except ImportError as e:
    # Generation 5 components not available due to missing dependencies
    GENERATION5_AVAILABLE = False
    import warnings
    warnings.warn(f"Generation 5 components not available: {e}", ImportWarning)

__all__ = [
    # Generation 4 exports
    "AutonomousSDLCOrchestrator",
    "SDLCConfiguration", 
    "GenerationTracker",
    "AutonomousResearchDiscovery",
    "HypothesisGenerator",
    "NovelAlgorithmSynthesizer",
    "AdaptiveCodeGenerator",
    "IntelligentRefactorer",
    "PerformanceOptimizer",
    "AutonomousPublicationPipeline",
    "PatentGenerator", 
    "ResearchMetricsAnalyzer",
    "SelfImprovingSDLC",
    "EvolutionaryFramework",
    "ContinuousLearningEngine",
]

# Add Generation 5 exports if available
if GENERATION5_AVAILABLE:
    __all__.extend([
        # Generation 5 exports - Cognitive AI Enhancements
        "CognitiveSDLCOrchestrator",
        "CognitiveMechanism",
        "ConsciousnessLevel", 
        "CognitiveState",
        "MetaLearningEngine",
        "LearningTask",
        "MetaKnowledge",
        "LearningParadigm",
        "TaskDomain",
        "QuantumOptimizationEngine",
        "QuantumOptimizationProblem",
        "QuantumAlgorithm",
        "OptimizationTarget",
        "NeuromorphicSDLCProcessor",
        "SpikingNeuralNetwork",
        "NeuronModel",
        "PlasticityRule",
        "Generation5MasterOrchestrator",
        "Generation5Configuration", 
        "Generation5Result",
        "Generation5Capability",
    ])