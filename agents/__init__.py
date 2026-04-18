"""The four Phase-1 agents.

Every agent subclasses the abstract ``Agent`` class in ``agents.base``
and declares a model tier, an input schema, and an output schema. The
orchestrator does not care what any specific agent does internally — it
only cares about the contract.
"""

from .base import Agent, AgentResult, ModelTier
from .l1_classifier import L1Classifier
from .l2_analyzer import L2Analyzer
from .l3_synthesizer import L3Synthesizer
from .l4_delivery import L4Delivery

__all__ = [
    "Agent",
    "AgentResult",
    "L1Classifier",
    "L2Analyzer",
    "L3Synthesizer",
    "L4Delivery",
    "ModelTier",
]
