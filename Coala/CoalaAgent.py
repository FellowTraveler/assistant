
"""
Agent (Abstract Base)
└── CoalaAgent
    ├── DecisionMaker
    │   └── PlanningProcess
    │       ├── ReasoningAction
    │       └── RetrievalAction
    ├── MemoryManager
    │   ├── BaseMemory (Abstract Base)
    │   │   ├── WorkingMemory
    │   │   ├── EpisodicMemory
    │   │   ├── SemanticMemory
    │   │   └── ProceduralMemory
    └── ActionExecutor (Manages execution of)
        ├── InternalAction (Interface)
        │   ├── ReasoningAction
        │   ├── RetrievalAction
        │   └── LearningAction
        └── ExternalAction (Interface)
            └── GroundingAction

AgentInterface (Abstract Base)
├── LLMInterface
└── EnvironmentInterface



Agent (Abstract Base Class)
└── CoalaAgent
    ├── DecisionMaker
    │   └── PlanningProcess (Utilizes ReasoningAction and RetrievalAction for planning)
    ├── MemoryManager
    │   ├── BaseMemory (Abstract Base Class)
    │   │   ├── WorkingMemory
    │   │   ├── EpisodicMemory
    │   │   ├── SemanticMemory
    │   │   └── ProceduralMemory
    └── ActionExecutor
        ├── InternalAction (Abstract Base Class or Interface)
        │   ├── ReasoningAction
        │   ├── RetrievalAction
        │   └── LearningAction
        └── ExternalAction (Abstract Base Class or Interface)
            └── GroundingAction

AgentInterface (Abstract Base Class or Interface)
├── LLMInterface
└── EnvironmentInterface

"""

from Agent import Agent
from DecisionMaker import DecisionMaker
from MemoryManager import MemoryManager
from ActionExecutor import ActionExecutor

class CoalaAgent(Agent):
    def __init__(self):
        self.decision_maker = DecisionMaker(self)
        self.memory_manager = MemoryManager(self)
        self.action_executor = ActionExecutor(self)

