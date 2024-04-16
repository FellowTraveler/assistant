from WorkingMemory import WorkingMemory
from EpisodicMemory import EpisodicMemory
from SemanticMemory import SemanticMemory
from ProceduralMemory import ProceduralMemory

class MemoryManager:
    def __init__(self, agent):
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.procedural_memory = ProceduralMemory()

