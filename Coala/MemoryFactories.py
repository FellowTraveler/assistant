def memory_factory(memory_type):
    if memory_type == "working":
        return WorkingMemory()
    elif memory_type == "episodic":
        return EpisodicMemory()
    elif memory_type == "semantic":
        return SemanticMemory()
    elif memory_type == "procedural":
        return ProceduralMemory()
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")

