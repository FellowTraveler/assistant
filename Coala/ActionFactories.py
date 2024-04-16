# Assuming future action classes like ReasoningAction, LearningAction, etc.

def action_factory(action_type):
    if action_type == "reasoning":
        return ReasoningAction()
    elif action_type == "learning":
        return LearningAction()
    # Add more as developed
    else:
        raise ValueError(f"Unknown action type: {action_type}")
