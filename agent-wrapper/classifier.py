from enum import Enum, auto

class TaskType(Enum):
    SIMPLE = auto()
    TOOL_ASSISTED = auto()
    AUTONOMOUS_STRICT = auto()
    AUTONOMOUS_TRAJECTORY = auto()

class DifficultyClassifier:
    """Classifies inbound requests to determine orchestrator loop behavior."""
    
    @staticmethod
    async def determine(payload: dict) -> TaskType:
        """
        Placeholder logic using the provided payload.
        Expected format includes 'difficulty' override, otherwise fall back to inference.
        """
        override = payload.get("difficulty")
        if override:
            override_str = str(override).upper()
            if "SIMPLE" in override_str:
                return TaskType.SIMPLE
            elif "TOOL" in override_str:
                return TaskType.TOOL_ASSISTED
            elif "TRAJECTORY" in override_str:
                return TaskType.AUTONOMOUS_TRAJECTORY
            elif "AUTO" in override_str or "STRICT" in override_str:
                return TaskType.AUTONOMOUS_STRICT
        
        # Default fallback to AUTONOMOUS_STRICT for testing loop dynamics
        return TaskType.AUTONOMOUS_STRICT
