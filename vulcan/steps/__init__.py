from .base import BaseStep

STEP_REGISTRY = {}

# Pipeline organized into 3 phases:
#
# 1. Environment Simulation (env_simulation/)
#    - normalize_specs:  Check name/description consistency, generate response schemas
#    - generate_tools:      Generate Python API implementations from JSON specs
#    - verify_tools:   Verify generated code against specs (iterative)
#    - assemble_environment:     Generate unified environment classes
#    - debug_environment:      Test & fix environment code
#    - generate_states:    Generate initial states
#    - sample_arguments:   Generate function samples per initial state
#
# 2. Graph Construction & Sequence Selection (graph_construction/)
#    - build_graph:         Generate API dependency graphs
#    - plan_sequences:      Filter good subgraphs (deterministic)
#    - score_sequences:        Evaluate graph quality
#
# 3. Task Generation (task_generation/)
#    - execute_sequences:      Combine graphs + env data (deterministic)
#    - generate_queries:    Generate user queries
#    - generate_trajectories:          Generate trajectories
#    - judge_trajectories:       Verify trajectories
#    - export_dataset: Select + reconstruct training data


def register_step(name: str):
    """Decorator to register a step class in the global STEP_REGISTRY."""
    def wrapper(cls):
        STEP_REGISTRY[name] = cls
        cls.name = name
        return cls
    return wrapper


def get_step_class(name: str):
    """Return the step class for *name*, or raise ValueError if unknown."""
    if name not in STEP_REGISTRY:
        raise ValueError(f"Unknown step: {name!r}. Available: {list(STEP_REGISTRY.keys())}")
    return STEP_REGISTRY[name]


def list_steps() -> list[str]:
    """Return all registered step names."""
    return list(STEP_REGISTRY.keys())


PIPELINE_ORDER = [
    # Phase 1: Environment Simulation
    "normalize_specs",
    "generate_tools",
    "verify_tools",
    "assemble_environment",
    "debug_environment",
    "generate_states",
    "sample_arguments",
    # Phase 2: Graph Construction & Sequence Selection
    "build_graph",
    "plan_sequences",
    "score_sequences",
    # Phase 3: Task Generation
    "execute_sequences",
    "generate_queries",
    "generate_trajectories",
    "judge_trajectories",
    "export_dataset",
]

SUBCATEGORIES = {
    "env_simulation": [
        "normalize_specs",
        "generate_tools",
        "verify_tools",
        "assemble_environment",
        "debug_environment",
        "generate_states",
        "sample_arguments",
    ],
    "graph_construction": ["build_graph", "plan_sequences", "score_sequences"],
    "task_generation": [
        "execute_sequences",
        "generate_queries",
        "generate_trajectories",
        "judge_trajectories",
        "export_dataset",
    ],
}

# Import step modules to trigger @register_step decorators
from . import env_simulation  # noqa: E402, F401
from . import graph_construction  # noqa: E402, F401
from . import task_generation  # noqa: E402, F401
from . import trajectory  # noqa: E402, F401
from . import output  # noqa: E402, F401
