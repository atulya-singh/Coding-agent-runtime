"""Phase 1/5: the model-driven agent loop and its configuration."""
from .client import ModelClient, ModelError, ModelResponse
from .config import (
    CONFIG_FILENAME,
    DEFAULT_MODEL,
    AgentConfig,
    load_config,
    save_config,
    validate_config,
)
from .loop import AgentLoop, AgentRun, StopReason
from .pricing import ModelPricing, cost_usd
from .prompts import build_initial_messages, build_system_prompt
from .run import RunConfig, TaskRun, run_task
from .tools import FINISH_TOOL_NAME, FINISH_TOOL_SPEC, build_tool_specs

__all__ = [
    "AgentConfig",
    "DEFAULT_MODEL",
    "CONFIG_FILENAME",
    "validate_config",
    "load_config",
    "save_config",
    "AgentLoop",
    "AgentRun",
    "StopReason",
    "run_task",
    "RunConfig",
    "TaskRun",
    "ModelClient",
    "ModelResponse",
    "ModelError",
    "ModelPricing",
    "cost_usd",
    "build_system_prompt",
    "build_initial_messages",
    "build_tool_specs",
    "FINISH_TOOL_NAME",
    "FINISH_TOOL_SPEC",
]
