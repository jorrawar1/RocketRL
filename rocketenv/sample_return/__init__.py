"""Autonomous crater sample-return environment and supporting primitives."""

from .config import DEFAULT_SAMPLE_RETURN_CONFIG, SampleReturnConfig
from .controllers import CallableController, ContinuousController
from .env import (
    ABORTED,
    CRASHED_OUTBOUND,
    CRASHED_RETURN,
    OUT_OF_BOUNDS,
    OUT_OF_FUEL,
    SAMPLE_RETURNED,
    TIMEOUT,
    TIPPED_OUTBOUND,
    TIPPED_RETURN,
    SampleReturnEnv,
)
from .evaluation import evaluate_policy, run_episode, summarize_episodes
from .scripted import PayloadAwareScriptedController, scripted_sample_return_action
from .observation import (
    ACTOR_OBSERVATION_DIM,
    ACTOR_OBSERVATION_INDEX,
    ACTOR_OBSERVATION_NAMES,
    OBSERVATION_DIM,
    OBSERVATION_INDEX,
    OBSERVATION_NAMES,
    actor_observation,
)
from .terrain import CraterSampleTerrain, CraterTerrainSpec
from .training import (
    CRITIC_OBSERVATION_DIM,
    CRITIC_OBSERVATION_INDEX,
    CRITIC_OBSERVATION_NAMES,
    SampleReturnTrainingWrapper,
    TrainingTask,
    make_training_env,
    make_vector_env,
)
from .mission_types import MissionPhase, PayloadSpec, SampleReturnState
from .vehicle import VehicleModel

__all__ = [
    "ABORTED",
    "ACTOR_OBSERVATION_DIM",
    "ACTOR_OBSERVATION_INDEX",
    "ACTOR_OBSERVATION_NAMES",
    "CallableController",
    "CRASHED_OUTBOUND",
    "CRASHED_RETURN",
    "CraterSampleTerrain",
    "CraterTerrainSpec",
    "ContinuousController",
    "CRITIC_OBSERVATION_DIM",
    "CRITIC_OBSERVATION_INDEX",
    "CRITIC_OBSERVATION_NAMES",
    "DEFAULT_SAMPLE_RETURN_CONFIG",
    "MissionPhase",
    "OBSERVATION_DIM",
    "OBSERVATION_INDEX",
    "OBSERVATION_NAMES",
    "OUT_OF_BOUNDS",
    "OUT_OF_FUEL",
    "PayloadAwareScriptedController",
    "PayloadSpec",
    "SAMPLE_RETURNED",
    "SampleReturnConfig",
    "SampleReturnEnv",
    "SampleReturnState",
    "SampleReturnTrainingWrapper",
    "TIMEOUT",
    "TIPPED_OUTBOUND",
    "TIPPED_RETURN",
    "TrainingTask",
    "VehicleModel",
    "actor_observation",
    "evaluate_policy",
    "make_training_env",
    "make_vector_env",
    "run_episode",
    "scripted_sample_return_action",
    "summarize_episodes",
]
