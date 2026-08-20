"""Export the recurrent PPO actor and a browser parity reference.

The critic remains training-only, while the browser receives both parameters
of the learned action distribution:

    (observation, previous bounded action, GRU hidden state)
        -> (latent mean, standard deviation, next hidden, activations)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocketenv.sample_return import (
    ACTOR_OBSERVATION_DIM,
    ACTOR_OBSERVATION_NAMES,
    TrainingTask,
    make_training_env,
)
from rocketenv.sample_return.ppo import Actor


class WebActor(nn.Module):
    """ONNX-friendly view of the trained actor with diagnostic activations."""

    def __init__(self, actor: Actor) -> None:
        super().__init__()
        self.actor = actor

    def forward(
        self,
        observation: torch.Tensor,
        previous_action: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        layer1 = torch.tanh(self.actor.layer1(observation))
        previous_features = torch.tanh(
            self.actor.previous_action_layer(previous_action)
        )
        layer2 = torch.tanh(self.actor.layer2(layer1 + previous_features))
        next_hidden = self.actor.gru(layer2, hidden)
        mean = self.actor.mean_head(next_hidden)
        log_std = self.actor.log_std_head(next_hidden)
        # Export the raw learned head, not the exploration floor used by the
        # training behavior distribution. The demo deliberately avoids
        # injecting floor-driven noise.
        std = torch.clamp(log_std, max=1.0).exp()
        return mean, std, next_hidden, layer1, layer2


def _load_actor(checkpoint_path: Path) -> tuple[Actor, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    observation_dim = int(checkpoint["actor_observation_dim"])
    hidden_width = int(checkpoint["hidden_width"])
    action_dim = int(checkpoint["action_dim"])
    if observation_dim != ACTOR_OBSERVATION_DIM:
        raise ValueError(
            f"checkpoint actor dim {observation_dim} != {ACTOR_OBSERVATION_DIM}"
        )
    actor = Actor(observation_dim, hidden_width, action_dim)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor, checkpoint


def _deterministic_action(mean: torch.Tensor) -> torch.Tensor:
    squashed = torch.tanh(mean)
    return torch.stack(((squashed[..., 0] + 1.0) / 2.0, squashed[..., 1]), dim=-1)


@torch.inference_mode()
def _parity_steps(actor: Actor, count: int) -> dict:
    env = make_training_env(task=TrainingTask.FULL_MISSION, action_repeat=4)
    try:
        observation, _ = env.reset(
            seed=8_410_037,
            options={
                "payload_mass": 0.35,
                "payload_offset_body_x": 0.80,
                "payload_offset_body_y": 0.0,
            },
        )
        previous_action = torch.zeros((1, 2), dtype=torch.float32)
        hidden = actor.initial_hidden_state(1)
        records: list[dict[str, list[float]]] = []
        observation_cases: list[dict] = []

        for _ in range(count):
            observation_tensor = torch.as_tensor(observation).reshape(1, -1)
            layer1 = torch.tanh(actor.layer1(observation_tensor))
            previous_features = torch.tanh(
                actor.previous_action_layer(previous_action)
            )
            layer2 = torch.tanh(actor.layer2(layer1 + previous_features))
            next_hidden = actor.gru(layer2, hidden)
            mean = actor.mean_head(next_hidden)
            log_std = actor.log_std_head(next_hidden)
            std = torch.clamp(log_std, max=1.0).exp()
            action = _deterministic_action(mean)

            records.append(
                {
                    "observation": observation_tensor[0].tolist(),
                    "previous_action": previous_action[0].tolist(),
                    "hidden": hidden[0].tolist(),
                    "mean": mean[0].tolist(),
                    "std": std[0].tolist(),
                    "action": action[0].tolist(),
                    "next_hidden": next_hidden[0].tolist(),
                    "layer1": layer1[0].tolist(),
                    "layer2": layer2[0].tolist(),
                }
            )
            if len(observation_cases) < 8:
                physical = env.mission_env
                observation_cases.append(
                    {
                        "state": physical.state.tolist(),
                        "target_x": float(physical.target_x),
                        "phase": physical.phase.name,
                        "payload_attached": bool(
                            physical.mission_state.payload_attached
                        ),
                        "expected": observation_tensor[0].tolist(),
                    }
                )

            observation, _, terminated, truncated, info = env.step(
                action[0].numpy()
            )
            previous_action = action
            hidden = next_hidden
            if info["auto_advanced_sampling_steps"] > 0:
                previous_action.zero_()
                hidden.zero_()
            if terminated or truncated:
                break
        if len(records) != count:
            raise RuntimeError(
                f"parity rollout ended after {len(records)} of {count} decisions"
            )
        return {
            "steps": records,
            "observation_contract": {
                "terrain": env.mission_env.terrain.vertices.tolist(),
                "cases": observation_cases,
            },
        }
    finally:
        env.close()


def export_actor(
    checkpoint_path: Path,
    model_path: Path,
    metadata_path: Path,
    reference_path: Path,
    *,
    reference_steps: int,
) -> None:
    actor, checkpoint = _load_actor(checkpoint_path)
    wrapper = WebActor(actor).eval()
    hidden_width = actor.hidden_width
    action_dim = int(checkpoint["action_dim"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (
            torch.zeros((1, ACTOR_OBSERVATION_DIM), dtype=torch.float32),
            torch.zeros((1, action_dim), dtype=torch.float32),
            torch.zeros((1, hidden_width), dtype=torch.float32),
        ),
        model_path,
        input_names=["observation", "previous_action", "hidden"],
        output_names=["mean", "std", "next_hidden", "layer1", "layer2"],
        dynamic_axes={
            "observation": {0: "batch"},
            "previous_action": {0: "batch"},
            "hidden": {0: "batch"},
            "mean": {0: "batch"},
            "std": {0: "batch"},
            "next_hidden": {0: "batch"},
            "layer1": {0: "batch"},
            "layer2": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "checkpoint": checkpoint_path.name,
        "checkpoint_training_mode": checkpoint.get("training_mode"),
        "checkpoint_update": checkpoint.get(
            "finetune_update", checkpoint.get("overnight_update")
        ),
        "model_sha256": model_sha256,
        "actor_observation_dim": ACTOR_OBSERVATION_DIM,
        "observation_names": list(ACTOR_OBSERVATION_NAMES),
        "hidden_width": hidden_width,
        "action_dim": action_dim,
        "action_repeat": 4,
        "action_sampling": "latent=mean+std*N(0,1); throttle=(tanh(latent[0])+1)/2; gimbal=tanh(latent[1])",
        "std_source": "raw exp(clamp(log_std,max=1)) before the training exploration floor",
        "memory_reset": "episode boundary and auto-advanced sampling handoff",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    parity = _parity_steps(actor, reference_steps)
    reference = {
        "schema_version": 1,
        "model_sha256": model_sha256,
        "absolute_tolerance": 2e-5,
        "steps": parity["steps"],
        "observation_contract": parity["observation_contract"],
    }
    reference_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")

    print(f"wrote {model_path} ({model_path.stat().st_size / 1024:.1f} KiB)")
    print(f"wrote {metadata_path}")
    print(f"wrote {reference_path} ({reference_steps} recurrent decisions)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/ppo_sample_return_overnight_final.pt"),
    )
    parser.add_argument(
        "--model", type=Path, default=Path("web/public/models/final_actor.onnx")
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("web/public/models/final_actor.json"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("web/test/data/policy_reference.json"),
    )
    parser.add_argument("--reference-steps", type=int, default=24)
    args = parser.parse_args()
    if args.reference_steps < 1:
        parser.error("--reference-steps must be positive")
    export_actor(
        args.checkpoint,
        args.model,
        args.metadata,
        args.reference,
        reference_steps=args.reference_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
