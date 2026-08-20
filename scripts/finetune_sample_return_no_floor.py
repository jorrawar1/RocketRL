"""Short isolated PPO continuation with the exploration floor removed.

This script starts from the completed overnight full-mission checkpoint,
preserves its networks and Adam state, zeros the actor's std-floor buffer, and
continues only on broad-payload full missions. It never overwrites the source.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rocketenv.sample_return import (  # noqa: E402
    ACTOR_OBSERVATION_DIM,
    CRITIC_OBSERVATION_DIM,
    TrainingTask,
    make_vector_env,
)
from rocketenv.sample_return.ppo import (  # noqa: E402
    Actor,
    Critic,
    collect_rollout,
    compute_gae,
    evaluate_actor,
    ppo_update,
    save_checkpoint,
)


TRAINING_MODE = "no_floor_full_mission_finetune_v1"
SOURCE_NAME = "ppo_sample_return_overnight_final.pt"
WORKING_NAME = "ppo_sample_return_no_floor_finetune_working.pt"
FINAL_NAME = "ppo_sample_return_no_floor_finetune.pt"

NUM_ENVS = 16
ROLLOUT_STEPS = 512
BATCH_SIZE = 512
SEQUENCE_LENGTH = 64
PPO_EPOCHS = 4
GAMMA = 0.9999
GAE_LAMBDA = 0.995
REWARD_SCALE = 0.01
ACTOR_LR = 1e-5
CRITIC_LR = 3e-4

RESET_OPTIONS = {
    "payload_mass_range": (0.0, 0.35),
    "payload_offset_body_x_range": (0.0, 0.80),
    "payload_offset_body_y": 0.0,
}


def load_networks(checkpoint: dict):
    if int(checkpoint["actor_observation_dim"]) != ACTOR_OBSERVATION_DIM:
        raise ValueError("actor observation dimension does not match")
    if int(checkpoint["critic_observation_dim"]) != CRITIC_OBSERVATION_DIM:
        raise ValueError("critic observation dimension does not match")

    hidden_width = int(checkpoint["hidden_width"])
    action_dim = int(checkpoint["action_dim"])
    actor = Actor(ACTOR_OBSERVATION_DIM, hidden_width, action_dim)
    critic = Critic(CRITIC_OBSERVATION_DIM, hidden_width, 1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    for group in actor_optimizer.param_groups:
        group["lr"] = ACTOR_LR

    # This is the only intended behavioral change at startup. With a zero
    # floor, exp(log_std) receives gradients everywhere again.
    actor.std_floor.zero_()
    return actor, critic, actor_optimizer, critic_optimizer


def atomic_save(path, actor, critic, actor_optimizer, critic_optimizer, metadata):
    temporary = path.with_name(path.name + ".tmp")
    save_checkpoint(
        temporary,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        metadata,
    )
    os.replace(temporary, path)


def metadata(actor, update, source_global_update):
    return {
        "training_mode": TRAINING_MODE,
        "source_checkpoint": SOURCE_NAME,
        "source_global_update": source_global_update,
        "finetune_update": update,
        "global_update": source_global_update + update,
        "num_envs": NUM_ENVS,
        "rollout_steps": ROLLOUT_STEPS,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "ppo_epochs": PPO_EPOCHS,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "reward_scale": REWARD_SCALE,
        "actor_lr": ACTOR_LR,
        "critic_lr": CRITIC_LR,
        "reset_options": RESET_OPTIONS,
        "std_floor": actor.std_floor.detach().cpu().tolist(),
        "torch_rng_state": torch.get_rng_state(),
    }


def run(target_updates: int, *, smoke: bool = False) -> Path | None:
    artifacts = REPO_ROOT / "artifacts"
    source_path = artifacts / SOURCE_NAME
    working_path = artifacts / WORKING_NAME
    final_path = artifacts / FINAL_NAME

    if smoke:
        checkpoint_path = source_path
    elif working_path.exists():
        checkpoint_path = working_path
    elif final_path.exists():
        checkpoint_path = final_path
    else:
        checkpoint_path = source_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint_path != source_path and checkpoint.get("training_mode") != TRAINING_MODE:
        raise ValueError(f"{checkpoint_path.name} is not a no-floor fine-tune")

    actor, critic, actor_optimizer, critic_optimizer = load_networks(checkpoint)
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    update = 0 if checkpoint_path == source_path else int(checkpoint["finetune_update"])
    source_global_update = (
        int(checkpoint.get("global_update", 0))
        if checkpoint_path == source_path
        else int(checkpoint["source_global_update"])
    )

    env_count = 2 if smoke else NUM_ENVS
    rollout_steps = 64 if smoke else ROLLOUT_STEPS
    batch_size = 128 if smoke else BATCH_SIZE
    epochs = 1 if smoke else PPO_EPOCHS
    tasks = (TrainingTask.FULL_MISSION,) * env_count
    envs = make_vector_env(
        env_count,
        task=tasks,
        action_repeat=4,
        asynchronous=True,
    )
    failed = True
    recent_successes: deque[bool] = deque(maxlen=100)
    try:
        seed_base = 7_300_000 + update * env_count
        observations, reset_info = envs.reset(
            seed=[seed_base + index for index in range(env_count)],
            options=dict(RESET_OPTIONS),
        )
        critic_observations = np.asarray(
            reset_info["critic_observation"], dtype=np.float32
        )
        previous_actions = np.zeros((env_count, 2), dtype=np.float32)
        hidden_state = actor.initial_hidden_state(env_count)

        stop_update = 1 if smoke else target_updates
        while update < stop_update:
            (
                rollout_data,
                observations,
                critic_observations,
                previous_actions,
                hidden_state,
            ) = collect_rollout(
                envs,
                observations,
                critic_observations,
                previous_actions,
                hidden_state,
                actor,
                critic,
                rollout_steps,
                reset_options=RESET_OPTIONS,
            )
            done = np.logical_or(
                rollout_data["terminated"], rollout_data["truncated"]
            )
            for t, env_index in np.argwhere(done):
                recent_successes.append(
                    bool(rollout_data["task_success"][t, env_index])
                )

            advantages, returns = compute_gae(
                rollout_data,
                gamma=GAMMA,
                lam=GAE_LAMBDA,
                reward_scale=REWARD_SCALE,
            )
            stats = ppo_update(
                actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                rollout_data,
                advantages,
                returns,
                epochs=epochs,
                batch_size=batch_size,
                sequence_length=SEQUENCE_LENGTH,
            )
            if not all(np.isfinite(value) for value in stats.values()):
                raise RuntimeError("PPO produced a non-finite statistic")
            update += 1
            success = (
                f"{np.mean(recent_successes):.0%}" if recent_successes else "n/a"
            )
            print(
                f"update {update:03d}/{stop_update:03d} | success={success} | "
                f"std=({stats['throttle_std']:.4f},{stats['gimbal_std']:.4f}) | "
                f"kl={stats['approx_kl']:.4f} clip={stats['clip_fraction']:.1%} "
                f"epochs={stats['actor_epochs']:.0f}",
                flush=True,
            )
            if not smoke:
                atomic_save(
                    working_path,
                    actor,
                    critic,
                    actor_optimizer,
                    critic_optimizer,
                    metadata(actor, update, source_global_update),
                )
        failed = False
    finally:
        envs.close(terminate=failed)

    if smoke:
        print("no-floor smoke passed", flush=True)
        return None

    evaluation = evaluate_actor(
        actor,
        TrainingTask.FULL_MISSION,
        reset_options=RESET_OPTIONS,
        episodes=12,
        base_seed=8_300_000,
    )
    final_metadata = metadata(actor, update, source_global_update)
    final_metadata["deterministic_evaluation"] = evaluation
    atomic_save(
        final_path,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        final_metadata,
    )
    print(f"evaluation: {evaluation}", flush=True)
    print(f"saved {final_path}", flush=True)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    torch.manual_seed(73)
    run(args.updates, smoke=args.smoke)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
