"""Resumable, wall-clock-bounded recurrent PPO curriculum for the full mission.

This driver intentionally lives outside ``ppo.py``.  It imports the proven PPO
building blocks, starts from the successful dry opening policy, and writes to a
separate checkpoint family so neither the source experiment nor its artifacts
are overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass
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
    make_training_env,
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


TRAINING_MODE = "overnight_full_mission_v1"
SOURCE_NAME = "ppo_sample_return_opening_severity_v3_working.pt"
SOURCE_SHA256 = "6276bf65595010f93817b2dbfa27be526552c7d1681ab8d0c80e5c1cc6687e10"
NUM_ENVS = 16
ROLLOUT_STEPS = 512
BATCH_SIZE = 512
SEQUENCE_LENGTH = 64
PPO_EPOCHS = 4
GAMMA = 0.9999
GAE_LAMBDA = 0.995
REWARD_SCALE = 0.01
CHECKPOINT_INTERVAL = 10
FULL_EVALUATION_INTERVAL = 50
FINAL_RESERVE_HOURS = 1.5


@dataclass(frozen=True)
class Stage:
    name: str
    updates: int | None
    tasks: tuple[TrainingTask, ...]
    reset_options: dict
    decision_limit: int | None = None
    evaluation_episodes: int = 6


def repeated(task: TrainingTask, count: int) -> tuple[TrainingTask, ...]:
    return (task,) * count


def build_stages() -> tuple[Stage, ...]:
    outbound = TrainingTask.OUTBOUND_LEG
    returning = TrainingTask.RETURN_LEG
    full = TrainingTask.FULL_MISSION
    dry = repeated(outbound, NUM_ENVS)
    balanced = repeated(outbound, 8) + repeated(returning, 8)

    airborne = {
        "spawn_altitude_reference": "crater_rim",
        "spawn_altitude_range": (5.0, 10.0),
        "spawn_theta_range": (-0.08, 0.08),
        "spawn_omega_range": (-0.05, 0.05),
    }
    local = {
        "spawn_pad_edge_distance_range": (0.25, 10.0),
        "spawn_inside_pad_probability": 1.0 / 6.0,
        "spawn_pad_center_probability": 1.0 / 6.0,
        **airborne,
    }
    outer = {
        "spawn_pad_edge_distance_range": (7.0, 10.0),
        "spawn_inside_pad_probability": 1.0 / 8.0,
        "spawn_pad_center_probability": 1.0 / 8.0,
        **airborne,
    }
    broad_payload = {
        "payload_mass_range": (0.0, 0.35),
        "payload_offset_body_x_range": (0.0, 0.80),
        "payload_offset_body_y": 0.0,
    }

    payload_stages = (
        ("massless return context", 40, (0.0, 0.0), (0.0, 0.0)),
        ("light centered payload", 50, (0.0, 0.15), (0.0, 0.0)),
        ("light offset payload", 60, (0.05, 0.22), (0.10, 0.40)),
        ("medium-to-production payload", 70, (0.12, 0.35), (0.25, 0.80)),
        ("broad payload range", 80, (0.0, 0.35), (0.0, 0.80)),
    )

    stages: list[Stage] = [
        Stage("dry outer-navigation consolidation", 60, dry, outer),
    ]
    for name, updates, mass_range, offset_range in payload_stages:
        stages.append(
            Stage(
                name,
                updates,
                balanced,
                {
                    **local,
                    "payload_mass_range": mass_range,
                    "payload_offset_body_x_range": offset_range,
                    "payload_offset_body_y": 0.0,
                },
            )
        )

    for maximum_distance, updates in ((35.0, 50), (60.0, 60), (90.0, 70), (117.0, 90)):
        stages.append(
            Stage(
                f"airborne natural route 15-{maximum_distance:g} m",
                updates,
                balanced,
                {
                    "spawn_distance_from_target_range": (15.0, maximum_distance),
                    **airborne,
                    **broad_payload,
                },
            )
        )

    stages.extend(
        (
            Stage(
                "controlled ground takeoff to rim height",
                50,
                balanced,
                {"takeoff_rim_clearance": 5.0, **broad_payload},
                decision_limit=300,
            ),
            Stage("complete ground legs", 140, balanced, dict(broad_payload)),
            Stage(
                "full-mission rehearsal",
                150,
                repeated(full, 12) + repeated(outbound, 2) + repeated(returning, 2),
                dict(broad_payload),
            ),
            Stage(
                "full missions until wall-clock deadline",
                None,
                repeated(full, NUM_ENVS),
                dict(broad_payload),
                evaluation_episodes=8,
            ),
        )
    )
    return tuple(stages)


def stage_plan_signature(stages: tuple[Stage, ...]) -> str:
    serializable = [
        {
            "name": stage.name,
            "updates": stage.updates,
            "tasks": [task.value for task in stage.tasks],
            "reset_options": stage.reset_options,
            "decision_limit": stage.decision_limit,
        }
        for stage in stages
    ]
    encoded = json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(path: Path, actor, critic, actor_optimizer, critic_optimizer, metadata) -> None:
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


def unique_tasks(tasks: tuple[TrainingTask, ...]) -> tuple[TrainingTask, ...]:
    return tuple(dict.fromkeys(tasks))


def _fixed_payload_corner(options: dict) -> dict:
    fixed = dict(options)
    for scalar_name, range_name in (
        ("payload_mass", "payload_mass_range"),
        ("payload_offset_body_x", "payload_offset_body_x_range"),
        ("payload_offset_body_y", "payload_offset_body_y_range"),
    ):
        value_range = fixed.pop(range_name, None)
        if value_range is not None:
            fixed[scalar_name] = float(value_range[1])
    return fixed


def _stage_probes(stage: Stage) -> tuple[tuple[str, dict], ...]:
    whole = dict(stage.reset_options)
    probes: list[tuple[str, dict]] = [("whole", whole)]
    if "spawn_pad_edge_distance_range" in whole:
        upper_edge = float(whole["spawn_pad_edge_distance_range"][1])
        frontier = _fixed_payload_corner(whole)
        frontier.pop("spawn_pad_edge_distance_range", None)
        frontier.pop("spawn_inside_pad_probability", None)
        frontier.pop("spawn_pad_center_probability", None)
        frontier.update({"spawn_mode": "airborne", "spawn_reference": "target"})
        left = dict(frontier)
        left["spawn_x_offset"] = -(5.0 + upper_edge)
        right = dict(frontier)
        right["spawn_x_offset"] = 5.0 + upper_edge
        probes.extend((("left_frontier", left), ("right_frontier", right)))
    elif "spawn_distance_from_target_range" in whole:
        frontier = _fixed_payload_corner(whole)
        maximum_distance = float(frontier.pop("spawn_distance_from_target_range")[1])
        frontier["spawn_distance_from_target"] = maximum_distance
        probes.append(("route_frontier", frontier))
    return tuple(probes)


def evaluate_stage(actor: Actor, stage: Stage, base_seed: int) -> dict[str, dict[str, float]]:
    results = {}
    result_index = 0
    for probe_name, probe_options in _stage_probes(stage):
        for task in unique_tasks(stage.tasks):
            result = evaluate_actor(
                actor,
                task,
                reset_options=probe_options,
                episodes=stage.evaluation_episodes,
                base_seed=base_seed + result_index * 10_000,
                decision_limit=stage.decision_limit,
            )
            label = task.value if probe_name == "whole" else f"{task.value}:{probe_name}"
            results[label] = result
            result_index += 1
    return results


def format_evaluation(results: dict[str, dict[str, float]]) -> str:
    return " | ".join(
        f"{task}: success={values['success_rate']:.0%}, "
        f"sample={values['sample_acquired_rate']:.0%}, "
        f"length={values['mean_length']:.0f}"
        for task, values in results.items()
    )


def has_minimum_stage_signal(stage: Stage, results: dict[str, dict[str, float]]) -> bool:
    """Prevent a zero-signal stage from being propagated into a harder one."""
    for label, values in results.items():
        success = values["success_rate"]
        if stage.name == "full-mission rehearsal" and label == "full_mission":
            if values["sample_acquired_rate"] < 0.25:
                return False
        elif success < 0.25:
            return False
    return True


def load_networks(checkpoint: dict) -> tuple[Actor, Critic, torch.optim.Optimizer, torch.optim.Optimizer]:
    if int(checkpoint["actor_observation_dim"]) != ACTOR_OBSERVATION_DIM:
        raise ValueError("checkpoint actor observation dimension does not match")
    if int(checkpoint["critic_observation_dim"]) != CRITIC_OBSERVATION_DIM:
        raise ValueError("checkpoint critic observation dimension does not match")
    hidden_width = int(checkpoint["hidden_width"])
    action_dim = int(checkpoint["action_dim"])
    actor = Actor(ACTOR_OBSERVATION_DIM, hidden_width, action_dim)
    critic = Critic(CRITIC_OBSERVATION_DIM, hidden_width, 1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    return actor, critic, actor_optimizer, critic_optimizer


def choose_resume_checkpoint(
    paths: tuple[Path, ...], plan_signature: str, stage_count: int
) -> tuple[Path, dict] | None:
    valid = []
    priority = {path.name: len(paths) - index for index, path in enumerate(paths)}
    for path in paths:
        if not path.exists():
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("training_mode") != TRAINING_MODE:
                continue
        except Exception as exc:
            print(f"ignoring unusable checkpoint {path.name}: {exc}", flush=True)
            continue
        if checkpoint.get("plan_signature") != plan_signature:
            raise ValueError(f"{path.name} belongs to a different overnight plan")
        try:
            required = {
                "actor_state_dict",
                "critic_state_dict",
                "actor_optimizer_state_dict",
                "critic_optimizer_state_dict",
                "stage_index",
                "stage_update",
                "overnight_update",
                "elapsed_seconds",
                "source_sha256",
            }
            missing = required.difference(checkpoint)
            if missing:
                raise ValueError(f"missing keys: {sorted(missing)}")
            stage_index = int(checkpoint["stage_index"])
            stage_update = int(checkpoint["stage_update"])
            overnight_update = int(checkpoint["overnight_update"])
            elapsed = float(checkpoint["elapsed_seconds"])
            if not 0 <= stage_index < stage_count:
                raise ValueError("stage index is out of range")
            if stage_update < 0 or overnight_update < 0:
                raise ValueError("checkpoint update counters cannot be negative")
            if not np.isfinite(elapsed) or elapsed < 0.0:
                raise ValueError("checkpoint elapsed time is invalid")
            if checkpoint["source_sha256"] != SOURCE_SHA256:
                raise ValueError("source checkpoint hash does not match")
            load_networks(checkpoint)
        except Exception as exc:
            print(f"ignoring structurally invalid checkpoint {path.name}: {exc}", flush=True)
            continue
        valid.append(
            (
                int(checkpoint.get("overnight_update", 0)),
                priority[path.name],
                path,
                checkpoint,
            )
        )
    if not valid:
        return None
    _, _, path, checkpoint = max(valid, key=lambda item: (item[0], item[1]))
    return path, checkpoint


def smoke_test(stages: tuple[Stage, ...], source_path: Path, artifacts: Path) -> None:
    print("smoke: validating source and every reset distribution", flush=True)
    if file_sha256(source_path) != SOURCE_SHA256:
        raise ValueError("the dry source checkpoint hash has changed")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    actor, critic, actor_optimizer, critic_optimizer = load_networks(source)

    for stage_index, stage in enumerate(stages):
        for _, probe_options in _stage_probes(stage):
            for task in unique_tasks(stage.tasks):
                env = make_training_env(
                    task=task,
                    action_repeat=4,
                    decision_limit=stage.decision_limit,
                )
                try:
                    observation, info = env.reset(
                        seed=70_000 + stage_index,
                        options=dict(probe_options),
                    )
                    if observation.shape != (ACTOR_OBSERVATION_DIM,):
                        raise RuntimeError("actor observation shape changed")
                    if info["critic_observation"].shape != (CRITIC_OBSERVATION_DIM,):
                        raise RuntimeError("critic observation shape changed")
                    env.step(np.zeros(2, dtype=np.float32))
                finally:
                    env.close()

    stage = stages[0]
    tasks = stage.tasks[:2]
    envs = make_vector_env(
        2,
        task=tasks,
        action_repeat=4,
        asynchronous=True,
    )
    failed = True
    try:
        observations, info = envs.reset(
            seed=[81_000, 81_001],
            options=dict(stage.reset_options),
        )
        critic_observations = np.asarray(info["critic_observation"], dtype=np.float32)
        previous_actions = np.zeros((2, 2), dtype=np.float32)
        hidden_state = actor.initial_hidden_state(2)
        data, *_ = collect_rollout(
            envs,
            observations,
            critic_observations,
            previous_actions,
            hidden_state,
            actor,
            critic,
            64,
            reset_options=stage.reset_options,
        )
        advantages, returns = compute_gae(
            data,
            gamma=GAMMA,
            lam=GAE_LAMBDA,
            reward_scale=REWARD_SCALE,
        )
        stats = ppo_update(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            data,
            advantages,
            returns,
            epochs=1,
            batch_size=128,
            sequence_length=64,
        )
        if not all(np.isfinite(value) for value in stats.values()):
            raise RuntimeError("the PPO smoke update produced a non-finite statistic")
        failed = False
    finally:
        envs.close(terminate=failed)

    smoke_path = artifacts / "ppo_sample_return_overnight_smoke.pt"
    atomic_save(
        smoke_path,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        {"training_mode": TRAINING_MODE, "smoke": True},
    )
    torch.load(smoke_path, map_location="cpu", weights_only=False)
    smoke_path.unlink()
    print("smoke passed: all stages reset, AsyncVectorEnv ran, PPO updated, checkpoint reloaded", flush=True)


def train(hours: float) -> None:
    stages = build_stages()
    plan_signature = stage_plan_signature(stages)
    artifacts = REPO_ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    source_path = artifacts / SOURCE_NAME
    working_path = artifacts / "ppo_sample_return_overnight_working.pt"
    latest_path = artifacts / "ppo_sample_return_overnight_latest.pt"
    previous_path = artifacts / "ppo_sample_return_overnight_previous.pt"
    final_path = artifacts / "ppo_sample_return_overnight_final.pt"
    best_full_path = artifacts / "ppo_sample_return_overnight_best_full.pt"

    resume_paths = (
        working_path,
        final_path,
        latest_path,
        previous_path,
        working_path.with_name(working_path.name + ".tmp"),
        latest_path.with_name(latest_path.name + ".tmp"),
        *tuple(sorted(artifacts.glob("ppo_sample_return_overnight_stage_*.pt"))),
    )
    existing_resume_artifacts = tuple(path for path in resume_paths if path.exists())
    resume = choose_resume_checkpoint(resume_paths, plan_signature, len(stages))
    if resume is None and existing_resume_artifacts:
        names = ", ".join(path.name for path in existing_resume_artifacts)
        raise RuntimeError(
            "overnight checkpoint artifacts exist but none are usable; "
            f"refusing to overwrite them or restart from the source ({names})"
        )
    if resume is None:
        if not source_path.exists():
            raise FileNotFoundError(f"missing source checkpoint: {source_path}")
        source_hash = file_sha256(source_path)
        if source_hash != SOURCE_SHA256:
            raise ValueError("the dry source checkpoint hash has changed")
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        if checkpoint.get("training_mode") != "fixed_opening_severity_v3":
            raise ValueError("the dry source checkpoint mode does not match")
        stage_index = 0
        stage_update = 0
        stage_goal_update = stages[0].updates
        overnight_update = 0
        prior_elapsed = 0.0
        best_full_key = (-1.0, -1.0)
        startup = f"bootstrapped from immutable {source_path.name}"
    else:
        resume_path, checkpoint = resume
        stage_index = int(checkpoint["stage_index"])
        stage_update = int(checkpoint["stage_update"])
        saved_goal = checkpoint.get("stage_goal_update", stages[stage_index].updates)
        stage_goal_update = None if saved_goal is None else int(saved_goal)
        overnight_update = int(checkpoint["overnight_update"])
        prior_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        best_full_key = tuple(checkpoint.get("best_full_key", (-1.0, -1.0)))
        startup = f"resumed {resume_path.name}"

    source_global_update = int(
        checkpoint.get(
            "source_global_update",
            int(checkpoint.get("global_update", 60)) - overnight_update,
        )
    )

    actor, critic, actor_optimizer, critic_optimizer = load_networks(checkpoint)
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])

    session_started = time.monotonic()
    budget_seconds = hours * 3600.0
    final_stage_index = len(stages) - 1

    def elapsed_seconds() -> float:
        return prior_elapsed + (time.monotonic() - session_started)

    def metadata(last_evaluation=None, stopped_reason=None):
        return {
            "training_mode": TRAINING_MODE,
            "plan_signature": plan_signature,
            "source_checkpoint": SOURCE_NAME,
            "source_sha256": SOURCE_SHA256,
            "stage_index": stage_index,
            "stage_update": stage_update,
            "stage_goal_update": stage_goal_update,
            "stage_name": stages[min(stage_index, final_stage_index)].name,
            "overnight_update": overnight_update,
            "source_global_update": source_global_update,
            "global_update": source_global_update + overnight_update,
            "elapsed_seconds": elapsed_seconds(),
            "wall_clock_budget_hours": hours,
            "best_full_key": best_full_key,
            "last_evaluation": last_evaluation,
            "stopped_reason": stopped_reason,
            "num_envs": NUM_ENVS,
            "rollout_steps": ROLLOUT_STEPS,
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "ppo_epochs": PPO_EPOCHS,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "reward_scale": REWARD_SCALE,
            "actor_lr": actor_optimizer.param_groups[0]["lr"],
            "critic_lr": critic_optimizer.param_groups[0]["lr"],
            "torch_rng_state": torch.get_rng_state(),
        }

    def save_working(last_evaluation=None, stopped_reason=None):
        atomic_save(
            working_path,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            metadata(last_evaluation, stopped_reason),
        )

    def promote_latest(stage_name: str, last_evaluation):
        stage_path = artifacts / (
            f"ppo_sample_return_overnight_stage_{stage_index:02d}_"
            f"{stage_name.replace(' ', '_').replace('-', '_')}.pt"
        )
        atomic_save(
            stage_path,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            metadata(last_evaluation),
        )
        if latest_path.exists():
            shutil.copy2(latest_path, previous_path)
        atomic_save(
            latest_path,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            metadata(last_evaluation),
        )
        working_path.unlink(missing_ok=True)

    print(
        f"{startup} | budget={hours:g}h | source global update="
        f"{checkpoint.get('global_update', 60)} | stages={len(stages)}",
        flush=True,
    )

    stopped_reason = "wall-clock deadline"
    normal_shutdown = False
    try:
        while stage_index < len(stages) and elapsed_seconds() < budget_seconds:
            if (
                stage_index < final_stage_index
                and elapsed_seconds() >= budget_seconds - FINAL_RESERVE_HOURS * 3600.0
            ):
                print(
                    "reserving the final 90 minutes: jumping to all-full-mission training",
                    flush=True,
                )
                stage_index = final_stage_index
                stage_update = 0
                stage_goal_update = None
                save_working(stopped_reason="forced final-stage handoff")

            stage = stages[stage_index]
            print(
                f"\nstage {stage_index + 1}/{len(stages)}: {stage.name} | "
                f"resume update={stage_update} | elapsed={elapsed_seconds() / 3600:.2f}h",
                flush=True,
            )
            envs = make_vector_env(
                NUM_ENVS,
                task=stage.tasks,
                action_repeat=4,
                decision_limit=stage.decision_limit,
                asynchronous=True,
            )
            failed = True
            try:
                seed_base = 200_000 + stage_index * 20_000 + stage_update
                observations, reset_info = envs.reset(
                    seed=[seed_base + index for index in range(NUM_ENVS)],
                    options=dict(stage.reset_options),
                )
                critic_observations = np.asarray(
                    reset_info["critic_observation"], dtype=np.float32
                )
                previous_actions = np.zeros((NUM_ENVS, 2), dtype=np.float32)
                hidden_state = actor.initial_hidden_state(NUM_ENVS)
                running_returns = np.zeros(NUM_ENVS, dtype=np.float64)
                recent_returns: deque[float] = deque(maxlen=100)
                recent_successes = {
                    task: deque(maxlen=100) for task in unique_tasks(stage.tasks)
                }

                while (
                    (stage_goal_update is None or stage_update < stage_goal_update)
                    and elapsed_seconds() < budget_seconds
                    and not (
                        stage_index < final_stage_index
                        and elapsed_seconds()
                        >= budget_seconds - FINAL_RESERVE_HOURS * 3600.0
                    )
                ):
                    update_started = time.perf_counter()
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
                        ROLLOUT_STEPS,
                        reset_options=stage.reset_options,
                    )

                    for t in range(ROLLOUT_STEPS):
                        running_returns += rollout_data["rewards"][t]
                        done = np.logical_or(
                            rollout_data["terminated"][t],
                            rollout_data["truncated"][t],
                        )
                        for env_index in np.flatnonzero(done):
                            recent_returns.append(float(running_returns[env_index]))
                            recent_successes[stage.tasks[env_index]].append(
                                bool(rollout_data["task_success"][t, env_index])
                            )
                        running_returns[done] = 0.0

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
                        epochs=PPO_EPOCHS,
                        batch_size=BATCH_SIZE,
                        sequence_length=SEQUENCE_LENGTH,
                    )
                    overnight_update += 1
                    stage_update += 1
                    seconds = time.perf_counter() - update_started
                    mean_return = (
                        float(np.mean(recent_returns)) if recent_returns else float("nan")
                    )
                    success_text = " ".join(
                        f"{task.value}={np.mean(values):.0%}"
                        if values
                        else f"{task.value}=n/a"
                        for task, values in recent_successes.items()
                    )
                    print(
                        f"update +{overnight_update:04d} | stage={stage_update:03d} "
                        f"| return={mean_return:8.2f} | {success_text} | {seconds:5.1f}s "
                        f"| remaining={(budget_seconds - elapsed_seconds()) / 3600:.2f}h",
                        flush=True,
                    )
                    print(
                        f"  kl={stats['approx_kl']:.5f} clip={stats['clip_fraction']:.1%} "
                        f"std=({stats['throttle_std']:.3f},{stats['gimbal_std']:.3f}) "
                        f"EV={stats['explained_variance']:.3f} "
                        f"gru_grad={stats['gru_grad_norm']:.4f} "
                        f"epochs={int(stats['actor_epochs'])}",
                        flush=True,
                    )

                    if overnight_update % CHECKPOINT_INTERVAL == 0:
                        save_working()

                    if (
                        stage.updates is None
                        and stage_update % FULL_EVALUATION_INTERVAL == 0
                        and elapsed_seconds() < budget_seconds - 60.0
                    ):
                        results = evaluate_stage(
                            actor, stage, base_seed=9_000_000
                        )
                        print(f"  deterministic: {format_evaluation(results)}", flush=True)
                        full_result = results[TrainingTask.FULL_MISSION.value]
                        full_key = (
                            full_result["success_rate"],
                            full_result["sample_acquired_rate"],
                        )
                        if full_key > best_full_key:
                            best_full_key = full_key
                            atomic_save(
                                best_full_path,
                                actor,
                                critic,
                                actor_optimizer,
                                critic_optimizer,
                                metadata(results),
                            )
                            print(f"  new best full-mission checkpoint: {best_full_path.name}", flush=True)
                        save_working(results)
                failed = False
            finally:
                envs.close(terminate=failed)

            if elapsed_seconds() >= budget_seconds:
                break

            if stage_goal_update is not None and stage_update >= stage_goal_update:
                results = evaluate_stage(
                    actor, stage, base_seed=8_000_000 + stage_index * 100_000
                )
                print(f"stage evaluation: {format_evaluation(results)}", flush=True)
                if not has_minimum_stage_signal(stage, results):
                    stage_goal_update += 20
                    print(
                        "stage held because at least one deterministic probe has "
                        f"less than 25% signal; extending to update {stage_goal_update}",
                        flush=True,
                    )
                    save_working(results, stopped_reason="stage extension")
                    continue
                completed_name = stage.name
                stage_index += 1
                stage_update = 0
                stage_goal_update = stages[stage_index].updates
                promote_latest(completed_name, results)

        if stage_index >= len(stages):
            stopped_reason = "curriculum complete"
        normal_shutdown = True
    except KeyboardInterrupt:
        stopped_reason = "keyboard interrupt"
        print(
            "interrupted; preserving the last completed periodic checkpoint",
            flush=True,
        )
        raise
    except BaseException:
        stopped_reason = "error; see traceback"
        raise
    finally:
        if normal_shutdown:
            atomic_save(
                final_path,
                actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                metadata(stopped_reason=stopped_reason),
            )
            save_working(stopped_reason=stopped_reason)
            print(
                f"overnight run stopped: {stopped_reason} | "
                f"elapsed={elapsed_seconds() / 3600:.2f}h",
                flush=True,
            )
            print(f"final checkpoint: {final_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.hours) or args.hours <= 0.0:
        parser.error("--hours must be a positive finite number")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    stages = build_stages()
    artifacts = REPO_ROOT / "artifacts"
    if args.smoke:
        smoke_test(stages, artifacts / SOURCE_NAME, artifacts)
    else:
        train(args.hours)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print("training stopped by user; the last periodic checkpoint is intact", flush=True)
