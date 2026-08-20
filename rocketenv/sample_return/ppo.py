"""Small PyTorch PPO building blocks for the sample-return environment."""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rocketenv.sample_return import (
    ACTOR_OBSERVATION_DIM,
    CRITIC_OBSERVATION_DIM,
    DEFAULT_SAMPLE_RETURN_CONFIG,
    TrainingTask,
    make_training_env,
    make_vector_env,
)


class Actor(nn.Module):
    """Recurrent state-dependent diagonal Gaussian policy."""

    def __init__(self, observation_dim: int, hidden_width: int, output_size: int) -> None:
        super().__init__()
        self.hidden_width = hidden_width
        self.layer1 = nn.Linear(observation_dim, hidden_width)
        self.previous_action_layer = nn.Linear(output_size, hidden_width)
        self.layer2 = nn.Linear(hidden_width, hidden_width)
        self.gru = nn.GRUCell(hidden_width, hidden_width)
        self.mean_head = nn.Linear(hidden_width, output_size)
        self.log_std_head = nn.Linear(hidden_width, output_size)

        for layer in (self.layer1, self.previous_action_layer, self.layer2):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)

        for weight in (self.gru.weight_ih, self.gru.weight_hh):
            for gate in weight.chunk(3, dim=0):
                nn.init.orthogonal_(gate)
        nn.init.zeros_(self.gru.bias_ih)
        nn.init.zeros_(self.gru.bias_hh)

        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -0.5)

        std_floor = torch.full((output_size,), 0.15)
        if output_size > 1:
            std_floor[1] = 0.30
        self.register_buffer("std_floor", std_floor)

    def initial_hidden_state(self, batch_size: int) -> torch.Tensor:
        """Create one zero GRU state for each parallel environment."""
        return self.std_floor.new_zeros((batch_size, self.hidden_width))

    def forward(
        self,
        observations: torch.Tensor,
        previous_actions: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.tanh(self.layer1(observations))
        previous_action_features = torch.tanh(self.previous_action_layer(previous_actions))
        x = torch.tanh(self.layer2(x + previous_action_features))
        next_hidden_state = self.gru(x, hidden_state)
        mean = self.mean_head(next_hidden_state)
        log_std = self.log_std_head(next_hidden_state)
        return mean, log_std, next_hidden_state

    def distribution_parameters(
        self,
        observations: torch.Tensor,
        previous_actions: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, next_hidden_state = self(observations, previous_actions, hidden_state)
        std = torch.maximum(torch.clamp(log_std, max=1.0).exp(), self.std_floor)
        return mean, std, next_hidden_state


class Critic(nn.Module):
    """Feed-forward value network over the full simulator observation."""

    def __init__(self, observation_dim: int, hidden_width: int, output_size: int = 1) -> None:
        super().__init__()
        self.layer1 = nn.Linear(observation_dim, hidden_width)
        self.layer2 = nn.Linear(hidden_width, hidden_width)
        self.layer3 = nn.Linear(hidden_width, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return self.layer3(x)


def squashed_log_prob(normal: torch.distributions.Normal, raw_action: torch.Tensor) -> torch.Tensor:
    """Log probability after tanh and the throttle's affine transform."""

    squashed = torch.tanh(raw_action)
    component_log_prob = normal.log_prob(raw_action)
    tanh_correction = torch.log(1.0 - squashed.pow(2) + 1e-6)

    return (component_log_prob - tanh_correction).sum(dim=-1) + math.log(2.0)


def choose_action(
    actor_network: Actor,
    observation_tensor: torch.Tensor,
    previous_action_tensor: torch.Tensor,
    hidden_state: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Sample a bounded throttle/gimbal action from the actor."""

    mean, std, next_hidden_state = actor_network.distribution_parameters(
        observation_tensor, previous_action_tensor, hidden_state
    )
    normal = torch.distributions.Normal(mean, std)
    sampled_unbounded = normal.rsample()
    squashed = torch.tanh(sampled_unbounded)

    throttle = (squashed[..., 0] + 1.0) / 2.0
    gimbal = squashed[..., 1]
    sampled_action = torch.stack((throttle, gimbal), dim=-1)

    log_unbounded = normal.log_prob(sampled_unbounded).sum(dim=-1)
    log_bounded = squashed_log_prob(normal, sampled_unbounded)

    return (
        mean,
        std,
        sampled_unbounded,
        sampled_action,
        log_unbounded,
        log_bounded,
        next_hidden_state,
    )


@torch.no_grad()
def collect_rollout(
    environments,
    observations: np.ndarray,
    critic_observations: np.ndarray,
    previous_actions: np.ndarray,
    hidden_state: torch.Tensor,
    actor_network: Actor,
    critic_network: Critic,
    rollout_steps: int,
    reset_options: dict | None = None,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    torch.Tensor,
]:
    """Collect a fixed number of decisions from a disabled-autoreset vector env."""

    observations = np.asarray(observations, dtype=np.float32)
    critic_observations = np.asarray(critic_observations, dtype=np.float32)
    previous_actions = np.asarray(previous_actions, dtype=np.float32)
    num_envs = observations.shape[0]
    observation_dim = observations.shape[-1]
    critic_observation_dim = critic_observations.shape[-1]
    action_dim = environments.single_action_space.shape[0]
    device = next(actor_network.parameters()).device
    hidden_state = hidden_state.to(device)

    observations_matrix = np.empty((rollout_steps, num_envs, observation_dim), dtype=np.float32)
    next_observations_matrix = np.empty_like(observations_matrix)
    critic_observations_matrix = np.empty(
        (rollout_steps, num_envs, critic_observation_dim), dtype=np.float32
    )
    next_critic_observations_matrix = np.empty_like(critic_observations_matrix)
    previous_actions_matrix = np.empty((rollout_steps, num_envs, action_dim), dtype=np.float32)
    hidden_states_matrix = np.empty(
        (rollout_steps, num_envs, hidden_state.shape[-1]), dtype=np.float32
    )
    means_matrix = np.empty((rollout_steps, num_envs, action_dim), dtype=np.float32)
    stds_matrix = np.empty_like(means_matrix)
    unclipped_actions_matrix = np.empty_like(means_matrix)
    sampled_actions_matrix = np.empty_like(means_matrix)

    rewards_matrix = np.empty((rollout_steps, num_envs), dtype=np.float32)
    critic_estimates_matrix = np.empty_like(rewards_matrix)
    bootstrap_values_matrix = np.empty_like(rewards_matrix)
    terminated_matrix = np.empty((rollout_steps, num_envs), dtype=np.bool_)
    truncated_matrix = np.empty_like(terminated_matrix)
    memory_reset_matrix = np.empty_like(terminated_matrix)
    task_success_matrix = np.zeros_like(terminated_matrix)
    decision_timeout_matrix = np.zeros_like(terminated_matrix)
    unbounded_logs_matrix = np.empty_like(rewards_matrix)
    bounded_logs_matrix = np.empty_like(rewards_matrix)

    actor_network.eval()
    critic_network.eval()

    for t in range(rollout_steps):
        observations_matrix[t] = observations
        critic_observations_matrix[t] = critic_observations
        previous_actions_matrix[t] = previous_actions
        hidden_states_matrix[t] = hidden_state.cpu().numpy()

        observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        critic_observation_tensor = torch.as_tensor(
            critic_observations, dtype=torch.float32, device=device
        )
        previous_action_tensor = torch.as_tensor(previous_actions, dtype=torch.float32, device=device)

        (
            mean,
            std,
            sampled_unbounded,
            action_tensor,
            log_unbounded,
            log_bounded,
            next_hidden_state,
        ) = choose_action(actor_network, observation_tensor, previous_action_tensor, hidden_state)

        value_tensor = critic_network(critic_observation_tensor).squeeze(-1)

        means_matrix[t] = mean.cpu().numpy()
        stds_matrix[t] = std.cpu().numpy()
        unclipped_actions_matrix[t] = sampled_unbounded.cpu().numpy()
        unbounded_logs_matrix[t] = log_unbounded.cpu().numpy()
        bounded_logs_matrix[t] = log_bounded.cpu().numpy()
        critic_estimates_matrix[t] = value_tensor.cpu().numpy()

        actions = action_tensor.cpu().numpy().astype(np.float32, copy=False)
        sampled_actions_matrix[t] = actions

        (
            next_observations,
            rewards,
            terminated,
            truncated,
            step_info,
        ) = environments.step(actions)

        next_critic_observations = np.asarray(step_info["critic_observation"], dtype=np.float32)
        next_observations_matrix[t] = next_observations
        next_critic_observations_matrix[t] = next_critic_observations
        rewards_matrix[t] = rewards
        terminated_matrix[t] = terminated
        truncated_matrix[t] = truncated
        task_success = np.asarray(step_info["task_success"], dtype=np.bool_)
        task_success_valid = np.asarray(step_info["_task_success"], dtype=np.bool_)
        task_success_matrix[t] = task_success & task_success_valid
        decision_timeout_matrix[t] = np.asarray(
            step_info["decision_timeout"], dtype=np.bool_
        )

        next_critic_observation_tensor = torch.as_tensor(
            next_critic_observations, dtype=torch.float32, device=device
        )
        next_values = (
            critic_network(next_critic_observation_tensor)
            .squeeze(-1)
            .cpu()
            .numpy()
        )

        bootstrap_values = next_values.copy()
        bootstrap_values[terminated] = 0.0
        bootstrap_values_matrix[t] = bootstrap_values

        done = np.logical_or(terminated, truncated)
        observations = np.asarray(next_observations, dtype=np.float32).copy()
        critic_observations = next_critic_observations.copy()
        previous_actions = actions.copy()
        hidden_state = next_hidden_state

        sampling_advanced = np.asarray(step_info["auto_advanced_sampling_steps"]) > 0
        memory_reset = np.logical_or(done, sampling_advanced)
        memory_reset_matrix[t] = memory_reset
        previous_actions[memory_reset] = 0.0

        if done.any():
            options = dict(reset_options or {})
            options["reset_mask"] = done
            reset_observations, reset_info = environments.reset(options=options)
            observations[done] = reset_observations[done]
            reset_critic_observations = np.asarray(reset_info["critic_observation"], dtype=np.float32)
            critic_observations[done] = reset_critic_observations[done]

        if memory_reset.any():
            hidden_state = hidden_state.clone()
            memory_reset_tensor = torch.as_tensor(memory_reset, device=device)
            hidden_state[memory_reset_tensor] = 0.0

    rollout_data = {
        "observations": observations_matrix,
        "next_observations": next_observations_matrix,
        "critic_observations": critic_observations_matrix,
        "next_critic_observations": next_critic_observations_matrix,
        "previous_actions": previous_actions_matrix,
        "hidden_states": hidden_states_matrix,
        "means": means_matrix,
        "stds": stds_matrix,
        "unclipped_actions": unclipped_actions_matrix,
        "actions": sampled_actions_matrix,
        "rewards": rewards_matrix,
        "values": critic_estimates_matrix,
        "bootstrap_values": bootstrap_values_matrix,
        "terminated": terminated_matrix,
        "truncated": truncated_matrix,
        "memory_reset": memory_reset_matrix,
        "task_success": task_success_matrix,
        "decision_timeout": decision_timeout_matrix,
        "unbounded_log_probs": unbounded_logs_matrix,
        "log_probs": bounded_logs_matrix,
    }

    return (
        rollout_data,
        observations,
        critic_observations,
        previous_actions,
        hidden_state,
    )


def compute_gae(
    data: dict[str, np.ndarray],
    gamma: float = 0.999,
    lam: float = 0.95,
    reward_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate GAE advantages and lambda-return critic targets."""

    rewards = data["rewards"] * reward_scale
    values = data["values"]
    bootstrap_values = data["bootstrap_values"]
    terminated = data["terminated"]
    truncated = data["truncated"]

    effective_bootstrap = np.where(terminated, 0.0, bootstrap_values)
    deltas = rewards + gamma * effective_bootstrap - values
    continuation = (~np.logical_or(terminated, truncated)).astype(np.float32)

    advantages = np.zeros_like(rewards)
    gae = np.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        gae = deltas[t] + gamma * lam * continuation[t] * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def _run_actor_sequence(
    actor: Actor,
    observations: torch.Tensor,
    previous_actions: torch.Tensor,
    initial_hidden_state: torch.Tensor,
    memory_reset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay each environment through the GRU without mixing time order."""

    hidden_state = initial_hidden_state
    means = []
    stds = []

    for t in range(observations.shape[0]):
        mean, std, hidden_state = actor.distribution_parameters(
            observations[t], previous_actions[t], hidden_state
        )
        means.append(mean)
        stds.append(std)

        # The action at t belongs to the finishing episode. Reset memory only
        # before processing the fresh observation at t + 1.
        hidden_state = hidden_state * (~memory_reset[t]).unsqueeze(-1)

    return torch.stack(means), torch.stack(stds)


def _split_into_sequences(values: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Turn (time, env, ...) into (sequence time, sequence, ...)."""

    time_steps, num_envs = values.shape[:2]
    if time_steps % sequence_length != 0:
        raise ValueError("rollout_steps must be divisible by sequence_length")

    sequences_per_env = time_steps // sequence_length
    trailing_shape = values.shape[2:]
    values = values.reshape(sequences_per_env, sequence_length, num_envs, *trailing_shape)
    dimension_order = (1, 0, 2, *range(3, values.ndim))
    values = values.permute(dimension_order)
    return values.reshape(sequence_length, sequences_per_env * num_envs, *trailing_shape)


def _gradient_norm(module: nn.Module) -> float:
    squared_norm = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared_norm += float(parameter.grad.detach().pow(2).sum())
    return math.sqrt(squared_norm)


def ppo_update(
    actor: Actor,
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    data: dict[str, np.ndarray],
    gae_advantages: np.ndarray,
    returns: np.ndarray,
    epochs: int = 4,
    batch_size: int = 256,
    sequence_length: int = 64,
    clip_epsilon: float = 0.2,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.015,
) -> dict[str, float]:
    """Update the recurrent actor and privileged critic from one rollout."""

    if batch_size % sequence_length != 0:
        raise ValueError("batch_size must be divisible by sequence_length")

    device = next(actor.parameters()).device
    observations = torch.as_tensor(data["observations"], dtype=torch.float32, device=device)
    critic_observations = torch.as_tensor(data["critic_observations"], dtype=torch.float32, device=device)
    previous_actions = torch.as_tensor(data["previous_actions"], dtype=torch.float32, device=device)
    hidden_states = torch.as_tensor(data["hidden_states"], dtype=torch.float32, device=device)
    unclipped_actions = torch.as_tensor(data["unclipped_actions"], dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(data["log_probs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(gae_advantages, dtype=torch.float32, device=device)
    return_targets = torch.as_tensor(returns, dtype=torch.float32, device=device)
    memory_reset = torch.as_tensor(
        data["memory_reset"], dtype=torch.bool, device=device
    )

    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    observation_sequences = _split_into_sequences(observations, sequence_length)
    previous_action_sequences = _split_into_sequences(previous_actions, sequence_length)
    hidden_state_sequences = _split_into_sequences(hidden_states, sequence_length)[0]
    unclipped_action_sequences = _split_into_sequences(unclipped_actions, sequence_length)
    old_log_prob_sequences = _split_into_sequences(old_log_probs, sequence_length)
    advantage_sequences = _split_into_sequences(advantages, sequence_length)
    memory_reset_sequences = _split_into_sequences(memory_reset, sequence_length)

    critic_observation_dim = critic_observations.shape[-1]
    flat_critic_observations = critic_observations.reshape(-1, critic_observation_dim)
    flat_return_targets = return_targets.reshape(-1)

    sequence_count = observation_sequences.shape[1]
    sequences_per_batch = batch_size // sequence_length
    critic_sample_count = flat_critic_observations.shape[0]

    actor.train()
    critic.train()
    actor_stopped = False
    actor_epochs = 0
    actor_grad_norm = 0.0
    critic_grad_norm = 0.0
    critic_clipped_steps = 0
    gru_grad_norm = 0.0
    actor_steps = 0
    critic_steps = 0

    for _ in range(epochs):
        if not actor_stopped:
            sequence_order = torch.randperm(sequence_count, device=device)
            for start in range(0, sequence_count, sequences_per_batch):
                batch_indices = sequence_order[start : start + sequences_per_batch]
                batch_observations = observation_sequences[:, batch_indices]
                batch_previous_actions = previous_action_sequences[:, batch_indices]
                batch_hidden_states = hidden_state_sequences[batch_indices]
                batch_unclipped_actions = unclipped_action_sequences[:, batch_indices]
                batch_old_log_probs = old_log_prob_sequences[:, batch_indices]
                batch_advantages = advantage_sequences[:, batch_indices]
                batch_memory_reset = memory_reset_sequences[:, batch_indices]

                mean, std = _run_actor_sequence(
                    actor, batch_observations, batch_previous_actions,
                    batch_hidden_states, batch_memory_reset,
                )
                normal = torch.distributions.Normal(mean, std)
                new_log_probs = squashed_log_prob(normal, batch_unclipped_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                objective_unclipped = ratio * batch_advantages
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                objective_clipped = clipped_ratio * batch_advantages
                actor_loss = -torch.min(objective_unclipped, objective_clipped).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                gru_grad_norm = _gradient_norm(actor.gru)
                actor_grad_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm))
                actor_optimizer.step()
                actor_steps += 1

            actor_epochs += 1
            with torch.no_grad():
                mean, std = _run_actor_sequence(
                    actor, observation_sequences, previous_action_sequences,
                    hidden_state_sequences, memory_reset_sequences,
                )
                normal = torch.distributions.Normal(mean, std)
                new_log_probs = squashed_log_prob(normal, unclipped_action_sequences)
                log_ratio = new_log_probs - old_log_prob_sequences
                ratio = log_ratio.exp()
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
            actor_stopped = bool(approx_kl.item() > target_kl)

        critic_order = torch.randperm(critic_sample_count, device=device)
        for start in range(0, critic_sample_count, batch_size):
            batch_indices = critic_order[start : start + batch_size]
            batch_observations = flat_critic_observations[batch_indices]
            batch_returns = flat_return_targets[batch_indices]
            new_values = critic(batch_observations).squeeze(-1)
            critic_loss = 0.5 * (new_values - batch_returns).pow(2).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            )
            critic_clipped_steps += int(critic_grad_norm > max_grad_norm)
            critic_optimizer.step()
            critic_steps += 1

    with torch.no_grad():
        mean, std = _run_actor_sequence(
            actor, observation_sequences, previous_action_sequences,
            hidden_state_sequences, memory_reset_sequences,
        )
        normal = torch.distributions.Normal(mean, std)
        new_log_probs = squashed_log_prob(normal, unclipped_action_sequences)
        log_ratio = new_log_probs - old_log_prob_sequences
        ratio = log_ratio.exp()
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        actor_loss = -torch.min(
            ratio * advantage_sequences, clipped_ratio * advantage_sequences
        ).mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (torch.abs(ratio - 1.0) > clip_epsilon).float().mean()

        new_values = critic(flat_critic_observations).squeeze(-1)
        critic_loss = 0.5 * (new_values - flat_return_targets).pow(2).mean()
        target_variance = flat_return_targets.var(unbiased=False)
        if target_variance.item() > 1e-8:
            residual_variance = (flat_return_targets - new_values).var(unbiased=False)
            explained_variance = 1.0 - residual_variance / target_variance
        else:
            explained_variance = torch.zeros((), device=device)

        target_mean = flat_return_targets.mean()
        target_std = flat_return_targets.std(unbiased=False)
        value_mean = new_values.mean()
        value_std = new_values.std(unbiased=False)
        if target_std.item() > 1e-8 and value_std.item() > 1e-8:
            covariance = (
                (flat_return_targets - target_mean) * (new_values - value_mean)
            ).mean()
            value_target_correlation = covariance / (target_std * value_std)
        else:
            value_target_correlation = torch.zeros((), device=device)

    hidden_norm = np.linalg.norm(data["hidden_states"], axis=-1).mean()
    mean_std = std.mean(dim=(0, 1))
    return {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "approx_kl": float(approx_kl.item()),
        "clip_fraction": float(clip_fraction.item()),
        "explained_variance": float(explained_variance.item()),
        "throttle_std": float(mean_std[0].item()),
        "gimbal_std": float(mean_std[1].item()),
        "hidden_norm": float(hidden_norm),
        "gru_grad_norm": gru_grad_norm,
        "actor_grad_norm": actor_grad_norm,
        "critic_grad_norm": critic_grad_norm,
        "critic_clip_fraction": critic_clipped_steps / critic_steps,
        "target_mean": float(target_mean.item()),
        "target_std": float(target_std.item()),
        "value_mean": float(value_mean.item()),
        "value_std": float(value_std.item()),
        "value_target_correlation": float(value_target_correlation.item()),
        "actor_epochs": float(actor_epochs),
        "actor_steps": float(actor_steps),
        "critic_steps": float(critic_steps),
    }


@torch.no_grad()
def evaluate_actor(
    actor: Actor,
    task: TrainingTask,
    reset_options: dict | None = None,
    episodes: int = 20,
    base_seed: int = 20_000,
    decision_limit: int | None = None,
) -> dict[str, float]:
    """Evaluate the recurrent policy using its deterministic mean action."""

    env = make_training_env(
        task=task,
        action_repeat=4,
        decision_limit=decision_limit,
    )
    device = next(actor.parameters()).device
    successes = 0
    samples_acquired = 0
    lengths = []
    actor.eval()

    try:
        for episode in range(episodes):
            observation, _ = env.reset(
                seed=base_seed + episode,
                options=dict(reset_options or {}),
            )
            previous_action = torch.zeros((1, 2), dtype=torch.float32, device=device)
            hidden_state = actor.initial_hidden_state(1).to(device)
            sample_acquired = False

            for decision in range(1, env.decision_limit + 1):
                observation_tensor = torch.as_tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
                mean, _, hidden_state = actor.distribution_parameters(
                    observation_tensor, previous_action, hidden_state
                )
                squashed = torch.tanh(mean)
                action_tensor = torch.stack(
                    ((squashed[..., 0] + 1.0) / 2.0, squashed[..., 1]), dim=-1
                )
                action = action_tensor.squeeze(0).cpu().numpy().astype(
                    np.float32, copy=False
                )
                observation, _, terminated, truncated, info = env.step(action)
                previous_action = action_tensor
                sample_acquired = sample_acquired or bool(info["has_sample"])

                if info["auto_advanced_sampling_steps"] > 0:
                    previous_action.zero_()
                    hidden_state.zero_()

                if terminated or truncated:
                    successes += int(info["task_success"])
                    samples_acquired += int(sample_acquired)
                    lengths.append(decision)
                    break
    finally:
        env.close()

    return {
        "success_rate": successes / episodes,
        "sample_acquired_rate": samples_acquired / episodes,
        "mean_length": float(np.mean(lengths)),
    }


def save_checkpoint(
    path: Path,
    actor: Actor,
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    metadata: dict,
) -> None:
    """Save the networks, optimizers, and experiment metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "actor_observation_dim": ACTOR_OBSERVATION_DIM,
        "critic_observation_dim": CRITIC_OBSERVATION_DIM,
        "hidden_width": actor.hidden_width,
        "action_dim": actor.mean_head.out_features,
    }
    checkpoint.update(metadata)
    torch.save(checkpoint, path)


def evaluate_tasks(
    actor: Actor,
    tasks: tuple[TrainingTask, ...],
    reset_options: dict,
    episodes: int,
    base_seed: int,
) -> dict[str, dict[str, float]]:
    """Evaluate each task separately over the same reset distribution."""

    return {
        task.value: evaluate_actor(
            actor,
            task,
            reset_options=reset_options,
            episodes=episodes,
            base_seed=base_seed + 10_000 * task_index,
        )
        for task_index, task in enumerate(tasks)
    }


def main():
    torch.manual_seed(0)

    num_envs = 16
    outbound_leg = TrainingTask.OUTBOUND_LEG
    return_leg = TrainingTask.RETURN_LEG
    dry_tasks = (outbound_leg,) * num_envs
    balanced_tasks = (outbound_leg,) * 8 + (return_leg,) * 8

    rollout_steps = 512
    batch_size = 512
    gamma = 0.9999
    gae_lambda = 0.995
    reward_scale = 0.01
    evaluation_interval = 5
    max_updates_per_region = 60
    required_gate_streak = 2
    whole_evaluation_episodes = 50
    frontier_evaluation_episodes = 20
    anchor_evaluation_episodes = 20
    whole_success_threshold = 0.90
    frontier_success_threshold = 0.80
    anchor_success_threshold = 0.90
    payload_success_threshold = 0.80

    distance_step = 0.0
    initial_edge_distance = 10.0
    minimum_edge_distance = 0.25
    target_edge_distance = initial_edge_distance
    minimum_altitude = 5.0
    maximum_altitude = 10.0
    altitude_reference = "crater_rim"
    theta_range = (-0.08, 0.08)
    center_probability = 1.0 / 6.0
    inside_probability = 1.0 / 6.0
    payload_curriculum = (
        ("return context", (0.0, 0.0), (0.0, 0.0)),
        ("light centered payload", (0.0, 0.15), (0.0, 0.0)),
        ("light offset payload", (0.05, 0.22), (0.10, 0.40)),
        ("medium offset payload", (0.12, 0.30), (0.25, 0.65)),
        ("near-production payload", (0.20, 0.35), (0.50, 0.80)),
        ("broad payload certification", (0.0, 0.35), (0.0, 0.80)),
    )
    training_mode = "fixed_opening_severity_v3"

    checkpoint_dir = Path(__file__).resolve().parents[2] / "artifacts"
    latest_checkpoint_path = checkpoint_dir / "ppo_sample_return_opening_severity_v3_latest.pt"
    previous_checkpoint_path = checkpoint_dir / "ppo_sample_return_opening_severity_v3_previous.pt"
    working_checkpoint_path = checkpoint_dir / "ppo_sample_return_opening_severity_v3_working.pt"

    hidden_width = 128
    action_dim = 2
    actorNetwork = Actor(ACTOR_OBSERVATION_DIM, hidden_width, action_dim)
    criticNetwork = Critic(CRITIC_OBSERVATION_DIM, hidden_width, 1)
    actorOptimizer = torch.optim.Adam(actorNetwork.parameters(), lr=1e-4)
    criticOptimizer = torch.optim.Adam(criticNetwork.parameters(), lr=3e-4)

    checkpoint_path = next(
        (
            path
            for path in (
                working_checkpoint_path,
                latest_checkpoint_path,
                previous_checkpoint_path,
            )
            if path.exists()
        ),
        None,
    )
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint_mode = checkpoint.get("training_mode")
        if checkpoint_mode != training_mode:
            raise ValueError("opening checkpoint training mode does not match")
        if int(checkpoint["actor_observation_dim"]) != ACTOR_OBSERVATION_DIM:
            raise ValueError("checkpoint actor observation dimension does not match")
        if int(checkpoint["critic_observation_dim"]) != CRITIC_OBSERVATION_DIM:
            raise ValueError("checkpoint critic observation dimension does not match")
        expected_settings = {
            "distance_step": distance_step,
            "minimum_edge_distance": minimum_edge_distance,
            "target_edge_distance": target_edge_distance,
            "minimum_altitude": minimum_altitude,
            "maximum_altitude": maximum_altitude,
            "inside_probability": inside_probability,
            "center_probability": center_probability,
        }
        for name, expected in expected_settings.items():
            if float(checkpoint[name]) != float(expected):
                raise ValueError(f"checkpoint {name} does not match this experiment")
        if checkpoint["altitude_reference"] != altitude_reference:
            raise ValueError("checkpoint altitude reference does not match")
        if tuple(checkpoint["theta_range"]) != theta_range:
            raise ValueError("checkpoint theta range does not match")
        actorNetwork.load_state_dict(checkpoint["actor_state_dict"])
        criticNetwork.load_state_dict(checkpoint["critic_state_dict"])
        actorOptimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        criticOptimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        accepted_edge_distance = float(checkpoint["accepted_edge_distance"])
        training_phase = str(checkpoint["training_phase"])
        if training_phase not in {"dry", "payload", "balanced"}:
            raise ValueError("checkpoint training phase is not valid")
        if tuple(checkpoint["payload_curriculum"]) != payload_curriculum:
            raise ValueError("checkpoint payload curriculum does not match")
        payload_stage_index = int(checkpoint["payload_stage_index"])
        region_index = int(checkpoint["region_index"])
        phase_update = int(checkpoint.get("phase_update", 0))
        gate_streak = int(checkpoint.get("gate_streak", 0))
        if gate_streak >= required_gate_streak:
            # A working checkpoint is saved just before promotion. Recheck
            # once after a restart instead of getting stuck in the old phase.
            gate_streak = required_gate_streak - 1
        region_certified = bool(checkpoint.get("region_certified", False))
        global_update = int(checkpoint["global_update"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        startup = f"resumed {checkpoint_path.name}"
    else:
        accepted_edge_distance = initial_edge_distance
        training_phase = "dry"
        payload_stage_index = 0
        region_index = 0
        global_update = 0
        phase_update = 0
        gate_streak = 0
        region_certified = False
        startup = "started fresh actor, critic, and optimizers"

    print(
        f"{startup} | symmetric start mixture | "
        f"inside={inside_probability:.0%} center={center_probability:.0%} "
        f"outside={1.0 - inside_probability - center_probability:.0%}"
    )

    def checkpoint_metadata(gate_results):
        return {
            "training_mode": training_mode,
            "training_phase": training_phase,
            "payload_curriculum": payload_curriculum,
            "payload_stage_index": payload_stage_index,
            "accepted_edge_distance": accepted_edge_distance,
            "region_certified": region_certified,
            "region_index": region_index,
            "global_update": global_update,
            "phase_update": phase_update,
            "gate_streak": gate_streak,
            "distance_step": distance_step,
            "minimum_edge_distance": minimum_edge_distance,
            "target_edge_distance": target_edge_distance,
            "minimum_altitude": minimum_altitude,
            "maximum_altitude": maximum_altitude,
            "altitude_reference": altitude_reference,
            "theta_range": theta_range,
            "inside_probability": inside_probability,
            "center_probability": center_probability,
            "whole_success_threshold": whole_success_threshold,
            "frontier_success_threshold": frontier_success_threshold,
            "anchor_success_threshold": anchor_success_threshold,
            "payload_success_threshold": payload_success_threshold,
            "required_gate_streak": required_gate_streak,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "reward_scale": reward_scale,
            "batch_size": batch_size,
            "sequence_length": 64,
            "gate_results": gate_results,
            "torch_rng_state": torch.get_rng_state(),
        }

    while True:
        if training_phase == "dry":
            training_tasks = dry_tasks
            evaluation_tasks = (outbound_leg,)
            candidate_edge_distance = accepted_edge_distance
            stage_name = "dry navigation"
            payload_options = {}
            payload_corner = None
        elif training_phase == "payload":
            training_tasks = balanced_tasks
            evaluation_tasks = (outbound_leg, return_leg)
            candidate_edge_distance = accepted_edge_distance
            payload_name, mass_range, offset_range = payload_curriculum[
                payload_stage_index
            ]
            payload_options = {
                "payload_mass_range": mass_range,
                "payload_offset_body_x_range": offset_range,
                "payload_offset_body_y": 0.0,
            }
            payload_corner = (mass_range[1], offset_range[1])
            stage_name = f"payload {payload_stage_index + 1}/{len(payload_curriculum)}: {payload_name}"
        else:
            training_tasks = balanced_tasks
            evaluation_tasks = (outbound_leg, return_leg)
            _, mass_range, offset_range = payload_curriculum[-1]
            payload_options = {
                "payload_mass_range": mass_range,
                "payload_offset_body_x_range": offset_range,
                "payload_offset_body_y": 0.0,
            }
            payload_corner = (mass_range[1], offset_range[1])
            if not region_certified:
                candidate_edge_distance = accepted_edge_distance
                stage_name = "balanced certification"
            elif accepted_edge_distance >= target_edge_distance:
                print(
                    "fixed opening distribution complete | "
                    f"outside pad edge={minimum_edge_distance:g}-"
                    f"{accepted_edge_distance:g} m | rim clearance="
                    f"{minimum_altitude:g}-{maximum_altitude:g} m"
                )
                print(f"mastered checkpoint: {latest_checkpoint_path}")
                return
            else:
                candidate_edge_distance = min(
                    target_edge_distance, accepted_edge_distance + distance_step
                )
                stage_name = "balanced expansion"

        fixed_spawn_options = {
            "spawn_altitude_reference": altitude_reference,
            "spawn_altitude_range": (minimum_altitude, maximum_altitude),
            "spawn_theta_range": theta_range,
        }
        region_options = {
            "spawn_pad_edge_distance_range": (
                minimum_edge_distance,
                candidate_edge_distance,
            ),
            "spawn_inside_pad_probability": inside_probability,
            "spawn_pad_center_probability": center_probability,
            **fixed_spawn_options,
            **payload_options,
        }
        pad_half_width = DEFAULT_SAMPLE_RETURN_CONFIG.flight.pad_half_w
        left_frontier_options = {
            "spawn_mode": "airborne",
            "spawn_reference": "target",
            "spawn_x_offset": -(pad_half_width + candidate_edge_distance),
            **fixed_spawn_options,
            **payload_options,
        }
        right_frontier_options = dict(left_frontier_options)
        right_frontier_options["spawn_x_offset"] = (
            pad_half_width + candidate_edge_distance
        )
        inside_anchor_options = {
            "spawn_pad_edge_distance_range": (0.0, 0.0),
            "spawn_inside_pad_probability": 1.0,
            **fixed_spawn_options,
            **payload_options,
        }
        center_anchor_options = {
            "spawn_mode": "airborne",
            "spawn_reference": "target",
            "spawn_x_offset": 0.0,
            **fixed_spawn_options,
            **payload_options,
        }
        payload_left_corner_options = None
        payload_right_corner_options = None
        if payload_corner is not None:
            payload_left_corner_options = dict(left_frontier_options)
            payload_right_corner_options = dict(right_frontier_options)
            for options in (
                payload_left_corner_options,
                payload_right_corner_options,
            ):
                options.pop("payload_mass_range")
                options.pop("payload_offset_body_x_range")
                options["payload_mass"] = payload_corner[0]
                options["payload_offset_body_x"] = payload_corner[1]

        candidate_number = region_index + 1
        print(
            f"\nregion {candidate_number} | phase={stage_name} | outside pad edge="
            f"{minimum_edge_distance:g}-{candidate_edge_distance:g} m "
            f"| center offset up to {pad_half_width + candidate_edge_distance:g} m "
            f"| rim clearance={minimum_altitude:g}-{maximum_altitude:g} m "
            f"| theta={theta_range[0]:g}-{theta_range[1]:g} rad"
        )

        envs = make_vector_env(
            num_envs, task=training_tasks, action_repeat=4, asynchronous=True
        )
        interrupted = False
        mastered = False
        gate_results = {}
        try:
            stage_seed = 100 + candidate_number * 10_000
            observations, reset_info = envs.reset(
                seed=[stage_seed + index for index in range(num_envs)],
                options=dict(region_options),
            )
            critic_observations = np.asarray(
                reset_info["critic_observation"], dtype=np.float32
            )
            previous_actions = np.zeros(
                (num_envs, envs.single_action_space.shape[0]), dtype=np.float32
            )
            hidden_state = actorNetwork.initial_hidden_state(num_envs)

            running_returns = np.zeros(num_envs, dtype=np.float64)
            recent_returns: deque[float] = deque(maxlen=100)
            recent_task_successes = {
                task: deque(maxlen=100) for task in evaluation_tasks
            }

            updates_this_run = 0
            while updates_this_run < max_updates_per_region:
                global_update += 1
                phase_update += 1
                updates_this_run += 1
                candidate_update = phase_update
                update_started_at = time.perf_counter()
                (
                    rollout_data,
                    observations,
                    critic_observations,
                    previous_actions,
                    hidden_state,
                ) = collect_rollout(
                    envs, observations, critic_observations, previous_actions,
                    hidden_state, actorNetwork, criticNetwork, rollout_steps,
                    reset_options=region_options,
                )

                for t in range(rollout_steps):
                    running_returns += rollout_data["rewards"][t]
                    done = np.logical_or(
                        rollout_data["terminated"], rollout_data["truncated"]
                    )[t]
                    for env_index in np.flatnonzero(done):
                        recent_returns.append(float(running_returns[env_index]))
                        recent_task_successes[training_tasks[env_index]].append(
                            bool(rollout_data["task_success"][t, env_index])
                        )
                    running_returns[done] = 0.0

                advantages, returns = compute_gae(
                    rollout_data, gamma=gamma, lam=gae_lambda,
                    reward_scale=reward_scale,
                )
                stats = ppo_update(
                    actorNetwork, criticNetwork, actorOptimizer, criticOptimizer,
                    rollout_data, advantages, returns, batch_size=batch_size,
                )

                mean_return = (
                    float(np.mean(recent_returns)) if recent_returns else float("nan")
                )
                task_summary = " ".join(
                    f"{task.value}={np.mean(recent_task_successes[task]):.0%}"
                    if recent_task_successes[task]
                    else f"{task.value}=n/a"
                    for task in evaluation_tasks
                )
                update_seconds = time.perf_counter() - update_started_at
                print(
                    f"update {global_update:3d} | phase_update={candidate_update:3d} "
                    f"| block={updates_this_run:2d}/{max_updates_per_region} "
                    f"| return={mean_return:8.3f} "
                    f"| {task_summary} | time={update_seconds:5.1f}s"
                )
                print(
                    f"  kl={stats['approx_kl']:.5f} "
                    f"| clip={stats['clip_fraction']:6.1%} "
                    f"| std=({stats['throttle_std']:.3f}, "
                    f"{stats['gimbal_std']:.3f}) "
                    f"| value_fit={stats['explained_variance']:.3f} "
                    f"| gru_grad={stats['gru_grad_norm']:.4f} "
                    f"| actor_epochs={int(stats['actor_epochs'])}"
                )

                if candidate_update % evaluation_interval != 0:
                    continue

                evaluation_seed = 1_000_000
                whole_results = evaluate_tasks(
                    actorNetwork, evaluation_tasks, region_options,
                    whole_evaluation_episodes, evaluation_seed,
                )
                left_frontier_results = evaluate_tasks(
                    actorNetwork, evaluation_tasks, left_frontier_options,
                    frontier_evaluation_episodes, evaluation_seed + 100,
                )
                right_frontier_results = evaluate_tasks(
                    actorNetwork, evaluation_tasks, right_frontier_options,
                    frontier_evaluation_episodes, evaluation_seed + 200,
                )
                inside_anchor_results = evaluate_tasks(
                    actorNetwork, evaluation_tasks, inside_anchor_options,
                    anchor_evaluation_episodes, evaluation_seed + 300,
                )
                center_anchor_results = evaluate_tasks(
                    actorNetwork, evaluation_tasks, center_anchor_options,
                    anchor_evaluation_episodes, evaluation_seed + 400,
                )
                payload_left_corner_results = None
                payload_right_corner_results = None
                if payload_left_corner_options is not None:
                    payload_left_corner_results = evaluate_tasks(
                        actorNetwork, evaluation_tasks, payload_left_corner_options,
                        frontier_evaluation_episodes, evaluation_seed + 500,
                    )
                    payload_right_corner_results = evaluate_tasks(
                        actorNetwork, evaluation_tasks, payload_right_corner_options,
                        frontier_evaluation_episodes, evaluation_seed + 600,
                    )
                gate_results = {
                    "whole": whole_results,
                    "left_frontier": left_frontier_results,
                    "right_frontier": right_frontier_results,
                    "inside_anchor": inside_anchor_results,
                    "center_anchor": center_anchor_results,
                }
                if payload_left_corner_results is not None:
                    gate_results["payload_left_corner"] = payload_left_corner_results
                    gate_results["payload_right_corner"] = payload_right_corner_results

                def success_summary(results):
                    return "/".join(
                        f"{results[task.value]['success_rate']:.0%}"
                        for task in evaluation_tasks
                    )

                task_names = "/".join(task.value for task in evaluation_tasks)
                print(
                    f"  deterministic gate {task_names} | "
                    f"whole={success_summary(whole_results)} "
                    f"left={success_summary(left_frontier_results)} "
                    f"right={success_summary(right_frontier_results)}"
                )
                print(
                    f"  landing retention {task_names} | "
                    f"inside={success_summary(inside_anchor_results)} "
                    f"center={success_summary(center_anchor_results)}"
                )
                if payload_left_corner_results is not None:
                    print(
                        f"  payload upper corner {payload_corner[0]:g} kg/"
                        f"{payload_corner[1]:g} m | "
                        f"left={success_summary(payload_left_corner_results)} "
                        f"right={success_summary(payload_right_corner_results)}"
                    )

                whole_passed = all(
                    result["success_rate"] >= whole_success_threshold
                    for result in whole_results.values()
                )
                left_frontier_passed = all(
                    result["success_rate"] >= frontier_success_threshold
                    for result in left_frontier_results.values()
                )
                right_frontier_passed = all(
                    result["success_rate"] >= frontier_success_threshold
                    for result in right_frontier_results.values()
                )
                inside_anchor_passed = all(
                    result["success_rate"] >= anchor_success_threshold
                    for result in inside_anchor_results.values()
                )
                center_anchor_passed = all(
                    result["success_rate"] >= anchor_success_threshold
                    for result in center_anchor_results.values()
                )
                payload_corner_passed = (
                    payload_left_corner_results is None
                    or (
                        all(
                            result["success_rate"] >= payload_success_threshold
                            for result in payload_left_corner_results.values()
                        )
                        and all(
                            result["success_rate"] >= payload_success_threshold
                            for result in payload_right_corner_results.values()
                        )
                    )
                )
                gate_passed = (
                    whole_passed
                    and left_frontier_passed
                    and right_frontier_passed
                    and inside_anchor_passed
                    and center_anchor_passed
                    and payload_corner_passed
                )
                gate_streak = gate_streak + 1 if gate_passed else 0
                print(f"  gate streak {gate_streak}/{required_gate_streak}")

                working_temporary_path = working_checkpoint_path.with_suffix(".tmp")
                save_checkpoint(
                    working_temporary_path,
                    actorNetwork,
                    criticNetwork,
                    actorOptimizer,
                    criticOptimizer,
                    checkpoint_metadata(gate_results),
                )
                working_temporary_path.replace(working_checkpoint_path)

                if gate_streak >= required_gate_streak:
                    mastered = True
                    break
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            envs.close(terminate=interrupted)

        if not mastered:
            print(
                "region held after maximum updates; "
                "last mastered checkpoint remains unchanged"
            )
            if latest_checkpoint_path.exists():
                print(f"mastered checkpoint: {latest_checkpoint_path}")
            else:
                print("no symmetric region has been mastered yet")
            return

        if training_phase == "dry":
            training_phase = "payload"
            payload_stage_index = 0
            region_certified = False
            print("  dry navigation mastered; beginning the payload curriculum")
        elif training_phase == "payload":
            if payload_stage_index + 1 < len(payload_curriculum):
                payload_stage_index += 1
                next_payload_name = payload_curriculum[payload_stage_index][0]
                print(f"  payload stage mastered; next: {next_payload_name}")
            else:
                training_phase = "balanced"
                region_certified = True
                print("  broad payload range certified on the fixed distribution")
        elif not region_certified:
            region_certified = True
            print("  balanced policy certified; beginning distance expansion")
        else:
            accepted_edge_distance = candidate_edge_distance
        region_index += 1
        phase_update = 0
        gate_streak = 0

        metadata = checkpoint_metadata(gate_results)
        temporary_checkpoint_path = latest_checkpoint_path.with_suffix(".tmp")
        save_checkpoint(
            temporary_checkpoint_path,
            actorNetwork,
            criticNetwork,
            actorOptimizer,
            criticOptimizer,
            metadata,
        )
        if latest_checkpoint_path.exists():
            latest_checkpoint_path.replace(previous_checkpoint_path)
        working_checkpoint_path.unlink(missing_ok=True)
        temporary_checkpoint_path.replace(latest_checkpoint_path)
        print(f"  region mastered and saved: {latest_checkpoint_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ntraining interrupted; last mastered rectangle preserved")


__all__ = [
    "Actor",
    "Critic",
    "choose_action",
    "collect_rollout",
    "compute_gae",
    "ppo_update",
    "squashed_log_prob",
]
