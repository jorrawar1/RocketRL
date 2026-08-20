# RocketRL sample-return training contract

This scaffold deliberately stops at the boundary of the learning algorithm.
It supplies curriculum episode boundaries, agent-rate stepping, deterministic
vector environments, and reward-independent evaluation. The actor, critic,
rollout storage, GAE, PPO loss, optimizer, and checkpoints remain user-owned.

## One policy, five training tasks

Every task uses the same action space, 16-value observation, vehicle physics,
and network parameters. A task changes only the initial-state distribution and
the point at which that shorter curriculum episode counts as complete.

| Task | Start | Successful end |
|---|---|---|
| `sample_landing` | Empty rocket, airborne above the sample target | Safe sample touchdown |
| `return_landing` | Loaded rocket, airborne above base | Safe base touchdown |
| `outbound_leg` | Empty rocket on the base pad | Safe sample touchdown |
| `return_leg` | Loaded rocket on the sample pad | Safe base touchdown |
| `full_mission` | Empty rocket on the base pad | Sample returned to base |

A reasonable first curriculum is to mix both landing tasks, then both complete
legs, and finally full missions. Do not create a different neural network for
each task. Reuse the same policy and keep some easier episodes in later batches
if the policy begins forgetting its landing behavior.

The reset API also accepts deterministic target-relative perturbations:

```python
observation, info = env.reset(
    seed=7,
    options={
        "task": "return_landing",
        "spawn_altitude": 12.0,
        "spawn_x_offset": -2.0,
        "spawn_vx": 0.5,
        "spawn_vy": -1.0,
        "spawn_theta": 0.08,
        "spawn_omega": -0.05,
    },
)
```

Sample those values in the training code when it is time to widen the
curriculum. The scaffold does not contain an automatic difficulty scheduler.

## Observation contract

The default policy input has exactly 16 `float32` values:

| Index | Name | Scale or meaning |
|---:|---|---|
| 0 | `target_dx` | Active target x minus vehicle x, divided by world width |
| 1 | `target_dy` | Active target ground y minus vehicle y, divided by world height |
| 2 | `velocity_x` | World x velocity divided by `v_ref` |
| 3 | `velocity_y` | World y velocity divided by `v_ref` |
| 4 | `sin_theta` | Sine of vehicle attitude |
| 5 | `cos_theta` | Cosine of vehicle attitude |
| 6 | `angular_velocity` | Angular velocity divided by `omega_ref` |
| 7 | `fuel_fraction` | Remaining fuel divided by initial fuel |
| 8–12 | `terrain_ray_0` … `terrain_ray_4` | World-frame downward ray distances divided by maximum range |
| 13 | `payload_attached` | Binary flag |
| 14 | `phase_outbound` | Binary flag |
| 15 | `phase_return` | Binary flag |

`OBSERVATION_NAMES`, `OBSERVATION_INDEX`, and `OBSERVATION_DIM` are exported by
`rocketenv.sample_return`. Exact payload mass and attachment offset appear in
`info` for analysis but are not part of the policy observation.

## Action and timing contract

The action is:

```text
[throttle in 0..1, normalized gimbal in -1..1]
```

The simulator runs at 60 physics frames per second. The training wrapper
defaults to `action_repeat=4`, so the policy makes decisions at 15 Hz. It sums
the four frame rewards and stops early on termination or truncation.

The 90-frame sample-loading dwell is automatically advanced with zero actions
and folded into the sample-touchdown transition. The wrapper then stops that
decision immediately, so an outbound touchdown action is never reused after
the loaded return vehicle becomes active.

Use a bounded policy distribution. Do not depend on environment clipping for
Gaussian samples: PPO's stored log probability must describe the action that
was actually applied.

## Vector environments

```python
import numpy as np

from rocketenv.sample_return import TrainingTask, make_vector_env

tasks = [
    TrainingTask.SAMPLE_LANDING,
    TrainingTask.SAMPLE_LANDING,
    TrainingTask.RETURN_LANDING,
    TrainingTask.RETURN_LANDING,
]
envs = make_vector_env(4, task=tasks, action_repeat=4)
observations, infos = envs.reset(seed=0)

# actions has shape (4, 2)
next_observations, rewards, terminated, truncated, infos = envs.step(actions)
done = np.logical_or(terminated, truncated)
terminal_observations = next_observations.copy()

if done.any():
    # Save/use terminal_observations before resetting. The returned array keeps
    # current observations for live slots and replaces only the reset slots.
    next_observations, reset_infos = envs.reset(
        options={"reset_mask": done.copy()}
    )
```

Autoreset is intentionally disabled. This keeps the true terminal observation
available to the PPO rollout code:

- `terminated=True`: use zero as the bootstrap value.
- `truncated=True`: bootstrap from the value of the terminal observation.
- Stop the GAE recurrence at both kinds of episode boundary so it cannot flow
  into a reset episode.

Vector `info` is a dictionary of arrays. Gymnasium adds a `_name` boolean mask
alongside optional field `name` to indicate which environments supplied it.

## Evaluation

Evaluation uses outcome and physical flight metrics, not shaped reward:

```powershell
.venv\Scripts\python.exe scripts\evaluate_sample_return.py `
  --controller scripted `
  --task full_mission `
  --seed-start 10000 `
  --episodes 20 `
  --action-repeat 4 `
  --output artifacts\eval_scripted.json
```

The scripted controller is only a feasibility oracle. A learned policy plugs
into the programmatic evaluator through an object with `reset()` and
`act(observation)`, or through `CallableController`. Evaluate the learned policy
deterministically on a fixed held-out seed list, while retaining training runs
across several random seeds.

## Code still to write for PPO

The next implementation work belongs in the learning layer:

1. Actor distribution and critic network.
2. Vector rollout collection with observations, actions, rewards, values,
   log probabilities, termination flags, and truncation flags.
3. Discounted returns and GAE, with hand-calculated unit tests.
4. PPO clipped policy loss, value loss, and entropy term.
5. Advantage normalization, shuffled minibatches, and repeated update epochs.
6. Logging for success rate, physical outcomes, approximate KL, clip fraction,
   entropy, value loss, and explained variance.
7. Checkpoint save/load and deterministic held-out evaluation.

Begin with the fixed payload. Payload randomization is a later environment
distribution change, not part of the first PPO implementation.
