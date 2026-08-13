"""Generate a behaviour-cloning dataset from the scripted expert.

Records (observation, expert action) pairs: the observation is what a policy
will actually see at inference time, the action is what the expert -- which
reads privileged raw state -- did at that moment.

Run from the repo root:
    .venv\\Scripts\\python learning\\generate_dataset.py --episodes 100

The interesting choices are all flags, so they are easy to change and easy
to see. Defaults: flat terrain, action noise on, failures discarded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# running a script inside learning/ puts learning/ on sys.path, not the repo
# root, so rocketenv would not be importable without this
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rocketenv import FlatTerrain, RocketEnv, generate_terrain
from rocketenv.reward import TOUCHDOWN
from rocketenv.scripted import scripted_action

# The notebook's evaluate() starts at seed 10_000; training episodes must stay
# below that so evaluation is never run on a spawn the policy trained on.
EVAL_SEED_START = 10_000


def collect(n_episodes=100, noise_std=0.05, keep_failures=False,
            terrain="flat", seed=0):
    """Roll out the expert and return (obs, act, episode_id, outcomes)."""
    assert n_episodes < EVAL_SEED_START, "would collide with evaluation seeds"

    rng = np.random.default_rng(seed)          # drives terrain + action noise
    env = RocketEnv()

    obs_all, act_all, ep_all, outcomes = [], [], [], []
    kept = 0

    for ep in range(n_episodes):
        # --- pick the map -------------------------------------------------
        if terrain == "flat":
            env.terrain = FlatTerrain()
            options = None
        else:
            env.terrain, pad_x = generate_terrain(rng, env.base_config)
            options = {"pad_x": pad_x}         # keep the env's pad on the flat bit

        obs, info = env.reset(seed=ep, options=options)

        # --- fly one episode, buffering so a failure can be dropped whole --
        ep_obs, ep_act = [], []
        while True:
            # expert reads raw state (privileged); this clean action is the label
            action = scripted_action(env.state, env.cfg, env.terrain)
            ep_obs.append(obs)                 # obs at time t pairs with action at t
            ep_act.append(action)

            # execute a perturbed action so the trajectory wanders slightly and
            # the dataset covers states a slightly-wrong policy would reach
            executed = action
            if noise_std > 0.0:
                executed = np.clip(action + rng.normal(0.0, noise_std, 2),
                                   [0.0, -1.0], [1.0, 1.0])

            obs, _reward, terminated, truncated, info = env.step(executed)
            if terminated or truncated:
                break

        outcome = info.get("outcome", "TIMEOUT")   # no "outcome" key on timeout
        outcomes.append(outcome)

        if keep_failures or outcome == TOUCHDOWN:
            obs_all.extend(ep_obs)
            act_all.extend(ep_act)
            ep_all.extend([ep] * len(ep_obs))
            kept += 1

        print(f"episode {ep:4d}  {outcome:<16} {len(ep_obs):5d} steps  "
              f"(kept {kept}/{ep + 1})")

    return (np.asarray(obs_all, dtype=np.float32),
            np.asarray(act_all, dtype=np.float32),
            np.asarray(ep_all, dtype=np.int32),
            outcomes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--noise", type=float, default=0.05,
                   help="std of gaussian noise added to the EXECUTED action")
    p.add_argument("--keep-failures", action="store_true",
                   help="record crashed episodes too (default: discard)")
    p.add_argument("--terrain", choices=["flat", "generated"], default="flat")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data.npz")
    args = p.parse_args()

    obs, act, ep, outcomes = collect(
        n_episodes=args.episodes, noise_std=args.noise,
        keep_failures=args.keep_failures, terrain=args.terrain, seed=args.seed,
    )

    np.savez_compressed(args.out, obs=obs, act=act, episode=ep)

    from collections import Counter
    print()
    print(f"wrote {args.out}")
    print(f"  outcomes over all episodes : {dict(Counter(outcomes))}")
    print(f"  episodes kept              : {len(np.unique(ep))}")
    print(f"  samples                    : {len(obs)}")
    print(f"  obs {obs.shape} {obs.dtype}   act {act.shape} {act.dtype}")
    print(f"  throttle  mean {act[:, 0].mean():.3f}  std {act[:, 0].std():.3f}  "
          f"frac at 1.0: {(act[:, 0] > 0.999).mean():.1%}")
    print(f"  gimbal    mean {act[:, 1].mean():+.3f}  std {act[:, 1].std():.3f}  "
          f"frac saturated: {(np.abs(act[:, 1]) > 0.999).mean():.1%}")


if __name__ == "__main__":
    main()
