# rocket-rl — Phase 0/1: environment + playable harness

A deterministic 2D rocket-landing environment (Gymnasium API) and a `pygame`
keyboard harness. This is the physics an RL agent will later train on and a
TypeScript port will mirror — correctness and determinism over features.

## Setup & run

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python play.py          # fly it
.venv\Scripts\python -m pytest tests  # test it
```

(`pygame-ce` is the community fork of pygame — it imports as `pygame`
unchanged; it's used here for prompt Python 3.14 wheel support.)

## Controls

| Key | Action |
|---|---|
| `W` | throttle (ramps while held) |
| `Space` | full throttle (instant) |
| `A` / `D` | steer (meaning depends on SAS mode) |
| `S` | cut throttle instantly |
| `G` | cycle SAS mode: DAMP → HOLD → MANUAL |
| `C` | toggle CRT effects (phosphor decay + scanlines) |
| `V` | toggle raycast display (off by default) |
| `R` | reset episode |
| `Esc` | quit |

The keyboard drives a *flight computer* whose output is the same continuous
`[throttle, gimbal]` action an agent would emit — identical env code path
for human and agent. SAS modes: **DAMP** (default) — A/D command a turn
rate, releasing kills the spin but does *not* auto-level; **HOLD** — A/D
command a lean angle, releasing auto-levels (easy mode); **MANUAL** — raw
gimbal, the exact control problem the RL agent gets. The dotted arc is the
ballistic prediction (where you go if you cut thrust now), with predicted
impact point and speed marked.

**Presentation** (all instrument heritage, not HUD decoration): the camera
zooms in smoothly for final approach (vector *Lunar Lander* style); an
ILS-style dashed corridor rises from the pad; drafting-style dimension
lines annotate altitude and lateral offset below 25 m; the panel carries an
Apollo LM cross-pointer (both needles centered = safe to land) beside the
attitude ball.

**Landing model:** the legs absorb a hard *vertical* hit (up to 5 m/s — a
decisive suicide burn is a valid, fuel-efficient landing), but the rocket
**tips over** if it arrives with too much lateral speed, spin, or tilt: a
tip-over energy check compares rotational energy about the downhill leg
against the barrier of rotating the CoM over it (`reward.sticks_upright`).
Kill your drift and stay upright; you don't have to feather it. Gimbal
torque is proportional to thrust — at zero throttle you have **no attitude
control**. That's the game.

## Conventions (load-bearing)

- **World is y-up.** Ground at `y = 0`, gravity along `-y`. pygame renders
  y-down; the flip happens **only** in `play.py:world_to_screen()`.
- **θ** = tilt from vertical, radians, **positive counter-clockwise**;
  `θ = 0` is upright. Nose direction `n = (-sin θ, cos θ)`.
- Gimbal `φ = cmd · φ_max` rotates thrust relative to the body axis; torque
  `τ = -L·T·sin φ`. Positive gimbal command → rocket tips clockwise (right).
- **State** `[x, y, vx, vy, θ, ω, fuel]`, float64. `y` is the **CoM**; the
  rocket is a segment from base `p − (H/2)n` to tip `p + (H/2)n`, and the
  episode terminates on first endpoint–terrain contact (landing criteria
  evaluated at that instant; no resting-contact physics).
- Integration: **semi-implicit Euler** (velocity, then position), fixed
  `dt = 1/60`. Timeout is Gymnasium **truncation**, never termination.
- Observation: 13 × float32, **egocentric only** — pad offset, velocity,
  sin/cos θ, ω, fuel fraction, 5 world-frame downward raycasts. The agent
  never sees absolute level info.

## Layout

```
rocketenv/
  config.py    all constants; per-episode overrides; JSON export for TS port
  physics.py   pure step_dynamics(state, action, cfg) -> state
  terrain.py   Terrain interface + FlatTerrain (polyline drops in later)
  reward.py    isolated reward: shaping + attitude + graded terminal
  env.py       RocketEnv (Gymnasium API)
play.py        pygame harness — ALL rendering lives here
tests/         determinism, physics sanity, API conformance, ray geometry
```

`physics.py` / `reward.py` import neither pygame nor gymnasium — they're the
pure math the TS port and training code reuse.

## Per-episode parameter overrides

Any `Config` field can be overridden for one episode — this is the Phase-3
domain-randomization hook, already wired:

```python
env.reset(seed=0, options={"g": 1.62, "wind_x": 2.0, "thrust_multiplier": 0.7})
```

Reserved no-op axes already in the physics: `drag_coeff` (quadratic),
`wind_gust_x`, `thrust_multiplier`.

Export the exact constants for the future TypeScript port:

```python
from rocketenv import DEFAULT_CONFIG
DEFAULT_CONFIG.dump_json("constants.json")
```

## Tuning

Difficulty knobs live in `rocketenv/config.py`: `twr` (2.3 — raised from the
spec's 1.8 after playtesting showed free-fall recovery felt impossible),
`phi_max` (10° — lowered from 15° to tame attitude twitchiness),
`fuel_0` / `burn_rate`, landing thresholds (`land_*`), spawn envelope
(`spawn_*`), pad width. Control feel (ramp rates, assist gains) lives at the
top of `FlightComputer` in `play.py`. Reward
coefficients are all `Config` fields too — tune while watching the live
reward readout in the harness. A scripted P-controller lands 100/100 seeds
with the defaults (avg impact ~1.2 m/s, ~20% fuel margin), so the task is
feasible; whether it's *fun* is decided on the keyboard.
