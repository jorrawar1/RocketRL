# Crater sample return — browser

Two live modes over procedurally seeded terrain and payloads:

- **POLICY** — runs `final_actor.onnx` locally in the browser and flies the
  TypeScript mission with the trained GRU policy.
- **FLY** — hands the same live mission to keyboard control.

```bash
npm install
npm run dev
```

`npm test` runs geometry and physics-parity suites; `npm run build` typechecks
and bundles.

## Controls

The desktop control bar switches between the PPO policy and human pilot,
restarts the mission, generates a new terrain/payload seed, and opens the live
policy network. The seed in the header is also clickable. `` ` `` remains the
raw-packet shortcut. After each mission result is displayed, the demo
automatically starts a new terrain and payload seed.

In HUMAN PILOT: `W` thrust · `A` steer left · `D` steer right.

## Layout

- `src/render/world.ts` — sky, terrain, pads, vehicle, reticle
- `src/scenario.ts` — deterministic terrain and payload generation
- `src/policy.ts` — ONNX actor, previous-action state, and GRU memory
- `src/render/activations.ts` — live policy activations
- `src/panels.ts` — readouts and mission feed
- `src/sim/` — the physics port (see its README)

## Exporting the trained actor

From the repository root:

```bash
.venv/Scripts/python scripts/export_web_actor.py
```

This writes the actor model and metadata to `web/public/models/` and refreshes
the PyTorch parity vectors in `web/test/data/`. The browser samples the raw
pre-floor state-dependent Gaussian (`mean + std × noise`), squashes it into
the bounded controls, holds each action for four 60 Hz physics frames, renders
the complete sampling dwell, and clears both previous action and GRU state at
episode boundaries and at the loaded-leg handoff. Action noise is seeded from
the visible mission seed, so restarting a seed reproduces the same flight.

`npm test` verifies physics parity, observation parity, 24 recurrent ONNX
decisions against PyTorch, and a complete browser-simulated full mission.
