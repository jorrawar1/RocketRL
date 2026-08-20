# sim/ — TypeScript physics port

Ports of the Python simulator, so the page can be flown live rather than only
replayed:

- `config.ts` — constants from `rocketenv/config.py` + `sample_return/config.py`
- `vehicle.ts` — payload-aware mass, COM offset, parallel-axis inertia
- `physics.ts` — `stepDynamics`, a line-for-line port of `sample_return/physics.py`
- `stability.ts` — the leg-pivot tip-over model from `sample_return/reward.py`
- `mission.ts` — the phase machine from `sample_return/env.py`, minus rewards

## Parity

`test/parity.test.ts` drives `stepDynamics` with every recorded airborne
state/action pair from `artifacts/mission_42.json` and asserts the result
matches the next recorded state. Both sides are IEEE-754 float64 running the
same operation order, so the tolerance is 1e-9.

`artifacts/` is gitignored; regenerate it with
`python scripts/export_sample_return_fixture.py` or the suite skips itself.
The copy under `public/data/` is rounded to four decimals and is no use here.
