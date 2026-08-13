# Codebase study guide

Work through this in order. For each module: read the code, read the notes
here, look up the concepts you don't know, then answer the questions **out
loud without looking**. If you can't, you don't own it yet.

The questions are the point. The notes tell you *what* the code does; they
deliberately don't tell you *why* it's built that way — that's what you're
being asked.

**Method:** read a file, close it, explain it aloud for two minutes, recorded.
Play it back. That's simultaneously codebase review and interview rehearsal,
and it exposes the parts you only think you understand.

---

## Module 0 — NumPy (prerequisite)

You cannot explain this codebase without this. Two days, and it pays off
again immediately in the ML work.

**Learn:** arrays vs Python lists · `dtype` and why it matters ·
indexing and slicing · boolean masks · broadcasting · vectorization as a way
of thinking (replacing loops with array expressions) · `np.clip`, `np.interp`,
`np.linspace`, `np.diff`, `np.cumsum`, `np.convolve`, `np.hypot` ·
`np.random.default_rng` and seeded generators · `savez_compressed` / `load`.

**Resources:** NumPy's official *"NumPy fundamentals"* guide, then Nicolas
Rougier's *From Python to NumPy* (free online) — the second one is specifically
about learning to think in array operations instead of loops.

**Questions**
1. What's the difference between `np.array([1,2,3])` and `[1,2,3]`, in memory
   and in what operations they support?
   Both arrays are contigous in memory. The major difference is that a python list stores pointers to python objects,
   while a numpy array is an object with metadata (size, etc) that points to a contiguous array in memory except the
   actual contents are int64, etc instead of objects. Also numpy arrays are of fixed size. 
   They both support normal operations such as indexing, etc. Numpy arrays can also have operations like *2 that
   do elementwise operations, there's also adding, etc which is done in a faster way than a python list with a for loop.
   We can also resize, etc numpy arrays although this is just changing the metadata.
2. What does `a[mask]` return when `mask` is a boolean array, and what shape is it?
a[mask] will return a view of the original array that only includes the corresponding elements of the array that are true in
the boolean array. The shape of the array will be 1 dimensional, but you can use np.ma.array to preserve it. 
3. Why does `float32` vs `float64` matter for a physics simulation? For a
   neural network?
   float64 and float32 are just different representations of numbers, (4 bits vs 8 bits). Float64 will take up more space
   in memory but also provide more accuracy, this can be a tradeoff when making neural networks (I believe this is basically
   what quantization is about). 
4. What does broadcasting do with shapes `(5,3)` and `(3,)`? With `(5,3)` and `(5,)`?
   For the first we'd basically have 5 rows and 3 columns, then a 1D array with 3 elements. 
   If adding them we compare rightmost, so (5,3) and (   3), then 3 elements match so each element is added
   columnwise. For the second one its the same logic except 5 is more than the number of columns in the first matrix
   so the operation would fail. 


---

## Module 1 — `rocketenv/config.py`

Every constant in the system, in one frozen dataclass.

**The header comment block** states the coordinate conventions: y-up world,
`theta` measured from vertical and positive counter-clockwise, nose direction
`n = (-sin θ, cos θ)`, `y` is the centre of mass. Read it twice — every sign in
the codebase depends on it.

```python
from __future__ import annotations
```
Makes all type annotations lazily evaluated strings. It's why `dict | None`
works on older Python versions. Ok so this basically lets us use a type before we've actually 
written it in the code since python evaluates line by line unlike say c++. 

```python
@dataclass(frozen=True)
class Config:
    dt: float = 1.0 / 60.0
```
`@dataclass` generates `__init__`, `__repr__`, and `__eq__` from the annotated
fields. `frozen=True` makes instances immutable — assigning to a field raises.

**Field groups:** integration (`dt`), world (`g`, `world_w/h`), rocket (`m`,
`twr`, `H`, `phi_max`, `fuel_0`, `burn_rate`), reserved randomization axes
(`drag_coeff`, `wind_x`, `wind_gust_x`, `thrust_multiplier` — all no-ops at
their defaults), pad, terrain generation, spawn envelope, episode length,
observation normalization constants, reward coefficients.

```python
@property
def T_max(self) -> float:
    return self.twr * self.m * self.g
```
Three derived quantities — max thrust, moment of inertia `I = mH²/12`, moment
arm `L = H/2` — computed rather than stored. `@property` makes them look like
attributes at the call site.

```python
def with_overrides(self, params: dict | None) -> "Config":
    unknown = set(params) - {f.name for f in dataclasses.fields(self)}
    if unknown:
        raise KeyError(...)
    return dataclasses.replace(self, **params)
```
Set difference against the field names catches typos. `dataclasses.replace`
builds a *new* instance with some fields changed — necessary because the class
is frozen.

`to_dict` uses `dataclasses.asdict` and then adds the three derived values.
`dump_json` writes it with `sort_keys=True`.

**Concepts to look up:** dataclasses (Python docs) · `@property` ·
set operations · `**kwargs` unpacking · why immutability is useful.

**Questions**
1. Why is `Config` frozen? What could go wrong if it weren't, given that
   `reset()` hands a config to the physics every step?
   A frozen dataclass basically just means once that object is created it's immutable,
   similar to a const in c++. If it weren't frozen the individual config object could change
   which would create poisioned data and inaccuracies.
2. Why are `T_max`, `I`, and `L` properties instead of fields with defaults?
   They are calculated from the instrinsic fields so they depend on them and need
   to be calculated after the other fields are already chosen. 
3. `with_overrides` returns a new object rather than mutating. What does that
   buy you when you run 64 environments in parallel?
   It allows you to do them in parallel in the first place, if you were modifying the same original
   object all of the 64 runs would be referring to the same config and using the same setup. With overrides
   you can have a seperate object with different settings for each run. 
4. What is `dump_json` for? Who is the consumer, and what problem does it prevent?
   It turns all of the config parameters including derived into a json file. This is mainly for the typescript port later,
   it gets rid of any innacurracies such as floating point operations between the two languages. 
5. `drag_coeff`, `wind_gust_x`, and `thrust_multiplier` are all defaults that do
   nothing. Why are they in the code at all?
   Right now we don't have these effects in the environmnent, but we could add them later. For example test if the
   network can still land the rocket with randomized wind gust. 
6. Where does `I = mH²/12` come from?
   Thats the moment of inertia for the rocket modeled as a rod, important for the physics.

---

## Module 2 — `rocketenv/physics.py`

56 lines. The entire simulation. Know this cold.

```python
X, Y, VX, VY, THETA, OMEGA, FUEL = range(7)
STATE_DIM = 7
```
Named indices into the state array. `state[VX]` instead of `state[2]`.

```python
def nose_direction(theta):
    return (-math.sin(theta), math.cos(theta))
```
Body-up unit vector. At `theta = 0` this is `(0, 1)` — straight up.

```python
def body_endpoints(state, cfg):
    nx, ny = nose_direction(state[THETA])
    half = cfg.L
    pos = state[[X, Y]]
    offset = np.array([nx * half, ny * half])
    return pos - offset, pos + offset
```
The rocket as a line segment: base is `L` behind the CoM along the nose axis,
tip is `L` ahead. `state[[X, Y]]` is fancy indexing — a list of indices returns
a new array.

### `step_dynamics` — line by line

```python
throttle = min(max(float(action[0]), 0.0), 1.0)
gimbal_cmd = min(max(float(action[1]), -1.0), 1.0)
phi = gimbal_cmd * cfg.phi_max
```
Clip into the valid range, then map the normalized gimbal command to a physical
angle in radians.

```python
thrust = throttle * cfg.T_max * cfg.thrust_multiplier if fuel > 0.0 else 0.0
```
No fuel, no thrust.

```python
fx = thrust * -math.sin(theta + phi) + cfg.wind_x + cfg.wind_gust_x
fy = thrust * math.cos(theta + phi) - cfg.m * cfg.g
```
The thrust direction is the nose direction *rotated by the gimbal angle*.
Gravity is a constant force on `fy`.

```python
if cfg.drag_coeff > 0.0:
    speed = math.hypot(vx, vy)
    fx -= cfg.drag_coeff * speed * vx
    fy -= cfg.drag_coeff * speed * vy
```
Quadratic drag, opposing velocity. Off by default.

```python
tau = -cfg.L * thrust * math.sin(phi)
```
Torque from thrust applied at the engine, offset `L` from the CoM.

```python
vx += (fx / cfg.m) * cfg.dt
vy += (fy / cfg.m) * cfg.dt
x  += vx * cfg.dt
y  += vy * cfg.dt
omega += (tau / cfg.I) * cfg.dt
theta += omega * cfg.dt
```
Velocity updated first, *then* position using the new velocity. This ordering
has a name and a reason.

```python
fuel = max(0.0, fuel - cfg.burn_rate * throttle * cfg.dt)
return np.array([...], dtype=np.float64)
```
Returns a new array. Nothing passed in is modified.

**Concepts to look up:** semi-implicit (symplectic) Euler vs explicit Euler —
Gaffer On Games, *"Integration Basics"* · torque as `r × F` in 2D · moment of
inertia · pure functions and why they're testable.

**Questions**
1. Derive `tau = -L·T·sin(φ)`. Where does the sign come from, and what happens
   physically when `gimbal_cmd` is positive?
   Skipping this, I don't think the details of the physics are too important, could have just looked up formulas as well.
2. Why velocity before position? What visibly breaks in a simulation that does
   it the other way?
   The other way could be unstable, for example you have a rocket that lands with a high velocity.
   At the point it touches velocity is still high so the rocket goes into the ground since thats updated before
   position and it could lead to a feedback.
3. Why is `dt` fixed rather than measured from the frame time?
   Then if there's a different fps it could make the actual speed of everything different.
4. This function takes `state` and returns a *new* state instead of mutating.
   Name three things that depend on that.
   Return later
5. At `theta = 0` and full throttle, what is `vy` after one step? Derive it.
6. Why `float64` here when the observation is `float32`?

7. What is the moment arm `L`, and why is it `H/2` rather than `H`?

---

## Module 3 — `rocketenv/terrain.py`

```python
class Terrain(ABC):
    @abstractmethod
    def height_at(self, x: float) -> float: ...
    @abstractmethod
    def ray_distance(self, ox, oy, dx, dy, max_range) -> float: ...
```
An abstract base class: two methods, no implementation. Attempting to
instantiate `Terrain` directly raises. This is the entire interface between the
environment and the ground. Alright so basically like a virtual method in c++. 

**`FlatTerrain`** — `height_at` returns a constant. `ray_distance` solves ray
versus a horizontal line in closed form: if `dy >= 0` the ray points up or is
parallel, otherwise `t = (height - oy) / dy`, clamped. I don't really get what the whole ray thing is fully. 

### `PolylineTerrain`

```python
self._px = self.xs[:-1]
self._py = self.ys[:-1]
self._sx = np.diff(self.xs)
self._sy = np.diff(self.ys)
```
Precomputed segment starts and direction vectors. `np.diff` gives consecutive
differences, so `_sx[i], _sy[i]` is the vector from vertex `i` to `i+1`.


```python
def height_at(self, x):
    return float(np.interp(x, self.xs, self.ys))
```
Linear interpolation between vertices.

```python
denom = dx * self._sy - dy * self._sx
qpx = self._px - ox
qpy = self._py - oy
t = (qpx * self._sy - qpy * self._sx) / denom
u = (qpx * dy - qpy * dx) / denom
valid = (np.abs(denom) > 1e-12) & (t > 1e-9) & (u >= 0.0) & (u <= 1.0)
return float(min(t[valid].min(), max_range))
```
Ray–segment intersection, solved for **all segments at once**. Setting
`o + t·d = p + u·s` and taking 2D cross products gives `t` and `u`. `denom` is
the cross product of the two directions — zero when parallel. The `valid` mask
encodes: not parallel, intersection ahead of the origin, and the hit lands
within the segment rather than on its infinite extension. `np.errstate`
suppresses the divide-by-zero warnings from parallel segments, which the mask
then discards.

### `generate_terrain`

```python
h = np.cumsum(rng.normal(0.0, 1.0, n))
h = np.convolve(np.pad(h, 2, mode="edge"), np.ones(5) / 5.0, mode="valid")
h -= h.min()
h *= cfg.terrain_amp * rng.uniform(0.55, 1.0) / span
```
A random walk (cumulative sum of Gaussian steps), box-smoothed by convolution
with a length-5 averaging kernel, shifted so the minimum is zero, then rescaled
to a random fraction of the maximum amplitude.

```python
flatten_r = cfg.pad_half_w + cfg.terrain_res + 1.0
h[np.abs(xs - pad_x) <= flatten_r] = pad_y
```
Boolean-mask assignment flattens every vertex within `flatten_r` of the pad.

**Concepts to look up:** abstract base classes (`abc` module) · 2D cross product
and the parametric ray–segment intersection · convolution as a smoothing filter ·
`np.errstate` · boolean mask assignment.

**Questions**
1. Why is `Terrain` an ABC rather than just a convention? What does it buy you?
   It allows us to have Terrain as a parent class and basically fill in what Flat and Poly
   would have in common. We don't have to rewrite certain things, but actually more importantly
   it lets us call operations / interface with them basically the same. So a function could require terrain
   but we can pass either flat or poly and it works, if we had just two independant classes this wouldn't work
   and would add a lot of complexity to the codebase. 
2. `PolylineTerrain` was added long after `env.py` was written, and `env.py`
   didn't change at all. What property made that possible?
   As above, we have a parent class caleed terrain, so we can treat them the same (polymorphism I believe
   is the name of this phenomenon). 
3. Walk through the `valid` mask term by term. What does each condition exclude,
   and what would break if you dropped it?

4. Why is `ray_distance` vectorized across segments instead of looping? Estimate
   how many times per episode it's called.
5. Why does `generate_terrain` return `pad_x` instead of storing it on the
   terrain object?
6. Why is `flatten_r` wider than `pad_half_w`?
7. What does convolving with `np.ones(5)/5` do to a random walk, and why is the
   raw walk unsuitable as terrain?
   I'll need a walkthrough on the rest of the questions, I'm kind of confused on the point of the ray in the first place. 


---

## Module 4 — `rocketenv/reward.py`

```python
def potential(state, cfg, pad_y):
    d = math.hypot(state[X] - cfg.pad_x, state[Y] - pad_y)
    return -cfg.shaping_k * d
```
Φ(s), negative distance to the pad scaled by `k`.

```python
r = -cfg.step_penalty
r += cfg.gamma * potential(state, ...) - potential(prev_state, ...)
w = min(max((cfg.att_alt - state[Y]) / cfg.att_fade, 0.0), 1.0)
r -= w * (cfg.att_c_theta * abs(state[THETA]) + cfg.att_c_omega * abs(state[OMEGA]))
```
Three terms: a constant time penalty, potential-based shaping as a difference,
and an attitude penalty whose weight `w` ramps linearly from 0 to 1 as altitude
drops through a band rather than switching on at a threshold.

### `sticks_upright` — the tip-over model

```python
b, arm = cfg.leg_half_w, cfg.L
if theta >= math.atan2(b, arm):
    return False
r_pivot = math.hypot(arm, b)
com_h = arm * math.cos(theta) + b * math.sin(theta)
barrier = cfg.m * cfg.g * (r_pivot - com_h)
i_pivot = cfg.I + cfg.m * r_pivot * r_pivot
ang_mom = cfg.I * abs(state[OMEGA]) + cfg.m * arm * abs(state[VX])
tip_energy = ang_mom * ang_mom / (2.0 * i_pivot)
return tip_energy < barrier
```
Treats touchdown as an inelastic pivot about a landing leg. `atan2(b, arm)` is
the tilt at which the CoM is already past the pivot. `barrier` is the potential
energy needed to lift the CoM from its current height to directly above the
pivot. `i_pivot` uses the parallel-axis theorem. `ang_mom` combines body spin
and the angular momentum of the translating CoM about the pivot. Tips if the
rotational kinetic energy clears the barrier.

```python
on_pad = abs(state[X] - cfg.pad_x) < cfg.pad_half_w
survivable = abs(state[VY]) <= cfg.land_vy_max and abs(state[VX]) <= cfg.land_vx_max
if on_pad and survivable and sticks_upright(state, cfg):
    return cfg.reward_land + cfg.reward_fuel_bonus * fuel_frac, TOUCHDOWN
gentle = clamp(1 - impact / cfg.crash_speed_ref)
near   = clamp(1 - |x - pad_x| / cfg.partial_dist_ref)
r = cfg.partial_credit * gentle * near + cfg.reward_crash * (1.0 - gentle)
outcome = TIPPED if (on_pad and survivable) else CRASH
```
Three conditions for a landing. Otherwise graded partial credit: gentleness
times proximity earns credit, impact speed drives the penalty, both smooth.

**Concepts to look up:** potential-based reward shaping — Ng, Harada & Russell
(1999), *"Policy Invariance Under Reward Transformations"* · parallel-axis
theorem · reward hacking / farming.

**Questions**
1. Why is the shaping a *difference* of potentials rather than a bonus for being
   close to the pad? What does a difference telescope to over an episode?
   It gives feedback at every step instead of sparse rewards. For example, just being close to the pad
   wouldn't activate until the last 20% of the episode so the agent would struggle to train past the initial period. 
   The difference telescopes to 0. Also the agent could hack it by hovering close to the pad. 
2. `gamma` defaults to 1.0 with a comment warning against lowering it. Explain
   the failure mode concretely.
   Gamma is how much future rewards are discounted. If its bigger than 1, the later states
   will be weighed a lot more (due to exponent form) rather than the current state. If its less than 1, the 
   final state will approach 0, which could be not good since the final goal is obviously to land the rocket so
   we need to consider what the final state will look like. 
3. Why does the attitude penalty fade in over a band instead of switching on
   below a threshold?
   It basically penalizes the rocket being sideways or in a bad posititon/angle. It's more important
   at the lower altitudes since its harder to recover it then. If it were just switched on the agent could
   just be completely sideways at one position, then it recieves a huge penalty immediately which could lead
   to overcorrection, etc.
4. Derive `barrier`. Why is it `r_pivot - com_h`?
5. Why does `i_pivot` add `m·r_pivot²` to `I`?
6. Why is `ang_mom` a sum of two terms, and why worst-case signs?
7. A policy could learn to hover just above the pad forever. What in the reward
   prevents that, and how strong is it?
   The difference of potentials and the time penalty. If its in the same position the difference of potentials will be
   0 and the time penalty will steadily cut in so it'll recieve negative reward. 
8. Why does the terminal reward include a fuel bonus? What behaviour is that
   trying to produce?
   Its trying to reward the agent for reaching the pad but also not waste fuel.

---

## Module 5 — `rocketenv/env.py`

```python
class RocketEnv(gym.Env):
    metadata = {"render_modes": []}
```
Subclasses Gymnasium's base env. Empty render modes — nothing here draws.

```python
obs_dim = 8 + self.base_config.n_rays
self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
```
Spaces declare shape and dtype. Note the two config objects: `base_config` never
changes, `cfg` is the active per-episode one.

```python
super().reset(seed=seed)
self.cfg = self.base_config.with_overrides(options)
cfg, rng = self.cfg, self.np_random
```
`super().reset(seed=...)` seeds `self.np_random`, a generator Gymnasium provides.
Then seven uniform draws build the spawn state.

```python
action = np.clip(np.asarray(action, dtype=np.float64), [0.0, -1.0], [1.0, 1.0])
prev_state = self.state
self.state = step_dynamics(prev_state, action, cfg)
self.steps += 1
reward = rw.step_reward(prev_state, self.state, cfg, pad_y)
```
Both `prev_state` and the new state are needed because shaping is a difference.

```python
if self._out_of_bounds():
    terminated = True; ...
elif self._contact():
    terminated = True; ...
elif self.steps >= cfg.max_steps:
    truncated = True
return self._observe(), reward, terminated, truncated, self._info(outcome)
```
Ordering matters — out of bounds is checked before contact. Timeout sets
`truncated`, never `terminated`.

```python
def _contact(self):
    base, tip = body_endpoints(self.state, self.cfg)
    return base[1] <= self.terrain.height_at(base[0]) or tip[1] <= self.terrain.height_at(tip[0])
```
Either end of the segment touching the ground ends the episode.

```python
obs[0] = (cfg.pad_x - s[X]) / cfg.world_w
obs[1] = (pad_y - s[Y]) / cfg.world_h
obs[2] = s[VX] / cfg.v_ref
obs[4] = math.sin(s[THETA]);  obs[5] = math.cos(s[THETA])
obs[6] = s[OMEGA] / cfg.omega_ref
obs[7] = s[FUEL] / cfg.fuel_0
obs[8:] = self.ray_distances() / cfg.ray_max_range
return obs.astype(np.float32)
```
Built in `float64`, cast to `float32` at the boundary. Every component is
normalized and relative.

`_info` returns `{"state": copy, "steps": int}` plus `"outcome"` only when the
episode terminated.

**Concepts to look up:** Gymnasium API (`reset`/`step` signatures, `Box` spaces)
· their *"handling time limits"* page · why angles are encoded as sin/cos in ML ·
input normalization.

**Questions**
1. What exactly is the difference between `terminated` and `truncated`, and what
   does an RL algorithm do differently with each? Why would conflating them bias
   learning?
2. Why `sin θ, cos θ` instead of `θ`? What specifically breaks with raw angles?
3. Why is the observation egocentric and normalized? What would go wrong if you
   fed absolute pad coordinates?
4. Why is `info["state"]` a `.copy()`?
5. Why does the env clip incoming actions rather than trusting the caller?
6. Out of bounds is checked before contact. Construct a state where the order
   changes the outcome.
7. `base_config` vs `cfg` — why two, and what breaks with only one?
8. The observation contains altitude *relative to the pad*. On terrain where the
   pad sits in a valley, what information is unrecoverable? Why does that matter
   for behaviour cloning specifically?

---

## Module 6 — `rocketenv/scripted.py`

The classical PD controller — the expert for behaviour cloning and the baseline
to beat.

```python
if abs(dx) > ARRIVAL_DX:
    vx_des = clip(0.35 * dx, -5, 5)
    vy_des = clip((CRUISE_ALT - y) * 0.4, -3, 3)
else:
    vx_des = clip(0.3 * dx, -2, 2)
    vy_des = -min(6.0, 0.28 * clearance + 0.8)
```
Two phases: traverse at altitude until nearly above the pad, then descend on a
schedule proportional to clearance.

```python
ax_des = clip(0.8 * (vx_des - vx), -3, 3)
theta_des = clip(-ax_des / cfg.g * 0.9, -0.35, 0.35)
if clearance < FLARE_ALT:
    theta_des *= max(clearance, 0.0) / FLARE_ALT
gimbal = clip(-(6.0 * (theta_des - theta) - 3.0 * omega), -1, 1)
```
Velocity error becomes a desired tilt; tilt error plus a damping term on `omega`
becomes a gimbal command. The `6.0` and `3.0` are proportional and derivative
gains.

```python
tilt_loss = max(math.cos(theta), 0.3)
hover = 1.0 / (cfg.twr * cfg.thrust_multiplier * tilt_loss)
throttle = clip(hover + 0.35 * (vy_des - vy), 0.0, 1.0)
```
Feed-forward hover throttle plus proportional correction on vertical velocity
error.

```python
if noise_std > 0.0:
    action = action + rng.normal(0.0, noise_std, size=2)
    action = np.clip(action, [0.0, -1.0], [1.0, 1.0])
```
Optional perturbation — used during data collection.

**Concepts to look up:** PD control, proportional and derivative terms ·
feed-forward vs feedback control · cascaded control loops.

**Questions**
1. Why is the hover term `1 / (twr · cos θ)`? Derive it.
2. What does the derivative term on `omega` do? What happens if you remove it?
3. Why does `theta_des` fade to zero below `FLARE_ALT`?
4. This controller reads `env.state`, not the observation. Which quantities does
   it use that a policy would not have?
5. Why is there a noise option, and why is the *clean* action recorded as the
   label while the *noisy* one is executed?

---

## Module 7 — `learning/generate_dataset.py`

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```
Running a script inside `learning/` puts that directory on the import path, not
the repo root. This adds the root so `import rocketenv` resolves.

```python
assert n_episodes < EVAL_SEED_START
```
Training episodes use seeds `0..n-1`; evaluation starts at 10,000.

```python
ep_obs, ep_act = [], []
while True:
    action = scripted_action(env.state, env.cfg, env.terrain)
    ep_obs.append(obs); ep_act.append(action)
    executed = action if noise_std == 0 else clip(action + rng.normal(...))
    obs, _reward, terminated, truncated, info = env.step(executed)
    if terminated or truncated: break
outcome = info.get("outcome", "TIMEOUT")
if keep_failures or outcome == TOUCHDOWN:
    obs_all.extend(ep_obs); act_all.extend(ep_act); ep_all.extend([ep] * len(ep_obs))
```
Per-episode buffers so a failed episode can be discarded whole. `obs` at the top
of the loop is the observation at time `t`; `step` overwrites it afterward.

```python
return (np.asarray(obs_all, dtype=np.float32),
        np.asarray(act_all, dtype=np.float32),
        np.asarray(ep_all, dtype=np.int32), outcomes)
```
Both arrays forced to `float32`.

`main()` uses `argparse` to expose episodes, noise, keep-failures, terrain mode,
seed, and output path as flags, then saves with `np.savez_compressed` and prints
summary statistics.

**Concepts to look up:** `argparse` · `sys.path` and Python's import resolution ·
`pathlib` · behaviour cloning · distribution shift / covariate shift and DAgger.

**Questions**
1. Why record the observation and not the state?
2. Why is the clean action the label when a noisy one was executed? What problem
   does that address?
3. Why buffer per episode instead of appending directly?
4. Why does `info.get("outcome", "TIMEOUT")` need a default?
5. Why are training and evaluation seeds kept disjoint?
6. Why is `episode` saved alongside `obs` and `act`? What would you be unable to
   do without it?
7. Both arrays are cast to `float32`. What error appears later if you skip that?

---

## Module 8 — `play.py` (architecture only)

750 lines of presentation. **Skim for structure, don't study.** Know the stack
and the shape.

**Stack:** pygame-ce for windowing, input, and 2D drawing. No engine, no
scene graph — an immediate-mode loop that clears and redraws every frame.

**Structure:**

| piece | role |
|---|---|
| `Theme` | frozen dataclass of five colours |
| `Camera` | follow-zoom; `world_to_screen` is the only place y flips |
| `FlightComputer` | keyboard → continuous `[throttle, gimbal]`; three SAS modes |
| `predict_ballistic` | re-runs `step_dynamics` at zero throttle for the prediction arc |
| `Phosphor` | an alpha surface that decays each frame — CRT trail effect |
| `Debris`, `StripChart`, `EventLog` | particles, scrolling telemetry, log lines |
| `Console` | every draw call: terrain, pad, rocket, rays, panel, stamps |
| `load_map` | flat or seeded generated terrain |
| `main` | the loop: poll events → produce action → `env.step` → draw → `clock.tick(60)` |

**The one architectural fact that matters:** the harness produces an ordinary
action and calls `env.step`. Keyboard and policy go through the identical path,
and nothing in `rocketenv/` imports pygame. That's what makes the env usable
headless, in parallel, and portable to TypeScript.

**Questions**
1. Why does no rendering code live in `rocketenv/`? Name two things it would
   break.
2. `world_to_screen` is the only y-flip in the project. Why concentrate it?
3. `predict_ballistic` calls the same `step_dynamics` as the env. Why is that
   possible, and why is the prediction exact rather than approximate?
4. What does the fixed-timestep loop guarantee, and how does it differ from
   stepping physics by measured frame time?

---

## Module 9 — `tests/`

Read these as documentation — each asserts one fact about how the system behaves.

- `test_physics.py` — purity, determinism, free-fall at −g, thrust cancelling
  gravity, zero fuel, torque signs, action clipping
- `test_env.py` — Gymnasium `check_env` conformance, seeding reproducibility,
  full-episode determinism, ray geometry in closed form, option overrides,
  timeout as truncation
- `test_reward.py` — the landing model: gentle landings, hard vertical landings
  that stick, tip-overs, off-pad partial credit
- `test_terrain.py` — interpolation, exact ray hits, flat-equivalence,
  generation determinism, pad flatness
- `test_scripted.py` — the expert actually lands, on flat and on generated maps,
  with and without noise

**Questions**
1. Which test catches a flipped sign in the torque expression?
2. Which catches swapping the velocity and position updates?
3. `test_terrain.py` asserts the polyline matches `FlatTerrain` exactly on flat
   input. Why is that a useful test?
4. Determinism is tested at both the physics and env level. Why both?

---

## Final exam

Answer these cold, out loud, before you call the codebase yours.

1. Why is `step_dynamics` a pure function? Name three consequences.
2. Why is a timeout `truncated` rather than `terminated`?
3. Why does the observation contain `sin θ, cos θ` instead of `θ`?
4. Why is the observation egocentric?
5. Why is the shaping reward a difference of potentials?
6. Why semi-implicit Euler?
7. Why does `config` dump to JSON?
8. Why `float64` physics and `float32` observations?
9. Explain the tip-over model — the physical argument, not the code.
10. What would you change about this design, and why?

Question 10 is the one interviewers actually ask.
