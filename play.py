"""Keyboard flight harness for RocketEnv.

Controls
  W           throttle (ramped while held)
  SPACE       full throttle (instant while held)
  A / D       steer: commanded tilt (STAB AUG) or raw gimbal (MANUAL)
  S           cut throttle instantly
  G           toggle STAB AUG (fly-by-wire attitude hold) / MANUAL
  C           toggle CRT effects (phosphor + scanlines)
  V           toggle raycast display
  R           reset episode
  ESC         quit

The keyboard drives a *flight computer* whose output is the same continuous
[throttle, gimbal] action an agent would emit — the env code path is
identical for human and agent.  STAB AUG closes an attitude loop for the
human (real rockets are flown this way); MANUAL is the raw problem the RL
agent will face.

Rendering: the world is y-up; the y-flip to pygame's y-down screen happens
only in world_to_screen().  No rendering code exists inside rocketenv.
"""

from __future__ import annotations

import ctypes
import math
import random
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np
import pygame

from rocketenv import RocketEnv
from rocketenv.physics import (
    FUEL, OMEGA, THETA, VX, VY, X, Y, body_endpoints, step_dynamics,
)
from rocketenv.reward import TOUCHDOWN

SCALE = 8                 # px per meter
VIEW_W, VIEW_H = 800, 800  # world viewport (100 m x 100 m)
PANEL_W = 420
WIN_W, WIN_H = VIEW_W + PANEL_W, VIEW_H
END_FREEZE_S = 2.5
CHART_SECONDS = 12.0      # strip-chart history window
PREDICT_S = 4.0           # ballistic prediction horizon


@dataclass(frozen=True)
class Theme:
    """Single dark instrument palette. Swapping colors later is trivial;
    the aesthetic investment is in the phosphor/CRT presentation, not hue."""
    field: tuple = (8, 11, 16)        # near-black ink
    grid: tuple = (20, 26, 34)        # graticule
    struct: tuple = (58, 70, 84)      # terrain / chrome / labels
    signal: tuple = (255, 176, 0)     # instrument amber: the system's voice
    fault: tuple = (255, 59, 48)      # genuine faults only


THEME = Theme()


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def world_to_screen(x, y):
    """The ONLY place y flips."""
    return int(x * SCALE), int(VIEW_H - y * SCALE)


class FlightComputer:
    """Keyboard -> continuous action.

    STAB AUG (default): A/D command a tilt angle; a PD loop flies the gimbal
    and auto-levels when released.  MANUAL: A/D slew the gimbal directly —
    the raw control problem the agent gets.  Either way the output is a
    plain action inside the env's action space.
    """

    THROTTLE_UP = 4.0     # /s toward 1 while W held
    THROTTLE_DOWN = 5.0   # /s toward 0 when released
    GIMBAL_SLEW = 3.0     # manual: /s toward +/-1 while A/D held
    GIMBAL_SPRING = 6.0   # manual: /s back to center
    TILT_CMD = 0.30       # rad, commanded tilt at full A/D in STAB AUG
    KP, KD = 6.0, 3.0     # attitude loop gains

    def __init__(self):
        self.assist = True
        self.throttle = 0.0
        self.gimbal = 0.0

    def reset(self):
        self.throttle = 0.0
        self.gimbal = 0.0

    def update(self, keys, state, dt):
        if keys[pygame.K_SPACE]:
            self.throttle = 1.0
        elif keys[pygame.K_s]:
            self.throttle = 0.0
        elif keys[pygame.K_w]:
            self.throttle = min(1.0, self.throttle + self.THROTTLE_UP * dt)
        else:
            self.throttle = max(0.0, self.throttle - self.THROTTLE_DOWN * dt)

        steer = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
        if self.assist:
            # commanded tilt: D -> lean right (theta < 0) -> translate +x
            theta_des = -steer * self.TILT_CMD
            err = theta_des - state[THETA]
            # positive gimbal -> negative torque, hence the leading minus
            self.gimbal = float(np.clip(
                -(self.KP * err - self.KD * state[OMEGA]), -1.0, 1.0))
        else:
            rate = self.GIMBAL_SLEW if steer != 0.0 else self.GIMBAL_SPRING
            target = steer
            if self.gimbal < target:
                self.gimbal = min(target, self.gimbal + rate * dt)
            elif self.gimbal > target:
                self.gimbal = max(target, self.gimbal - rate * dt)

    def action(self):
        return np.array([self.throttle, self.gimbal], dtype=np.float32)


def predict_ballistic(env):
    """Coast (zero-throttle) trajectory from the current state, reusing the
    pure physics.  Returns (sample points, impact (x, y, speed) or None)."""
    s = env.state.copy()
    zero = np.zeros(2)
    pts, impact = [], None
    for i in range(int(PREDICT_S * 60)):
        s = step_dynamics(s, zero, env.cfg)
        if i % 6 == 0:
            pts.append((s[X], s[Y]))
        base, tip = body_endpoints(s, env.cfg)
        if (base[1] <= env.terrain.height_at(base[0])
                or tip[1] <= env.terrain.height_at(tip[0])):
            impact = (s[X], s[Y], math.hypot(s[VX], s[VY]))
            break
    return pts, impact


class Phosphor:
    """Persistent decay layer: anything drawn here smears and fades like a
    scope trace.  Glow is a scalpel — only signal elements land on it."""

    def __init__(self, size):
        self.surf = pygame.Surface(size, pygame.SRCALPHA)

    def clear(self):
        self.surf.fill((0, 0, 0, 0))

    def decay(self):
        self.surf.fill((0, 0, 0, 10), special_flags=pygame.BLEND_RGBA_SUB)

    def line(self, a, b, color, alpha, width=1):
        pygame.draw.line(self.surf, (*color, alpha), a, b, width)

    def polygon(self, pts, color, alpha):
        pygame.draw.polygon(self.surf, (*color, alpha), pts, 1)


def make_scanlines(w, h):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    for yy in range(0, h, 3):
        pygame.draw.line(s, (0, 0, 0, 26), (0, yy), (w, yy))
    return s


class StripChart:
    """Scrolling instrument trace: fixed y-range, last CHART_SECONDS of data."""

    def __init__(self, label, lo, hi, zero_line=False):
        self.label = label
        self.lo, self.hi = lo, hi
        self.zero_line = zero_line
        self.data = deque(maxlen=int(CHART_SECONDS * 60))

    def reset(self):
        self.data.clear()

    def push(self, v):
        self.data.append(v)

    def draw(self, screen, x, y, w, h):
        pygame.draw.rect(screen, THEME.grid, (x, y, w, h), 1)
        if self.zero_line and self.lo < 0 < self.hi:
            zy = y + h - int((0 - self.lo) / (self.hi - self.lo) * h)
            pygame.draw.line(screen, THEME.grid, (x + 1, zy), (x + w - 1, zy))
        n = len(self.data)
        if n < 2:
            return
        maxn = self.data.maxlen
        pts = []
        for i, v in enumerate(self.data):
            px = x + w - 1 - int((n - 1 - i) * (w - 2) / (maxn - 1))
            frac = (min(max(v, self.lo), self.hi) - self.lo) / (self.hi - self.lo)
            pts.append((px, y + h - 1 - int(frac * (h - 2))))
        pygame.draw.lines(screen, THEME.signal, False, pts, 1)


class EventLog:
    """Terse system-log lines with T+ stamps."""

    def __init__(self, maxlines=5):
        self.lines = deque(maxlen=maxlines)

    def reset(self):
        self.lines.clear()

    def post(self, t, text, fault=False):
        self.lines.append((t, text, fault))


class Console:
    """All drawing. Reads env state; never writes it."""

    def __init__(self, screen):
        self.screen = screen
        names = "consolas, cascadia mono, jetbrains mono, courier new"
        self.f_num = pygame.font.SysFont(names, 16)
        self.f_lab = pygame.font.SysFont(names, 13)
        self.f_stamp = pygame.font.SysFont(names, 20, bold=True)

    def text(self, font, s, color, x, y, right_align=False):
        surf = font.render(s, True, color)
        if right_align:
            x -= surf.get_width()
        self.screen.blit(surf, (x, y))
        return surf.get_height()

    # ------------------------------------------------------------ world view
    def draw_graticule(self, cfg):
        for gx in range(0, int(cfg.world_w) + 1, 10):
            sx, _ = world_to_screen(gx, 0)
            pygame.draw.line(self.screen, THEME.grid, (sx, 0), (sx, VIEW_H))
        for gy in range(0, int(cfg.world_h) + 1, 10):
            _, sy = world_to_screen(0, gy)
            pygame.draw.line(self.screen, THEME.grid, (0, sy), (VIEW_W, sy))

    def draw_terrain(self, env):
        cfg = env.cfg
        pts = [world_to_screen(x, env.terrain.height_at(x))
               for x in range(0, int(cfg.world_w) + 1, 2)]
        pygame.draw.lines(self.screen, THEME.struct, False, pts, 1)

    def draw_pad(self, env, phosphor):
        cfg = env.cfg
        pad_y = env.terrain.height_at(cfg.pad_x)
        lx, ly = world_to_screen(cfg.pad_x - cfg.pad_half_w, pad_y)
        rx, _ = world_to_screen(cfg.pad_x + cfg.pad_half_w, pad_y)
        cx, _ = world_to_screen(cfg.pad_x, pad_y)
        arm = 8
        for ex, sgn in ((lx, 1), (rx, -1)):
            pygame.draw.line(self.screen, THEME.signal, (ex, ly), (ex, ly - arm), 1)
            pygame.draw.line(self.screen, THEME.signal, (ex, ly),
                             (ex + sgn * arm, ly), 1)
            if phosphor:  # steady soft glow on the target marker
                phosphor.line((ex, ly), (ex, ly - arm), THEME.signal, 26, 3)
                phosphor.line((ex, ly), (ex + sgn * arm, ly), THEME.signal, 26, 3)
        pygame.draw.line(self.screen, THEME.signal, (cx - 4, ly), (cx + 4, ly), 1)
        pygame.draw.line(self.screen, THEME.signal, (cx, ly - 4), (cx, ly + 4), 1)
        self.text(self.f_lab, f"PAD {2 * cfg.pad_half_w:.0f} M",
                  THEME.struct, cx + 12, ly - 18)

    def draw_trail(self, trail):
        n = len(trail)
        for i in range(1, n):
            t = i / n  # 0 = oldest, 1 = newest
            color = lerp_color(THEME.field, THEME.signal, 0.08 + 0.55 * t * t)
            pygame.draw.line(self.screen, color,
                             world_to_screen(*trail[i - 1]),
                             world_to_screen(*trail[i]), 1)

    def draw_prediction(self, pts, impact):
        dim = lerp_color(THEME.field, THEME.signal, 0.30)
        for wx, wy in pts:
            px, py = world_to_screen(wx, wy)
            if 0 <= px < VIEW_W and 0 <= py < VIEW_H:
                pygame.draw.line(self.screen, dim, (px, py), (px, py), 1)
                pygame.draw.line(self.screen, dim, (px - 1, py), (px + 1, py), 1)
        if impact is not None:
            wx, wy, speed = impact
            px, py = world_to_screen(wx, wy)
            for dx, dy in ((-4, -4), (-4, 4)):
                pygame.draw.line(self.screen, dim, (px + dx, py + dy),
                                 (px - dx, py - dy), 1)
            self.text(self.f_lab, f"{speed:4.1f} M/S", dim,
                      min(px + 8, VIEW_W - 70), max(py - 18, 2))

    def draw_rays(self, env):
        cfg = env.cfg
        ox, oy = env.state[X], env.state[Y]
        dists = env.ray_distances()
        angles = np.linspace(cfg.ray_angle_lo, cfg.ray_angle_hi, cfg.n_rays)
        for a, d in zip(angles, dists):
            hx, hy = ox + math.cos(a) * d, oy + math.sin(a) * d
            pygame.draw.line(self.screen, THEME.grid,
                             world_to_screen(ox, oy), world_to_screen(hx, hy), 1)
            if d < cfg.ray_max_range:
                px, py = world_to_screen(hx, hy)
                pygame.draw.line(self.screen, THEME.struct,
                                 (px - 2, py), (px + 2, py), 1)

    def draw_rocket(self, env, action, phosphor):
        cfg, s = env.cfg, env.state
        theta = s[THETA]
        nx, ny = -math.sin(theta), math.cos(theta)   # nose
        px, py = ny, -nx                             # body-right (perp)
        base, tip = body_endpoints(s, cfg)
        w = 0.5  # half-width, m

        def pt(cx, cy, along, across):
            return world_to_screen(cx + nx * along + px * across,
                                   cy + ny * along + py * across)

        bx, by = base
        corners = [pt(bx, by, 0, -w), pt(bx, by, 0, w),
                   pt(bx, by, cfg.H * 0.82, w),
                   pt(bx, by, cfg.H, 0),              # nose point
                   pt(bx, by, cfg.H * 0.82, -w)]
        pygame.draw.polygon(self.screen, THEME.signal, corners, 1)
        pygame.draw.line(self.screen, lerp_color(THEME.field, THEME.signal, 0.35),
                         world_to_screen(*base), world_to_screen(*tip), 1)
        if phosphor:  # ghost copy -> motion smear on the decay layer
            phosphor.polygon(corners, THEME.signal, 30)

        throttle, gimbal_cmd = float(action[0]), float(action[1])
        phi = gimbal_cmd * cfg.phi_max
        ex_dir = (math.sin(theta + phi), -math.cos(theta + phi))  # exhaust dir

        # gimbal indicator: nozzle stub along the gimbaled axis
        nz = (bx + ex_dir[0] * 0.8, by + ex_dir[1] * 0.8)
        pygame.draw.line(self.screen, THEME.signal,
                         world_to_screen(bx, by), world_to_screen(*nz), 2)

        if throttle > 0.01 and s[FUEL] > 0:
            length = 7.0 * throttle * cfg.thrust_multiplier
            hx, hy = bx + ex_dir[0] * length, by + ex_dir[1] * length
            a0 = world_to_screen(bx + ex_dir[0] * 0.9, by + ex_dir[1] * 0.9)
            a1 = world_to_screen(hx, hy)
            pygame.draw.line(self.screen, THEME.signal, a0, a1, 1)
            ang = math.atan2(a1[1] - a0[1], a1[0] - a0[0])
            for da in (2.6, -2.6):
                pygame.draw.line(self.screen, THEME.signal, a1,
                                 (a1[0] + 6 * math.cos(ang + da),
                                  a1[1] + 6 * math.sin(ang + da)), 1)
            if phosphor:  # jittered exhaust streaks, render-only randomness
                perp = (-ex_dir[1], ex_dir[0])
                for _ in range(3):
                    j = random.uniform(-0.5, 0.5)
                    frac = random.uniform(0.5, 1.1)
                    sx = bx + ex_dir[0] * 0.9 + perp[0] * j
                    sy = by + ex_dir[1] * 0.9 + perp[1] * j
                    exx = sx + ex_dir[0] * length * frac
                    exy = sy + ex_dir[1] * length * frac
                    phosphor.line(world_to_screen(sx, sy),
                                  world_to_screen(exx, exy),
                                  THEME.signal, 46, 2)

    # ------------------------------------------------------------- telemetry
    def bar(self, x, y, w, h, frac, color):
        pygame.draw.rect(self.screen, THEME.struct, (x, y, w, h), 1)
        fill = int((w - 2) * max(0.0, min(1.0, frac)))
        if fill > 0:
            pygame.draw.rect(self.screen, color, (x + 1, y + 1, fill, h - 2))

    def draw_panel(self, env, action, ep_reward, t_sim, seed, fuel_out,
                   charts, log, assist):
        cfg, s = env.cfg, env.state
        x0 = VIEW_W + 24
        xr = WIN_W - 24
        pygame.draw.line(self.screen, THEME.struct, (VIEW_W, 0), (VIEW_W, WIN_H))

        y = 18
        self.text(self.f_lab, "ROCKET GNC — FLIGHT CONSOLE", THEME.struct, x0, y)
        self.text(self.f_lab, "STAB AUG" if assist else "MANUAL",
                  THEME.signal, xr, y, right_align=True)
        y += 24

        alt = s[Y] - cfg.L - env.terrain.height_at(s[X])
        tilt = math.degrees(s[THETA])
        rows = [
            ("T+", f"{t_sim:6.2f} S"),
            ("SEED", f"{seed}"),
            ("ALT", f"{alt:7.2f} M"),
            ("VX", f"{s[VX]:+7.2f} M/S"),
            ("VY", f"{s[VY]:+7.2f} M/S"),
            ("TILT", f"{tilt:+7.2f} °"),
            ("OMEGA", f"{s[OMEGA]:+7.3f} R/S"),
            ("GIMBAL", f"{math.degrees(float(action[1]) * cfg.phi_max):+6.2f} °"),
            ("REWARD", f"{ep_reward:+8.2f}"),
        ]
        for label, val in rows:
            self.text(self.f_lab, label, THEME.struct, x0, y + 2)
            self.text(self.f_num, val, THEME.signal, xr, y, right_align=True)
            y += 20

        y += 6
        self.text(self.f_lab, "THROTTLE", THEME.struct, x0, y)
        self.bar(x0 + 90, y + 1, xr - x0 - 90, 10, float(action[0]), THEME.signal)
        y += 20
        fuel_frac = s[FUEL] / cfg.fuel_0
        self.text(self.f_lab, "FUEL", THEME.fault if fuel_out else THEME.struct,
                  x0, y)
        self.bar(x0 + 90, y + 1, xr - x0 - 90, 10, fuel_frac,
                 THEME.fault if fuel_out else THEME.signal)
        y += 26

        # attitude indicator: vertical reference + body line
        cx, cy, r = x0 + 45, y + 42, 38
        pygame.draw.circle(self.screen, THEME.struct, (cx, cy), r, 1)
        pygame.draw.line(self.screen, THEME.grid, (cx, cy - r), (cx, cy + r), 1)
        bxn, byn = -math.sin(s[THETA]), math.cos(s[THETA])
        pygame.draw.line(self.screen, THEME.signal,
                         (cx - int(bxn * r * 0.9), cy + int(byn * r * 0.9)),
                         (cx + int(bxn * r * 0.9), cy - int(byn * r * 0.9)), 1)
        self.text(self.f_lab, "ATT", THEME.struct, cx - 10, cy + r + 4)

        # velocity vector indicator
        cx2 = x0 + 150
        pygame.draw.circle(self.screen, THEME.struct, (cx2, cy), r, 1)
        pygame.draw.line(self.screen, THEME.grid, (cx2 - r, cy), (cx2 + r, cy), 1)
        vmax = 20.0
        vx_n = max(-1.0, min(1.0, s[VX] / vmax))
        vy_n = max(-1.0, min(1.0, s[VY] / vmax))
        pygame.draw.line(self.screen, THEME.signal, (cx2, cy),
                         (cx2 + int(vx_n * r), cy - int(vy_n * r)), 1)
        self.text(self.f_lab, "VEL", THEME.struct, cx2 - 10, cy + r + 4)

        # raycast readout, right of the dials
        rx0 = x0 + 230
        self.text(self.f_lab, "RANGING M", THEME.struct, rx0, cy - r)
        for i, v in enumerate(env.ray_distances()):
            self.text(self.f_num, f"{v:5.1f}", THEME.signal,
                      rx0, cy - r + 18 + i * 17)
        y = cy + r + 24

        # strip charts
        chart_w = xr - x0
        for chart in charts:
            self.text(self.f_lab, chart.label, THEME.struct, x0, y)
            chart.draw(self.screen, x0, y + 15, chart_w, 56)
            y += 78

        # event log
        self.text(self.f_lab, "LOG", THEME.struct, x0, y)
        y += 17
        for t, line, fault in log.lines:
            color = THEME.fault if fault else THEME.signal
            self.text(self.f_lab, f"T+{t:05.1f}  {line}", color, x0, y)
            y += 16

        self.text(self.f_lab,
                  "W THROTTLE   A/D STEER   S CUT   SPACE MAX",
                  THEME.struct, x0, WIN_H - 40)
        self.text(self.f_lab,
                  "G STAB AUG   C CRT   V RAYS   R RESET   ESC QUIT",
                  THEME.struct, x0, WIN_H - 22)

    def draw_stamp(self, text, fault=False):
        color = THEME.fault if fault else THEME.signal
        surf = self.f_stamp.render(text, True, color)
        x = (VIEW_W - surf.get_width()) // 2
        y = VIEW_H // 3
        pad = 10
        rect = (x - pad, y - pad, surf.get_width() + 2 * pad,
                surf.get_height() + 2 * pad)
        pygame.draw.rect(self.screen, THEME.field, rect)
        pygame.draw.rect(self.screen, color, rect, 1)
        self.screen.blit(surf, (x, y))


def stamp_for(outcome, state, cfg):
    """Returns (text, is_fault)."""
    if outcome == TOUCHDOWN:
        fuel_pct = 100.0 * state[FUEL] / cfg.fuel_0
        return (f"TOUCHDOWN — VY {abs(state[VY]):.1f} M/S · "
                f"VX {abs(state[VX]):.1f} M/S · FUEL {fuel_pct:.0f}%", False)
    if outcome == "OUT OF BOUNDS":
        return "OUT OF BOUNDS — CONTACT LOST", True
    impact = math.hypot(state[VX], state[VY])
    return f"LOSS OF VEHICLE — IMPACT {impact:.1f} M/S", True


def main():
    # Opt out of Windows DPI virtualization: without this the window is
    # bitmap-stretched at >100% display scaling and all text goes soft.
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("ROCKET GNC")
    clock = pygame.time.Clock()
    console = Console(screen)
    phosphor = Phosphor((VIEW_W, VIEW_H))
    scanlines = make_scanlines(WIN_W, WIN_H)
    crt_on = True

    env = RocketEnv()
    fc = FlightComputer()
    seed = random.SystemRandom().randrange(10_000)
    charts = [StripChart("ALT M", 0.0, 100.0),
              StripChart("VY M/S", -25.0, 25.0, zero_line=True)]
    log = EventLog()

    ep_reward = t_sim = end_timer = 0.0
    ended, stamp, trail = False, None, deque(maxlen=360)
    ignited = False
    fuel_marks: set[int] = set()

    def new_episode():
        nonlocal ep_reward, t_sim, ended, end_timer, stamp, trail, seed
        nonlocal ignited, fuel_marks
        seed += 1
        env.reset(seed=seed)
        fc.reset()
        for c in charts:
            c.reset()
        log.reset()
        phosphor.clear()
        trail = deque(maxlen=360)
        ep_reward, t_sim = 0.0, 0.0
        ended, end_timer, stamp = False, 0.0, None
        ignited = False
        fuel_marks = set()
        log.post(0.0, f"GUIDANCE ACTIVE — SEED {seed}")

    show_rays = True
    new_episode()

    running = True
    while running:
        dt = env.cfg.dt
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    new_episode()
                elif ev.key == pygame.K_v:
                    show_rays = not show_rays
                elif ev.key == pygame.K_c:
                    crt_on = not crt_on
                elif ev.key == pygame.K_g:
                    fc.assist = not fc.assist
                    log.post(t_sim,
                             "STAB AUG ENGAGED" if fc.assist else "MANUAL CONTROL")

        keys = pygame.key.get_pressed()
        if not ended:
            fc.update(keys, env.state, dt)
            action = fc.action()
            _, r, term, trunc, info = env.step(action)
            ep_reward += r
            t_sim += dt
            trail.append((env.state[X], env.state[Y]))
            charts[0].push(env.state[Y] - env.cfg.L)
            charts[1].push(env.state[VY])

            if not ignited and action[0] > 0.05:
                ignited = True
                log.post(t_sim, "IGNITION")
            fuel_frac = env.state[FUEL] / env.cfg.fuel_0
            for mark in (50, 25):
                if fuel_frac * 100 <= mark and mark not in fuel_marks:
                    fuel_marks.add(mark)
                    log.post(t_sim, f"FUEL {mark}%")
            if fuel_frac <= 0.0 and 0 not in fuel_marks:
                fuel_marks.add(0)
                log.post(t_sim, "FUEL DEPLETED", fault=True)

            if term or trunc:
                ended, end_timer = True, 0.0
                outcome = info.get("outcome", "TIMEOUT")
                if outcome == "TIMEOUT":
                    stamp = ("EPISODE TIMEOUT — 20.0 S", True)
                else:
                    stamp = stamp_for(outcome, info["state"], env.cfg)
                log.post(t_sim, stamp[0], fault=stamp[1])
        else:
            action = np.zeros(2, dtype=np.float32)
            end_timer += dt
            if end_timer > END_FREEZE_S:
                new_episode()

        # ------------------------------------------------------------ render
        screen.fill(THEME.field)
        console.draw_graticule(env.cfg)
        console.draw_terrain(env)
        ph = phosphor if crt_on else None
        console.draw_pad(env, ph)
        console.draw_trail(list(trail))
        if not ended:
            pts, impact = predict_ballistic(env)
            console.draw_prediction(pts, impact)
        if show_rays and not ended:
            console.draw_rays(env)
        if crt_on:
            phosphor.decay()
        console.draw_rocket(env, action, ph)
        if crt_on:
            screen.blit(phosphor.surf, (0, 0))
        fuel_out = env.state[FUEL] <= 0.0
        console.draw_panel(env, action, ep_reward, t_sim, seed, fuel_out,
                           charts, log, fc.assist)
        if ended and stamp:
            console.draw_stamp(*stamp)
        if crt_on:
            screen.blit(scanlines, (0, 0))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
