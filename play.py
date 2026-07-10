"""Keyboard flight harness for RocketEnv.

Controls
  W           throttle (ramped while held)
  SPACE       full throttle (instant while held)
  A / D       gimbal (ramped, springs back to center)
  S           cut throttle instantly
  T           cycle visual theme
  V           toggle raycast display
  R           reset episode
  ESC         quit

The keyboard drives a *virtual stick* whose output is the same continuous
[throttle, gimbal] action an agent would emit — the env code path is
identical for human and agent.

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
from rocketenv.physics import FUEL, OMEGA, THETA, VX, VY, X, Y, body_endpoints
from rocketenv.reward import TOUCHDOWN

SCALE = 8                 # px per meter
VIEW_W, VIEW_H = 800, 800  # world viewport (100 m x 100 m)
PANEL_W = 420
WIN_W, WIN_H = VIEW_W + PANEL_W, VIEW_H
END_FREEZE_S = 2.5
CHART_SECONDS = 12.0      # strip-chart history window


# ---------------------------------------------------------------- themes
@dataclass(frozen=True)
class Theme:
    name: str
    field: tuple      # background
    grid: tuple       # graticule, faintest
    struct: tuple     # terrain / chrome / labels
    signal: tuple     # the system's voice: rocket, telemetry, trail
    fault: tuple      # genuine faults only


THEMES = [
    # Oscilloscope heritage: instrument amber on near-black ink.
    Theme("SCOPE", (8, 11, 16), (20, 26, 34), (58, 70, 84),
          (255, 176, 0), (255, 59, 48)),
    # Cyanotype drafting print: pale linework on Prussian blue.
    Theme("BLUEPRINT", (13, 36, 64), (26, 54, 88), (110, 140, 170),
          (232, 242, 250), (255, 106, 61)),
    # Swiss technical drawing: ink on warm paper, one red accent.
    Theme("PAPER", (244, 241, 234), (227, 222, 210), (140, 134, 122),
          (28, 26, 22), (208, 52, 44)),
]


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def world_to_screen(x, y):
    """The ONLY place y flips."""
    return int(x * SCALE), int(VIEW_H - y * SCALE)


class VirtualStick:
    """Ramped keyboard -> continuous action. Same action space as the agent."""

    THROTTLE_UP = 4.0     # /s toward 1 while W held (0.25 s to full)
    THROTTLE_DOWN = 5.0   # /s toward 0 when released
    GIMBAL_SLEW = 3.0     # /s toward +/-1 while A/D held
    GIMBAL_SPRING = 6.0   # /s back to center

    def __init__(self):
        self.throttle = 0.0
        self.gimbal = 0.0

    def reset(self):
        self.throttle = 0.0
        self.gimbal = 0.0

    def update(self, keys, dt):
        if keys[pygame.K_SPACE]:
            self.throttle = 1.0
        elif keys[pygame.K_s]:
            self.throttle = 0.0
        elif keys[pygame.K_w]:
            self.throttle = min(1.0, self.throttle + self.THROTTLE_UP * dt)
        else:
            self.throttle = max(0.0, self.throttle - self.THROTTLE_DOWN * dt)

        target = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
        rate = self.GIMBAL_SLEW if target != 0.0 else self.GIMBAL_SPRING
        if self.gimbal < target:
            self.gimbal = min(target, self.gimbal + rate * dt)
        elif self.gimbal > target:
            self.gimbal = max(target, self.gimbal - rate * dt)

    def action(self):
        return np.array([self.throttle, self.gimbal], dtype=np.float32)


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

    def draw(self, screen, theme, x, y, w, h):
        pygame.draw.rect(screen, theme.grid, (x, y, w, h), 1)
        if self.zero_line and self.lo < 0 < self.hi:
            zy = y + h - int((0 - self.lo) / (self.hi - self.lo) * h)
            pygame.draw.line(screen, theme.grid, (x + 1, zy), (x + w - 1, zy))
        n = len(self.data)
        if n < 2:
            return
        maxn = self.data.maxlen
        pts = []
        for i, v in enumerate(self.data):
            px = x + w - 1 - int((n - 1 - i) * (w - 2) / (maxn - 1))
            frac = (min(max(v, self.lo), self.hi) - self.lo) / (self.hi - self.lo)
            pts.append((px, y + h - 1 - int(frac * (h - 2))))
        if len(pts) >= 2:
            pygame.draw.lines(screen, theme.signal, False, pts, 1)


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
        self.theme = THEMES[0]
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
            pygame.draw.line(self.screen, self.theme.grid, (sx, 0), (sx, VIEW_H))
        for gy in range(0, int(cfg.world_h) + 1, 10):
            _, sy = world_to_screen(0, gy)
            pygame.draw.line(self.screen, self.theme.grid, (0, sy), (VIEW_W, sy))

    def draw_terrain(self, env):
        cfg = env.cfg
        pts = [world_to_screen(x, env.terrain.height_at(x))
               for x in range(0, int(cfg.world_w) + 1, 2)]
        pygame.draw.lines(self.screen, self.theme.struct, False, pts, 1)

    def draw_pad(self, env):
        cfg = env.cfg
        pad_y = env.terrain.height_at(cfg.pad_x)
        lx, ly = world_to_screen(cfg.pad_x - cfg.pad_half_w, pad_y)
        rx, _ = world_to_screen(cfg.pad_x + cfg.pad_half_w, pad_y)
        cx, _ = world_to_screen(cfg.pad_x, pad_y)
        arm = 8
        sig = self.theme.signal
        for ex, sgn in ((lx, 1), (rx, -1)):
            pygame.draw.line(self.screen, sig, (ex, ly), (ex, ly - arm), 1)
            pygame.draw.line(self.screen, sig, (ex, ly), (ex + sgn * arm, ly), 1)
        pygame.draw.line(self.screen, sig, (cx - 4, ly), (cx + 4, ly), 1)
        pygame.draw.line(self.screen, sig, (cx, ly - 4), (cx, ly + 4), 1)
        self.text(self.f_lab, f"PAD {2 * cfg.pad_half_w:.0f} M",
                  self.theme.struct, cx + 12, ly - 18)

    def draw_trail(self, trail):
        n = len(trail)
        for i in range(1, n):
            t = i / n  # 0 = oldest, 1 = newest
            color = lerp_color(self.theme.field, self.theme.signal,
                               0.08 + 0.55 * t * t)
            pygame.draw.line(self.screen, color,
                             world_to_screen(*trail[i - 1]),
                             world_to_screen(*trail[i]), 1)

    def draw_rays(self, env):
        cfg = env.cfg
        ox, oy = env.state[X], env.state[Y]
        dists = env.ray_distances()
        angles = np.linspace(cfg.ray_angle_lo, cfg.ray_angle_hi, cfg.n_rays)
        for a, d in zip(angles, dists):
            hx, hy = ox + math.cos(a) * d, oy + math.sin(a) * d
            pygame.draw.line(self.screen, self.theme.grid,
                             world_to_screen(ox, oy), world_to_screen(hx, hy), 1)
            if d < cfg.ray_max_range:
                px, py = world_to_screen(hx, hy)
                pygame.draw.line(self.screen, self.theme.struct,
                                 (px - 2, py), (px + 2, py), 1)

    def draw_rocket(self, env, action):
        cfg, s = env.cfg, env.state
        theme = self.theme
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
        pygame.draw.polygon(self.screen, theme.signal, corners, 1)
        pygame.draw.line(self.screen, lerp_color(theme.field, theme.signal, 0.35),
                         world_to_screen(*base), world_to_screen(*tip), 1)

        throttle, gimbal_cmd = float(action[0]), float(action[1])
        phi = gimbal_cmd * cfg.phi_max
        ex_dir = (math.sin(theta + phi), -math.cos(theta + phi))  # exhaust dir

        # gimbal indicator: nozzle stub along the gimbaled axis
        nz = (bx + ex_dir[0] * 0.8, by + ex_dir[1] * 0.8)
        pygame.draw.line(self.screen, theme.signal,
                         world_to_screen(bx, by), world_to_screen(*nz), 2)

        # thrust vector arrow, length proportional to actual thrust
        if throttle > 0.01 and s[FUEL] > 0:
            length = 7.0 * throttle * cfg.thrust_multiplier
            hx, hy = bx + ex_dir[0] * length, by + ex_dir[1] * length
            a0 = world_to_screen(bx + ex_dir[0] * 0.9, by + ex_dir[1] * 0.9)
            a1 = world_to_screen(hx, hy)
            pygame.draw.line(self.screen, theme.signal, a0, a1, 1)
            ang = math.atan2(a1[1] - a0[1], a1[0] - a0[0])
            for da in (2.6, -2.6):
                pygame.draw.line(self.screen, theme.signal, a1,
                                 (a1[0] + 6 * math.cos(ang + da),
                                  a1[1] + 6 * math.sin(ang + da)), 1)

    # ------------------------------------------------------------- telemetry
    def bar(self, x, y, w, h, frac, color):
        pygame.draw.rect(self.screen, self.theme.struct, (x, y, w, h), 1)
        fill = int((w - 2) * max(0.0, min(1.0, frac)))
        if fill > 0:
            pygame.draw.rect(self.screen, color, (x + 1, y + 1, fill, h - 2))

    def draw_panel(self, env, action, ep_reward, t_sim, seed, fuel_out,
                   charts, log):
        cfg, s = env.cfg, env.state
        theme = self.theme
        x0 = VIEW_W + 24
        xr = WIN_W - 24
        pygame.draw.line(self.screen, theme.struct, (VIEW_W, 0), (VIEW_W, WIN_H))

        y = 18
        self.text(self.f_lab, "ROCKET GNC — FLIGHT CONSOLE", theme.struct, x0, y)
        self.text(self.f_lab, theme.name, theme.signal, xr, y, right_align=True)
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
            self.text(self.f_lab, label, theme.struct, x0, y + 2)
            self.text(self.f_num, val, theme.signal, xr, y, right_align=True)
            y += 20

        y += 6
        self.text(self.f_lab, "THROTTLE", theme.struct, x0, y)
        self.bar(x0 + 90, y + 1, xr - x0 - 90, 10, float(action[0]), theme.signal)
        y += 20
        fuel_frac = s[FUEL] / cfg.fuel_0
        self.text(self.f_lab, "FUEL", theme.fault if fuel_out else theme.struct,
                  x0, y)
        self.bar(x0 + 90, y + 1, xr - x0 - 90, 10, fuel_frac,
                 theme.fault if fuel_out else theme.signal)
        y += 26

        # attitude indicator: vertical reference + body line
        cx, cy, r = x0 + 45, y + 42, 38
        pygame.draw.circle(self.screen, theme.struct, (cx, cy), r, 1)
        pygame.draw.line(self.screen, theme.grid, (cx, cy - r), (cx, cy + r), 1)
        bxn, byn = -math.sin(s[THETA]), math.cos(s[THETA])
        pygame.draw.line(self.screen, theme.signal,
                         (cx - int(bxn * r * 0.9), cy + int(byn * r * 0.9)),
                         (cx + int(bxn * r * 0.9), cy - int(byn * r * 0.9)), 1)
        self.text(self.f_lab, "ATT", theme.struct, cx - 10, cy + r + 4)

        # velocity vector indicator
        cx2 = x0 + 150
        pygame.draw.circle(self.screen, theme.struct, (cx2, cy), r, 1)
        pygame.draw.line(self.screen, theme.grid, (cx2 - r, cy), (cx2 + r, cy), 1)
        vmax = 20.0
        vx_n = max(-1.0, min(1.0, s[VX] / vmax))
        vy_n = max(-1.0, min(1.0, s[VY] / vmax))
        pygame.draw.line(self.screen, theme.signal, (cx2, cy),
                         (cx2 + int(vx_n * r), cy - int(vy_n * r)), 1)
        self.text(self.f_lab, "VEL", theme.struct, cx2 - 10, cy + r + 4)

        # raycast readout, right of the dials
        rx0 = x0 + 230
        self.text(self.f_lab, "RANGING M", theme.struct, rx0, cy - r)
        for i, v in enumerate(env.ray_distances()):
            self.text(self.f_num, f"{v:5.1f}", theme.signal, rx0, cy - r + 18 + i * 17)
        y = cy + r + 24

        # strip charts
        chart_w = xr - x0
        for chart in charts:
            self.text(self.f_lab, chart.label, theme.struct, x0, y)
            chart.draw(self.screen, theme, x0, y + 15, chart_w, 56)
            y += 78

        # event log
        self.text(self.f_lab, "LOG", theme.struct, x0, y)
        y += 17
        for t, line, fault in log.lines:
            color = theme.fault if fault else theme.signal
            self.text(self.f_lab, f"T+{t:05.1f}  {line}", color, x0, y)
            y += 16

        self.text(self.f_lab,
                  "W THROTTLE   A/D GIMBAL   S CUT   SPACE MAX",
                  theme.struct, x0, WIN_H - 40)
        self.text(self.f_lab,
                  "T THEME   V RAYS   R RESET   ESC QUIT",
                  theme.struct, x0, WIN_H - 22)

    def draw_stamp(self, text, fault=False):
        color = self.theme.fault if fault else self.theme.signal
        surf = self.f_stamp.render(text, True, color)
        x = (VIEW_W - surf.get_width()) // 2
        y = VIEW_H // 3
        pad = 10
        rect = (x - pad, y - pad, surf.get_width() + 2 * pad,
                surf.get_height() + 2 * pad)
        pygame.draw.rect(self.screen, self.theme.field, rect)
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

    env = RocketEnv()
    stick = VirtualStick()
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
        stick.reset()
        for c in charts:
            c.reset()
        log.reset()
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
                elif ev.key == pygame.K_t:
                    idx = THEMES.index(console.theme)
                    console.theme = THEMES[(idx + 1) % len(THEMES)]

        keys = pygame.key.get_pressed()
        if not ended:
            stick.update(keys, dt)
            action = stick.action()
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
        screen.fill(console.theme.field)
        console.draw_graticule(env.cfg)
        console.draw_terrain(env)
        console.draw_pad(env)
        console.draw_trail(list(trail))
        if show_rays and not ended:
            console.draw_rays(env)
        console.draw_rocket(env, action)
        fuel_out = env.state[FUEL] <= 0.0
        console.draw_panel(env, action, ep_reward, t_sim, seed, fuel_out,
                           charts, log)
        if ended and stamp:
            console.draw_stamp(*stamp)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
