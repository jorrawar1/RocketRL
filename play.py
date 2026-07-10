"""Keyboard flight harness for RocketEnv.

Controls
  UP          throttle (ramped while held)
  SPACE       full throttle (instant while held)
  LEFT/RIGHT  gimbal (ramped, springs back to center)
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

import math
import random
from collections import deque

import numpy as np
import pygame

from rocketenv import RocketEnv
from rocketenv.physics import FUEL, OMEGA, THETA, VX, VY, X, Y, body_endpoints
from rocketenv.reward import TOUCHDOWN

# ---------------------------------------------------------------- palette
INK = (8, 11, 16)         # field            #080B10
GRID = (20, 26, 34)       # graticule        #141A22
STRUCT = (58, 70, 84)     # structure/chrome #3A4654
AMBER = (255, 176, 0)     # signal accent    #FFB000
FAULT = (255, 59, 48)     # faults only      #FF3B30

SCALE = 8                 # px per meter
VIEW_W, VIEW_H = 800, 800  # world viewport (100 m x 100 m)
PANEL_W = 420
WIN_W, WIN_H = VIEW_W + PANEL_W, VIEW_H
END_FREEZE_S = 2.5


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def world_to_screen(x, y):
    """The ONLY place y flips."""
    return int(x * SCALE), int(VIEW_H - y * SCALE)


class VirtualStick:
    """Ramped keyboard -> continuous action. Same action space as the agent."""

    THROTTLE_UP = 2.5     # /s toward 1 while UP held
    THROTTLE_DOWN = 3.5   # /s toward 0 when released
    GIMBAL_SLEW = 5.0     # /s toward +/-1 while LEFT/RIGHT held
    GIMBAL_SPRING = 8.0   # /s back to center

    def __init__(self):
        self.throttle = 0.0
        self.gimbal = 0.0

    def reset(self):
        self.throttle = 0.0
        self.gimbal = 0.0

    def update(self, keys, dt):
        if keys[pygame.K_SPACE]:
            self.throttle = 1.0
        elif keys[pygame.K_UP]:
            self.throttle = min(1.0, self.throttle + self.THROTTLE_UP * dt)
        else:
            self.throttle = max(0.0, self.throttle - self.THROTTLE_DOWN * dt)

        target = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
        rate = self.GIMBAL_SLEW if target != 0.0 else self.GIMBAL_SPRING
        if self.gimbal < target:
            self.gimbal = min(target, self.gimbal + rate * dt)
        elif self.gimbal > target:
            self.gimbal = max(target, self.gimbal - rate * dt)

    def action(self):
        return np.array([self.throttle, self.gimbal], dtype=np.float32)


class Console:
    """All drawing. Reads env state; never writes it."""

    def __init__(self, screen):
        self.screen = screen
        names = "jetbrains mono, cascadia mono, ibm plex mono, consolas, courier new"
        self.f_num = pygame.font.SysFont(names, 15)
        self.f_lab = pygame.font.SysFont(names, 12)
        self.f_stamp = pygame.font.SysFont(names, 19, bold=True)

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
            pygame.draw.line(self.screen, GRID, (sx, 0), (sx, VIEW_H))
        for gy in range(0, int(cfg.world_h) + 1, 10):
            _, sy = world_to_screen(0, gy)
            pygame.draw.line(self.screen, GRID, (0, sy), (VIEW_W, sy))

    def draw_terrain(self, env):
        cfg = env.cfg
        pts = [world_to_screen(x, env.terrain.height_at(x))
               for x in range(0, int(cfg.world_w) + 1, 2)]
        pygame.draw.lines(self.screen, STRUCT, False, pts, 1)

    def draw_pad(self, env):
        cfg = env.cfg
        pad_y = env.terrain.height_at(cfg.pad_x)
        lx, ly = world_to_screen(cfg.pad_x - cfg.pad_half_w, pad_y)
        rx, _ = world_to_screen(cfg.pad_x + cfg.pad_half_w, pad_y)
        cx, _ = world_to_screen(cfg.pad_x, pad_y)
        arm = 8
        # survey brackets at pad edges
        for ex, sgn in ((lx, 1), (rx, -1)):
            pygame.draw.line(self.screen, AMBER, (ex, ly), (ex, ly - arm), 1)
            pygame.draw.line(self.screen, AMBER, (ex, ly), (ex + sgn * arm, ly), 1)
        # center crosshair
        pygame.draw.line(self.screen, AMBER, (cx - 4, ly), (cx + 4, ly), 1)
        pygame.draw.line(self.screen, AMBER, (cx, ly - 4), (cx, ly + 4), 1)
        self.text(self.f_lab, f"PAD {2 * cfg.pad_half_w:.0f} M",
                  STRUCT, cx + 12, ly - 18)

    def draw_trail(self, trail):
        n = len(trail)
        for i in range(1, n):
            t = i / n  # 0 = oldest, 1 = newest
            color = lerp_color(INK, AMBER, 0.08 + 0.55 * t * t)
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
            pygame.draw.line(self.screen, GRID,
                             world_to_screen(ox, oy), world_to_screen(hx, hy), 1)
            if d < cfg.ray_max_range:
                px, py = world_to_screen(hx, hy)
                pygame.draw.line(self.screen, STRUCT, (px - 2, py), (px + 2, py), 1)

    def draw_rocket(self, env, action):
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
        pygame.draw.polygon(self.screen, AMBER, corners, 1)
        # centerline
        pygame.draw.line(self.screen, lerp_color(INK, AMBER, 0.35),
                         world_to_screen(*base), world_to_screen(*tip), 1)

        throttle, gimbal_cmd = float(action[0]), float(action[1])
        phi = gimbal_cmd * cfg.phi_max
        ex_dir = (math.sin(theta + phi), -math.cos(theta + phi))  # exhaust dir

        # gimbal indicator: nozzle stub along the gimbaled axis
        nz = (bx + ex_dir[0] * 0.8, by + ex_dir[1] * 0.8)
        pygame.draw.line(self.screen, AMBER,
                         world_to_screen(bx, by), world_to_screen(*nz), 2)

        # thrust vector arrow, length proportional to actual thrust
        if throttle > 0.01 and s[FUEL] > 0:
            length = 7.0 * throttle * cfg.thrust_multiplier
            hx, hy = bx + ex_dir[0] * length, by + ex_dir[1] * length
            a0 = world_to_screen(bx + ex_dir[0] * 0.9, by + ex_dir[1] * 0.9)
            a1 = world_to_screen(hx, hy)
            pygame.draw.line(self.screen, AMBER, a0, a1, 1)
            # arrowhead
            ang = math.atan2(a1[1] - a0[1], a1[0] - a0[0])
            for da in (2.6, -2.6):
                pygame.draw.line(self.screen, AMBER, a1,
                                 (a1[0] + 6 * math.cos(ang + da),
                                  a1[1] + 6 * math.sin(ang + da)), 1)

    # ------------------------------------------------------------- telemetry
    def bar(self, x, y, w, h, frac, color):
        pygame.draw.rect(self.screen, STRUCT, (x, y, w, h), 1)
        fill = int((w - 2) * max(0.0, min(1.0, frac)))
        if fill > 0:
            pygame.draw.rect(self.screen, color, (x + 1, y + 1, fill, h - 2))

    def draw_panel(self, env, action, ep_reward, t_sim, seed, fuel_out):
        cfg, s = env.cfg, env.state
        x0 = VIEW_W + 24
        xr = WIN_W - 24
        pygame.draw.line(self.screen, STRUCT, (VIEW_W, 0), (VIEW_W, WIN_H))

        y = 22
        self.text(self.f_lab, "ROCKET GNC — FLIGHT CONSOLE", STRUCT, x0, y)
        y += 26

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
            self.text(self.f_lab, label, STRUCT, x0, y + 2)
            self.text(self.f_num, val, AMBER, xr, y, right_align=True)
            y += 20

        y += 8
        self.text(self.f_lab, "THROTTLE", STRUCT, x0, y)
        self.bar(x0 + 90, y + 1, xr - x0 - 90, 10, float(action[0]), AMBER)
        y += 20
        fuel_frac = s[FUEL] / cfg.fuel_0
        self.text(self.f_lab, "FUEL", FAULT if fuel_out else STRUCT, x0, y)
        self.bar(x0 + 90, y + 1, xr - x0 - 90, 10, fuel_frac,
                 FAULT if fuel_out else AMBER)
        y += 30

        # attitude indicator: vertical reference + body line
        cx, cy, r = x0 + 45, y + 45, 40
        pygame.draw.circle(self.screen, STRUCT, (cx, cy), r, 1)
        pygame.draw.line(self.screen, GRID, (cx, cy - r), (cx, cy + r), 1)
        bxn = -math.sin(s[THETA])
        byn = math.cos(s[THETA])
        pygame.draw.line(self.screen, AMBER, (cx - int(bxn * r * 0.9), cy + int(byn * r * 0.9)),
                         (cx + int(bxn * r * 0.9), cy - int(byn * r * 0.9)), 1)
        self.text(self.f_lab, "ATT", STRUCT, cx - 10, cy + r + 6)

        # velocity vector indicator
        cx2 = x0 + 175
        pygame.draw.circle(self.screen, STRUCT, (cx2, cy), r, 1)
        pygame.draw.line(self.screen, GRID, (cx2 - r, cy), (cx2 + r, cy), 1)
        vmax = 20.0
        vx_n = max(-1.0, min(1.0, s[VX] / vmax))
        vy_n = max(-1.0, min(1.0, s[VY] / vmax))
        pygame.draw.line(self.screen, AMBER, (cx2, cy),
                         (cx2 + int(vx_n * r), cy - int(vy_n * r)), 1)
        self.text(self.f_lab, "VEL", STRUCT, cx2 - 10, cy + r + 6)
        y = cy + r + 28

        # raycast readout
        self.text(self.f_lab, "RANGING", STRUCT, x0, y)
        y += 16
        d = env.ray_distances()
        self.text(self.f_num, "  ".join(f"{v:5.1f}" for v in d), AMBER, x0, y)
        y += 28

        if fuel_out:
            self.text(self.f_stamp, "FUEL DEPLETED", FAULT, x0, y)
            y += 30

        self.text(self.f_lab,
                  "↑ THROTTLE   ←/→ GIMBAL   SPACE MAX", STRUCT, x0, WIN_H - 44)
        self.text(self.f_lab,
                  "V RAYS   R RESET   ESC QUIT", STRUCT, x0, WIN_H - 26)

    def draw_stamp(self, text, color):
        surf = self.f_stamp.render(text, True, color)
        x = (VIEW_W - surf.get_width()) // 2
        y = VIEW_H // 3
        pad = 10
        pygame.draw.rect(self.screen, INK,
                         (x - pad, y - pad, surf.get_width() + 2 * pad,
                          surf.get_height() + 2 * pad))
        pygame.draw.rect(self.screen, color,
                         (x - pad, y - pad, surf.get_width() + 2 * pad,
                          surf.get_height() + 2 * pad), 1)
        self.screen.blit(surf, (x, y))


def stamp_for(outcome, state, cfg):
    if outcome == TOUCHDOWN:
        fuel_pct = 100.0 * state[FUEL] / cfg.fuel_0
        return (f"TOUCHDOWN — VY {abs(state[VY]):.1f} M/S · "
                f"VX {abs(state[VX]):.1f} M/S · FUEL {fuel_pct:.0f}%", AMBER)
    if outcome == "OUT OF BOUNDS":
        return "OUT OF BOUNDS — CONTACT LOST", FAULT
    impact = math.hypot(state[VX], state[VY])
    return f"LOSS OF VEHICLE — IMPACT {impact:.1f} M/S", FAULT


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("ROCKET GNC")
    clock = pygame.time.Clock()
    console = Console(screen)

    env = RocketEnv()
    stick = VirtualStick()
    seed = random.SystemRandom().randrange(10_000)

    def new_episode():
        nonlocal ep_reward, t_sim, ended, end_timer, stamp, trail, seed
        seed += 1
        env.reset(seed=seed)
        stick.reset()
        trail = deque(maxlen=360)
        ep_reward, t_sim = 0.0, 0.0
        ended, end_timer, stamp = False, 0.0, None

    ep_reward = t_sim = end_timer = 0.0
    ended, stamp, trail = False, None, deque(maxlen=360)
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

        keys = pygame.key.get_pressed()
        if not ended:
            stick.update(keys, dt)
            action = stick.action()
            _, r, term, trunc, info = env.step(action)
            ep_reward += r
            t_sim += dt
            trail.append((env.state[X], env.state[Y]))
            if term or trunc:
                ended, end_timer = True, 0.0
                outcome = info.get("outcome", "TIMEOUT")
                if outcome == "TIMEOUT":
                    stamp = ("EPISODE TIMEOUT — 20.0 S", STRUCT)
                else:
                    stamp = stamp_for(outcome, info["state"], env.cfg)
        else:
            action = np.zeros(2, dtype=np.float32)
            end_timer += dt
            if end_timer > END_FREEZE_S:
                new_episode()

        # ------------------------------------------------------------ render
        screen.fill(INK)
        console.draw_graticule(env.cfg)
        console.draw_terrain(env)
        console.draw_pad(env)
        console.draw_trail(list(trail))
        if show_rays and not ended:
            console.draw_rays(env)
        console.draw_rocket(env, action)
        fuel_out = env.state[FUEL] <= 0.0
        console.draw_panel(env, action, ep_reward, t_sim, seed, fuel_out)
        if ended and stamp:
            console.draw_stamp(*stamp)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
