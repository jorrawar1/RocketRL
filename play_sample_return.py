"""Playable autonomous crater sample-return mission.

The renderer stays deliberately close to :mod:`play`: simple vector geometry,
one amber instrument palette, and no external assets.  The mission-specific
layer makes the sample pickup and altered center of mass legible while the
camera keeps the complete BASE -> CRATER -> BASE route in context.

Controls
  W           throttle (ramped while held)
  SPACE       full throttle while held
  A / D       steer
  S           cut throttle
  G           cycle SAS mode: DAMP -> HOLD -> MANUAL
  F3          toggle payload/contact debug overlay
  C           toggle CRT effects
  V           toggle raycast display
  R           replay the same mission seed
  ESC         quit
"""

from __future__ import annotations

import argparse
import ctypes
import math
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np
import pygame

from play import (
    CAMERA,
    SCALE,
    THEME,
    VIEW_H,
    VIEW_W,
    WIN_H,
    WIN_W,
    Console,
    Debris,
    EventLog,
    FlightComputer,
    Phosphor,
    StripChart,
    make_scanlines,
    world_to_screen,
)
from rocketenv.physics import FUEL, THETA, VX, VY, X, Y
from rocketenv.sample_return import SampleReturnEnv, scripted_sample_return_action

try:
    # Milestone-2 geometry uses the combined COM as the state position.
    from rocketenv.sample_return.vehicle import (
        dry_body_center as _mission_dry_body_center,
        engine_position as _mission_engine_position,
    )
except (ImportError, ModuleNotFoundError):  # Milestone-1 prototype fallback
    _mission_dry_body_center = None
    _mission_engine_position = None


# Preserve enough pixel size for the rocket on the original 100 m prototype,
# but allow the 180-280 m crater map to determine a smaller fit scale.
MISSION_MAX_SCALE = SCALE * 0.55
MISSION_FRAME_MARGIN_PX = 48


@dataclass(frozen=True)
class PodView:
    """Small render-only view of the fixed side-pod specification."""

    mass: float = 0.35
    offset_x: float = 0.80
    offset_y: float = 0.0
    width: float = 0.65
    height: float = 0.85


def _phase_name(env) -> str:
    phase = getattr(env, "phase", None)
    if phase is None and hasattr(env, "mission_state"):
        phase = env.mission_state.phase
    name = phase.name if hasattr(phase, "name") else str(phase or "OUTBOUND")
    return name.upper()


def _mission_state_attr(env, name: str, default=None):
    state = getattr(env, "mission_state", None)
    return getattr(state, name, default)


def _payload_attached(env) -> bool:
    attached = _mission_state_attr(env, "payload_attached")
    if attached is not None:
        return bool(attached)
    return bool(getattr(env, "has_sample", False))


def _sample_collected(env) -> bool:
    collected = _mission_state_attr(env, "sample_collected")
    if collected is not None:
        return bool(collected)
    return bool(getattr(env, "has_sample", False))


def _contact_armed(env) -> bool:
    armed = _mission_state_attr(env, "contact_armed")
    if armed is not None:
        return bool(armed)
    return bool(getattr(env, "launch_armed", False))


def _sampling_progress(env) -> float:
    progress = getattr(env, "sampling_progress", None)
    if progress is not None:
        return float(np.clip(progress, 0.0, 1.0))
    if _payload_attached(env):
        return 1.0
    return 0.0


def _target_label(env) -> str:
    target = getattr(env, "target", None)
    if target is not None:
        return str(target).upper()
    return "BASE" if "RETURN" in _phase_name(env) else "SAMPLE"


def _target_x(env) -> float:
    value = getattr(env, "target_x", None)
    if value is not None:
        return float(value)
    return float(env.base_x if _target_label(env) == "BASE" else env.sample_x)


def _pod_view(env) -> PodView:
    vehicle = getattr(env, "vehicle", None)
    spec = getattr(vehicle, "payload", None)
    if spec is None:
        spec = getattr(env, "payload_spec", None)
    if spec is None:
        return PodView(mass=float(getattr(env, "payload_mass", 0.35) or 0.35))
    return PodView(
        mass=float(getattr(spec, "mass", 0.35)),
        offset_x=float(getattr(spec, "offset_body_x", 0.80)),
        offset_y=float(getattr(spec, "offset_body_y", 0.0)),
        width=float(getattr(spec, "width", 0.65)),
        height=float(getattr(spec, "height", 0.85)),
    )


def _dry_mass(env) -> float:
    vehicle = getattr(env, "vehicle", None)
    return float(getattr(vehicle, "dry_mass", getattr(env.cfg, "m", 1.0)))


def _body_basis(theta: float) -> tuple[np.ndarray, np.ndarray]:
    nose = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    right = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    return nose, right


def _dry_center(env) -> np.ndarray:
    vehicle = getattr(env, "vehicle", None)
    if _mission_dry_body_center is not None and vehicle is not None:
        return np.asarray(_mission_dry_body_center(env.state, vehicle), dtype=float)

    center = np.asarray(env.state[[X, Y]], dtype=np.float64)
    offset = getattr(vehicle, "com_offset_body", None)
    if offset is None:
        return center
    offset = np.asarray(offset, dtype=np.float64)
    nose, right = _body_basis(float(env.state[THETA]))
    return center - right * offset[0] - nose * offset[1]


def _engine_point(env, body_center: np.ndarray) -> np.ndarray:
    vehicle = getattr(env, "vehicle", None)
    if _mission_engine_position is not None and vehicle is not None:
        return np.asarray(_mission_engine_position(env.state, vehicle), dtype=float)
    nose, right = _body_basis(float(env.state[THETA]))
    ex = float(getattr(vehicle, "engine_body_x", 0.0))
    ey = float(getattr(vehicle, "engine_body_y", -env.cfg.H / 2.0))
    return body_center + right * ex + nose * ey


def _terrain_bounds(env) -> tuple[float, float]:
    ys = getattr(env.terrain, "ys", None)
    if ys is not None and len(ys):
        return float(np.min(ys)), float(np.max(ys))
    samples = [env.terrain.height_at(x)
               for x in np.linspace(0.0, env.cfg.world_w, 65)]
    return min(samples), max(samples)


def _is_grounded(env) -> bool:
    grounded_pad = getattr(env, "grounded_pad", None)
    if grounded_pad is not None:
        return True
    if "SAMPLING" in _phase_name(env):
        return True
    body_center = _dry_center(env)
    nose, _ = _body_basis(float(env.state[THETA]))
    height = float(getattr(getattr(env, "vehicle", None),
                           "body_height", env.cfg.H))
    base = body_center - nose * height / 2.0
    clearance = base[1] - env.terrain.height_at(float(base[0]))
    speed = math.hypot(float(env.state[VX]), float(env.state[VY]))
    return clearance <= 0.08 and speed < 0.2


def update_mission_camera(env, dt: float, *, snap: bool = False) -> None:
    """Track softly while guaranteeing that both mission pads stay visible."""
    context_xs = [float(env.base_x), float(env.sample_x)]
    for name in ("left_rim_x", "right_rim_x", "rim_left_x", "rim_right_x"):
        value = getattr(env.terrain, name, None)
        if value is not None:
            context_xs.append(float(value))
    left_pad = min(context_xs)
    right_pad = max(context_xs)
    route_margin = max(14.0, float(env.cfg.pad_half_w) + 8.0)
    route_span = max(1.0, right_pad - left_pad + 2.0 * route_margin)
    horizontal_fit = (VIEW_W - MISSION_FRAME_MARGIN_PX) / route_span
    vertical_fit = (VIEW_H - 70.0) / max(60.0, float(env.cfg.world_h) + 12.0)
    scale_t = max(1.5, min(MISSION_MAX_SCALE, horizontal_fit, vertical_fit))

    half_w = VIEW_W / (2.0 * scale_t)
    follow_lo = right_pad + route_margin - half_w
    follow_hi = left_pad - route_margin + half_w
    if follow_lo <= follow_hi:
        center_x = float(np.clip(env.state[X], follow_lo, follow_hi))
    else:
        center_x = 0.5 * (left_pad + right_pad)

    ground_lo, _ = _terrain_bounds(env)
    half_h = VIEW_H / (2.0 * scale_t)
    # Places the crater floor near the lower fifth of the viewport and leaves
    # stable headroom for traversal without bobbing with rocket altitude.
    center_y = max(float(env.cfg.world_h) * 0.50,
                   ground_lo + half_h * 0.80)

    # Sampling is the one event where route awareness is temporarily less
    # useful than legibility.  Dynamics are frozen, so a gentle close-in can
    # show the regolith stream, pod fill, and COM shift before overview returns.
    if "SAMPLING" in _phase_name(env):
        scale_t = max(scale_t, SCALE * 1.30)
        center_x = float(env.state[X])
        half_h = VIEW_H / (2.0 * scale_t)
        local_ground = env.terrain.height_at(float(env.state[X]))
        center_y = local_ground + half_h * 0.65

    blend = 1.0 if snap else min(1.0, 3.0 * dt)
    CAMERA.s += (scale_t - CAMERA.s) * blend
    CAMERA.cx += (center_x - CAMERA.cx) * blend
    CAMERA.cy += (center_y - CAMERA.cy) * blend


class SampleReturnConsole(Console):
    """The original vector console with a compact mission/navigation layer."""

    def draw_named_pad(self, env, x: float, label: str, active: bool) -> None:
        ground = env.terrain.height_at(x)
        sx, sy = world_to_screen(x, ground)
        if sx < -50 or sx > VIEW_W + 50 or sy < -30 or sy > VIEW_H + 30:
            return

        color = THEME.signal if active else THEME.struct
        half_w = int(env.cfg.pad_half_w * CAMERA.s)
        half_w = max(8, half_w)
        pygame.draw.line(self.screen, color, (sx - half_w, sy),
                         (sx + half_w, sy), 1)
        for edge, direction in ((sx - half_w, 1), (sx + half_w, -1)):
            pygame.draw.line(self.screen, color, (edge, sy), (edge, sy - 8), 1)
            pygame.draw.line(self.screen, color, (edge, sy),
                             (edge + direction * 8, sy), 1)
        pygame.draw.line(self.screen, color, (sx - 3, sy - 4),
                         (sx + 3, sy - 4), 1)
        pygame.draw.line(self.screen, color, (sx, sy - 7), (sx, sy - 1), 1)
        self.text(self.f_lab, label, color, sx + 12,
                  min(sy + 6, VIEW_H - 18))

    def draw_navigation(self, env) -> None:
        target_x, label = _target_x(env), _target_label(env)
        target_y = env.terrain.height_at(target_x)
        sx, _ = world_to_screen(target_x, target_y)
        dx = target_x - env.state[X]

        # On-screen destinations already have a strong pad marker.  The edge
        # cue appears only when the destination is outside the viewport.
        if 18 <= sx <= VIEW_W - 18:
            return
        right = sx > VIEW_W // 2
        px = VIEW_W - 22 if right else 22
        py = 150
        direction = 1 if right else -1
        points = [(px + direction * 8, py),
                  (px - direction * 5, py - 7),
                  (px - direction * 5, py + 7)]
        pygame.draw.polygon(self.screen, THEME.signal, points, 1)
        text = f"{label}  {abs(dx):.0f} M"
        surf = self.f_lab.render(text, True, THEME.signal)
        tx = px - surf.get_width() - 14 if right else px + 14
        self.screen.blit(surf, (tx, py - surf.get_height() // 2))

    def draw_target_corridor(self, env) -> None:
        """A restrained ILS cone for the active pad only."""
        target_x = _target_x(env)
        target_y = env.terrain.height_at(target_x)
        edge = tuple((THEME.grid[i] + THEME.struct[i]) // 2 for i in range(3))
        angle = math.radians(16.0)
        for sign in (-1.0, 1.0):
            direction = np.array([sign * math.sin(angle), math.cos(angle)])
            distance = 4.0
            while distance < 48.0:
                start = np.array([target_x, target_y]) + direction * distance
                end = start + direction * 1.2
                pygame.draw.line(self.screen, edge,
                                 world_to_screen(*start),
                                 world_to_screen(*end), 1)
                distance += 3.4

    def draw_crater_landmarks(self, env) -> None:
        """Label explicit crater rims when the terrain exposes them."""
        terrain = env.terrain
        candidates = []
        for name in ("left_rim_x", "right_rim_x", "rim_left_x", "rim_right_x"):
            value = getattr(terrain, name, None)
            if value is not None:
                candidates.append(float(value))
        rim_xs = getattr(terrain, "rim_xs", None)
        if rim_xs is not None:
            candidates.extend(float(value) for value in rim_xs)

        # The terrain profile already tells the story; labels are reserved for
        # the crater class so a generic prototype hill is not misidentified.
        if not candidates or "CRATER" not in type(terrain).__name__.upper():
            return
        for i, rim_x in enumerate(dict.fromkeys(candidates)):
            rim_y = terrain.height_at(rim_x)
            sx, sy = world_to_screen(rim_x, rim_y)
            pygame.draw.line(self.screen, THEME.struct,
                             (sx, sy - 4), (sx, sy - 18), 1)
            label = "CRATER RIM" if i == 0 else "RIM"
            self.text(self.f_tiny, label, THEME.struct, sx + 5, sy - 28)

    def _deposit_x(self, env) -> float:
        return float(getattr(env, "sample_deposit_x",
                             env.sample_x + env.cfg.pad_half_w * 0.72))

    def draw_sample_deposit(self, env) -> None:
        """Small vector regolith pile adjacent to the sample pad."""
        x = self._deposit_x(env)
        y = env.terrain.height_at(x)
        color = THEME.struct if _sample_collected(env) else THEME.signal
        points = [world_to_screen(x - 1.2, y),
                  world_to_screen(x - 0.45, y + 0.45),
                  world_to_screen(x, y + 0.18),
                  world_to_screen(x + 0.5, y + 0.65),
                  world_to_screen(x + 1.15, y)]
        pygame.draw.lines(self.screen, color, False, points, 1)
        for ox, oy in ((-0.65, 0.16), (0.05, 0.32), (0.62, 0.14)):
            px, py = world_to_screen(x + ox, y + oy)
            pygame.draw.circle(self.screen, color, (px, py), 1)
        label = "SITE SECURED" if _sample_collected(env) else "REGOLITH"
        sx, sy = world_to_screen(x, y)
        self.text(self.f_tiny, label, color, sx + 8, sy - 20)

    def _draw_world_arrow(self, start, vector, color, label: str,
                          *, dashed: bool = False) -> None:
        start = np.asarray(start, dtype=float)
        vector = np.asarray(vector, dtype=float)
        if float(np.linalg.norm(vector)) < 1e-5:
            return
        a = world_to_screen(*start)
        b = world_to_screen(*(start + vector))
        if dashed:
            for i in range(0, 10, 2):
                p0 = (int(a[0] + (b[0] - a[0]) * i / 10),
                      int(a[1] + (b[1] - a[1]) * i / 10))
                p1 = (int(a[0] + (b[0] - a[0]) * (i + 1) / 10),
                      int(a[1] + (b[1] - a[1]) * (i + 1) / 10))
                pygame.draw.line(self.screen, color, p0, p1, 1)
        else:
            pygame.draw.line(self.screen, color, a, b, 1)

        dx, dy = b[0] - a[0], b[1] - a[1]
        length = max(1.0, math.hypot(dx, dy))
        ux, uy = dx / length, dy / length
        left = (int(b[0] - ux * 6 - uy * 3),
                int(b[1] - uy * 6 + ux * 3))
        right = (int(b[0] - ux * 6 + uy * 3),
                 int(b[1] - uy * 6 - ux * 3))
        pygame.draw.lines(self.screen, color, False, [left, b, right], 1)
        self.text(self.f_tiny, label, color, b[0] + 5, b[1] - 8)

    def draw_guidance_vectors(self, env, action) -> None:
        com = np.asarray(env.state[[X, Y]], dtype=float)
        velocity = np.asarray(env.state[[VX, VY]], dtype=float)
        speed = float(np.linalg.norm(velocity))
        if speed > 1e-4:
            velocity *= min(12.0 / speed, 0.75)
            self._draw_world_arrow(com, velocity, THEME.signal, "V")

        target = np.array([_target_x(env),
                           env.terrain.height_at(_target_x(env))], dtype=float)
        delta = target - com
        distance = float(np.linalg.norm(delta))
        if distance > 1e-4:
            delta *= min(15.0, distance) / distance
            self._draw_world_arrow(com, delta, THEME.struct, "TGT", dashed=True)

        throttle = float(action[0])
        if throttle > 0.01 and env.state[FUEL] > 0.0:
            theta = float(env.state[THETA])
            phi = float(action[1]) * env.cfg.phi_max
            thrust_dir = np.array([-math.sin(theta + phi),
                                   math.cos(theta + phi)])
            body_center = _dry_center(env)
            engine = _engine_point(env, body_center)
            self._draw_world_arrow(engine, thrust_dir * (3.0 + 8.0 * throttle),
                                   THEME.signal, "F")

    def _pod_geometry(self, env, body_center=None):
        pod = _pod_view(env)
        theta = float(env.state[THETA])
        nose, right = _body_basis(theta)
        if body_center is None:
            body_center = _dry_center(env)
        center = body_center + right * pod.offset_x + nose * pod.offset_y

        def point(across: float, along: float) -> np.ndarray:
            return center + right * across + nose * along

        corners = [point(-pod.width / 2.0, -pod.height / 2.0),
                   point(+pod.width / 2.0, -pod.height / 2.0),
                   point(+pod.width / 2.0, +pod.height / 2.0),
                   point(-pod.width / 2.0, +pod.height / 2.0)]
        return pod, center, corners, nose, right

    def _display_com(self, env, body_center: np.ndarray) -> np.ndarray:
        """Interpolate the visible COM marker as regolith fills the pod."""
        if _payload_attached(env):
            return np.asarray(env.state[[X, Y]], dtype=float)
        progress = _sampling_progress(env)
        if progress <= 0.0:
            return body_center
        pod = _pod_view(env)
        mass = pod.mass * progress
        offset = mass / (_dry_mass(env) + mass) * np.array(
            [pod.offset_x, pod.offset_y], dtype=float)
        nose, right = _body_basis(float(env.state[THETA]))
        return body_center + right * offset[0] + nose * offset[1]

    def draw_sampling_animation(self, env, tick: int) -> None:
        if "SAMPLING" not in _phase_name(env):
            return
        body_center = _dry_center(env)
        _, pod_center, _, _, _ = self._pod_geometry(env, body_center)
        source_x = self._deposit_x(env)
        source = np.array([source_x,
                           env.terrain.height_at(source_x) + 0.25])
        # Rendering-only intake hose: it has no collision or dynamics.
        pygame.draw.line(self.screen, THEME.struct,
                         world_to_screen(*source),
                         world_to_screen(*pod_center), 1)
        sx, sy = world_to_screen(*source)
        pygame.draw.circle(self.screen, THEME.signal, (sx, sy), 4, 1)

        progress = _sampling_progress(env)
        for index in range(14):
            travel = (tick * 0.028 + index / 14.0) % 1.0
            if travel > min(1.0, progress + 0.22):
                continue
            point = source * (1.0 - travel) + pod_center * travel
            point[1] += math.sin(math.pi * travel) * 1.3
            px, py = world_to_screen(*point)
            pygame.draw.circle(self.screen, THEME.signal, (px, py),
                               2 if index % 4 == 0 else 1)

    def draw_mission_vehicle(self, env, action, phosphor, tick: int) -> None:
        """Draw dry body, fixed side pod, plume, and combined-COM marker."""
        cfg = env.cfg
        theta = float(env.state[THETA])
        nose, right = _body_basis(theta)
        body_center = _dry_center(env)
        height = float(getattr(getattr(env, "vehicle", None),
                               "body_height", cfg.H))
        half_width = 0.50
        base = body_center - nose * (height / 2.0)
        tip = body_center + nose * (height / 2.0)

        def body_point(along: float, across: float):
            world = base + nose * along + right * across
            return world_to_screen(*world)

        body = [body_point(0.0, -half_width),
                body_point(0.0, +half_width),
                body_point(height * 0.82, +half_width),
                world_to_screen(*tip),
                body_point(height * 0.82, -half_width)]
        pygame.draw.polygon(self.screen, THEME.signal, body, 1)
        pygame.draw.line(self.screen,
                         tuple((THEME.field[i] + THEME.signal[i]) // 2
                               for i in range(3)),
                         world_to_screen(*base), world_to_screen(*tip), 1)
        for sign in (-1.0, 1.0):
            leg_top = body_point(0.70, sign * half_width)
            foot_world = base - nose * 0.15 + right * sign * cfg.leg_half_w
            pygame.draw.line(self.screen, THEME.signal, leg_top,
                             world_to_screen(*foot_world), 1)
        if phosphor:
            phosphor.polygon(body, THEME.signal, 42)

        pod, _, pod_corners, nose, right = self._pod_geometry(env, body_center)
        pod_screen = [world_to_screen(*point) for point in pod_corners]
        pygame.draw.polygon(self.screen, THEME.signal, pod_screen, 1)
        fill = 1.0 if _payload_attached(env) else _sampling_progress(env)
        if fill > 0.0:
            pod_center = body_center + right * pod.offset_x + nose * pod.offset_y
            bottom = -pod.height / 2.0
            top = bottom + pod.height * fill
            fill_world = [pod_center + right * (-pod.width * 0.38) + nose * bottom,
                          pod_center + right * (+pod.width * 0.38) + nose * bottom,
                          pod_center + right * (+pod.width * 0.38) + nose * top,
                          pod_center + right * (-pod.width * 0.38) + nose * top]
            fill_color = tuple(int(THEME.field[i] * 0.45
                                   + THEME.signal[i] * 0.55)
                               for i in range(3))
            pygame.draw.polygon(self.screen, fill_color,
                                [world_to_screen(*point) for point in fill_world])
        else:
            pygame.draw.line(self.screen, THEME.struct,
                             pod_screen[0], pod_screen[2], 1)

        engine = _engine_point(env, body_center)
        throttle = float(action[0])
        phi = float(action[1]) * cfg.phi_max
        exhaust = np.array([math.sin(theta + phi),
                            -math.cos(theta + phi)])
        nozzle = engine + exhaust * 0.75
        pygame.draw.line(self.screen, THEME.signal,
                         world_to_screen(*engine), world_to_screen(*nozzle), 2)
        if throttle > 0.01 and env.state[FUEL] > 0.0:
            flicker = 0.93 + 0.07 * math.sin(tick * 0.73)
            plume_tip = engine + exhaust * (2.5 + 7.0 * throttle) * flicker
            pygame.draw.line(self.screen, THEME.signal,
                             world_to_screen(*nozzle),
                             world_to_screen(*plume_tip), 2)
            if phosphor:
                phosphor.line(world_to_screen(*nozzle),
                              world_to_screen(*plume_tip),
                              THEME.signal, 90, 3)

        com = self._display_com(env, body_center)
        cx, cy = world_to_screen(*com)
        pygame.draw.circle(self.screen, THEME.signal, (cx, cy), 4, 1)
        pygame.draw.line(self.screen, THEME.signal, (cx - 6, cy), (cx + 6, cy), 1)
        pygame.draw.line(self.screen, THEME.signal, (cx, cy - 6), (cx, cy + 6), 1)

    def draw_debug_overlay(self, env, action) -> None:
        body_center = _dry_center(env)
        engine = _engine_point(env, body_center)
        _, payload_center, _, _, _ = self._pod_geometry(env, body_center)
        combined = self._display_com(env, body_center)

        markers = ((combined, "COM", THEME.signal),
                   (body_center, "BODY", THEME.struct),
                   (engine, "ENGINE", THEME.struct),
                   (payload_center, "POD", THEME.struct))
        for point, label, color in markers:
            px, py = world_to_screen(*point)
            pygame.draw.line(self.screen, color, (px - 3, py - 3),
                             (px + 3, py + 3), 1)
            pygame.draw.line(self.screen, color, (px - 3, py + 3),
                             (px + 3, py - 3), 1)
            self.text(self.f_tiny, label, color, px + 6, py - 7)

        theta = float(env.state[THETA])
        phi = float(action[1]) * env.cfg.phi_max
        force = np.array([-math.sin(theta + phi), math.cos(theta + phi)])
        lever = engine - combined
        torque = lever[0] * force[1] - lever[1] * force[0]
        sign = "+" if torque > 1e-6 else "-" if torque < -1e-6 else "0"
        status = "ARMED" if _contact_armed(env) else "DISARMED"
        text = f"CONTACT {status}  TORQUE {sign}"
        self.text(self.f_lab, text, THEME.struct, 18, 139)
        tx = _target_x(env)
        ty = env.terrain.height_at(tx)
        px, py = world_to_screen(tx, ty)
        pygame.draw.circle(self.screen, THEME.struct, (px, py), 10, 1)

    def draw_mission_hud(self, env, info: dict, ended: bool) -> None:
        phase = _phase_name(env).replace("_", " ")
        target_label = _target_label(env)
        dx = _target_x(env) - env.state[X]
        sampling = "SAMPLING" in phase
        attached = _payload_attached(env)
        fill = 1.0 if attached else _sampling_progress(env)
        pod = _pod_view(env)
        shown_mass = pod.mass * fill

        if ended:
            outcome = str(info.get("outcome") or "TIMEOUT")
            if "RETURN" in outcome or "COMPLETE" in outcome:
                prompt = "MISSION COMPLETE  /  R REPLAY"
            else:
                prompt = "MISSION LOST  /  R REPLAY"
        elif sampling:
            prompt = f"SAMPLE ACQUISITION  {fill * 100:3.0f}%"
        elif getattr(env, "grounded_pad", None) is not None:
            if (not hasattr(env, "mission_state")
                    and hasattr(env, "launch_armed")
                    and not env.launch_armed):
                prompt = "CUT THROTTLE [S] TO ARM LAUNCH"
            else:
                prompt = "W THROTTLE TO LAUNCH"
        else:
            prompt = f"LAND AT {target_label}  /  RANGE {abs(dx):.0f} M"

        rect = pygame.Rect(16, 16, 460, 112)
        pygame.draw.rect(self.screen, THEME.field, rect)
        pygame.draw.rect(self.screen, THEME.struct, rect, 1)
        self.text(self.f_lab, "AUTONOMOUS LUNAR SAMPLE RETURN",
                  THEME.struct, 28, 25)
        self.text(self.f_num, phase, THEME.signal, 28, 46)
        self.text(self.f_lab, f"TARGET {target_label}", THEME.struct, 300, 49)
        status = "LOCKED" if attached else "FILLING" if sampling else "EMPTY"
        color = THEME.signal if (attached or sampling) else THEME.struct
        self.text(self.f_lab,
                  f"SIDE POD {status}   {shown_mass:.2f} / {pod.mass:.2f} KG",
                  color, 28, 70)
        self.bar(298, 72, 158, 9, fill, color)
        self.text(self.f_lab, prompt, THEME.signal, 28, 96)

    def draw_mission_controls(self) -> None:
        # Replace play.py's map/pilot shortcuts with the smaller mission set.
        x0 = VIEW_W + 18
        pygame.draw.rect(self.screen, THEME.field,
                         (VIEW_W + 1, WIN_H - 47, WIN_W - VIEW_W - 2, 46))
        self.text(self.f_lab, "W THROTTLE   A/D STEER   S CUT   SPACE MAX",
                  THEME.struct, x0, WIN_H - 40)
        self.text(self.f_lab, "G SAS   F3 DEBUG   C CRT   V RAYS   R REPLAY",
                  THEME.struct, x0, WIN_H - 22)


def _success_stamp(info: dict) -> tuple[str, bool]:
    outcome = str(info.get("outcome") or "TIMEOUT")
    if "RETURN" in outcome or "COMPLETE" in outcome:
        return "SAMPLE RETURN COMPLETE - SCIENCE SECURED", False
    if outcome == "TIMEOUT":
        return "MISSION TIMEOUT - SAMPLE LOST", True
    return f"MISSION LOST - {outcome}", True


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fly a seeded lunar sample-return mission.")
    parser.add_argument("--seed", type=int, default=42,
                        help="deterministic mission seed (default: 42)")
    parser.add_argument("--controller", "--pilot", dest="controller",
                        choices=("human", "scripted"),
                        default="human",
                        help="controller mode (default: human)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    # Keep vector text sharp when Windows display scaling is above 100%.
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"ROCKET GNC - SAMPLE RETURN / SEED {args.seed}")
    clock = pygame.time.Clock()
    console = SampleReturnConsole(screen)
    phosphor = Phosphor((VIEW_W, VIEW_H))
    debris = Debris()
    scanlines = make_scanlines(WIN_W, WIN_H)

    env = SampleReturnEnv()
    fc = FlightComputer()
    charts = [StripChart("ALT M", 0.0, 100.0),
              StripChart("VY M/S", -25.0, 25.0, zero_line=True)]
    log = EventLog()

    crt_on = True
    show_rays = False
    show_debug = False
    trail = deque(maxlen=3600)
    info: dict = {}
    observation = None
    action = np.zeros(2, dtype=np.float32)
    ep_reward = t_sim = 0.0
    ended = False
    stamp: tuple[str, bool] | None = None
    fuel_marks: set[int] = set()
    tick = 0

    def reset_mission() -> None:
        nonlocal info, observation, trail, action, ep_reward, t_sim
        nonlocal ended, stamp, fuel_marks, tick
        observation, info = env.reset(seed=args.seed)
        fc.reset()
        for chart in charts:
            chart.reset()
        log.reset()
        phosphor.clear()
        debris.clear()
        trail = deque(maxlen=3600)
        action = np.zeros(2, dtype=np.float32)
        ep_reward = t_sim = 0.0
        ended = False
        stamp = None
        fuel_marks = set()
        tick = 0
        update_mission_camera(env, env.cfg.dt, snap=True)
        log.post(0.0, f"MISSION ACTIVE - SEED {args.seed}")
        log.post(0.0, f"CONTROLLER {args.controller.upper()}")
        log.post(0.0, "DESTINATION SAMPLE")

    reset_mission()
    running = True
    while running:
        dt = env.cfg.dt
        tick += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset_mission()
                elif event.key == pygame.K_c:
                    crt_on = not crt_on
                elif event.key == pygame.K_v:
                    show_rays = not show_rays
                elif event.key == pygame.K_F3:
                    show_debug = not show_debug
                    log.post(t_sim, f"DEBUG {'ON' if show_debug else 'OFF'}")
                elif event.key == pygame.K_g:
                    fc.cycle_mode()
                    log.post(t_sim, f"SAS {fc.mode}")

        if not ended:
            keys = pygame.key.get_pressed()
            was_grounded = _is_grounded(env)
            had_payload = _payload_attached(env)
            old_phase = _phase_name(env)
            old_target = _target_label(env)
            if args.controller == "scripted":
                action = scripted_sample_return_action(env).astype(np.float32)
            else:
                fc.update(keys, env.state, dt)
                action = fc.action()
            observation, reward, terminated, truncated, next_info = env.step(action)
            ep_reward += float(reward)
            t_sim += dt
            if not _is_grounded(env):
                trail.append((float(env.state[X]), float(env.state[Y])))
            body_center = _dry_center(env)
            nose, _ = _body_basis(float(env.state[THETA]))
            body_height = float(getattr(getattr(env, "vehicle", None),
                                        "body_height", env.cfg.H))
            body_base = body_center - nose * body_height / 2.0
            charts[0].push(body_base[1]
                           - env.terrain.height_at(float(body_base[0])))
            charts[1].push(env.state[VY])

            now_grounded = _is_grounded(env)
            has_payload = _payload_attached(env)
            new_phase = _phase_name(env)
            new_target = _target_label(env)
            if was_grounded and not now_grounded:
                log.post(t_sim, f"LIFTOFF - TARGET {new_target}")
            if old_phase != new_phase and "SAMPLING" in new_phase:
                log.post(t_sim, "SAMPLE PAD TOUCHDOWN")
                log.post(t_sim, "SAMPLE ACQUISITION")
            if has_payload and not had_payload:
                pod = _pod_view(env)
                log.post(t_sim, f"PAYLOAD LOCKED - {pod.mass:.2f} KG")
                log.post(t_sim, "RETURN GUIDANCE ACTIVE")
            if old_target != new_target:
                log.post(t_sim, f"DESTINATION {new_target}")

            fuel_frac = env.state[FUEL] / env.cfg.fuel_0
            for mark in (50, 25):
                if fuel_frac * 100 <= mark and mark not in fuel_marks:
                    fuel_marks.add(mark)
                    log.post(t_sim, f"FUEL {mark}%")
            if fuel_frac <= 0.0 and 0 not in fuel_marks:
                fuel_marks.add(0)
                log.post(t_sim, "FUEL DEPLETED", fault=True)

            info = next_info
            if terminated or truncated:
                ended = True
                stamp = _success_stamp(info)
                log.post(t_sim, stamp[0], fault=stamp[1])
                if stamp[1]:
                    debris.burst(float(env.state[X]),
                                 max(float(env.state[Y] - env.cfg.L), 0.2),
                                 math.hypot(float(env.state[VX]),
                                            float(env.state[VY])))
        else:
            action = np.zeros(2, dtype=np.float32)

        update_mission_camera(env, dt)
        screen.fill(THEME.field)
        # Zoomed crater vertices can project beyond the world viewport.  Keep
        # every world primitive out of the fixed telemetry panel.
        screen.set_clip(pygame.Rect(0, 0, VIEW_W, VIEW_H))
        console.draw_graticule(env.cfg)
        console.draw_target_corridor(env)
        console.draw_terrain(env)
        console.draw_crater_landmarks(env)
        console.draw_sample_deposit(env)
        target_label = _target_label(env)
        console.draw_named_pad(env, env.base_x, "BASE",
                               target_label == "BASE")
        console.draw_named_pad(env, env.sample_x, "SAMPLE",
                               target_label == "SAMPLE")
        console.draw_trail(list(trail))
        if show_rays and not ended:
            console.draw_rays(env)
        if crt_on:
            phosphor.decay()
        ph = phosphor if crt_on else None
        console.draw_sampling_animation(env, tick)
        console.draw_mission_vehicle(env, action, ph, tick)
        console.draw_guidance_vectors(env, action)
        debris.update_and_draw(screen, ph, dt, env.cfg.g)
        if crt_on:
            screen.blit(phosphor.surf, (0, 0))

        console.draw_navigation(env)
        if show_debug:
            console.draw_debug_overlay(env, action)
        console.draw_mission_hud(env, info, ended)
        screen.set_clip(None)
        console.draw_panel(
            env, action, ep_reward, t_sim, args.seed,
            env.state[FUEL] <= 0.0, charts, log,
            "ORACLE" if args.controller == "scripted"
            else f"SAS {fc.mode}",
            "CRATER RETURN",
        )
        console.draw_mission_controls()
        if stamp is not None:
            console.draw_stamp(*stamp)
        if crt_on:
            screen.blit(scanlines, (0, 0))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
