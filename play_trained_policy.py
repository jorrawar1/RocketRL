"""Visualize the trained recurrent policy from an exact airborne start.

Controls
  LEFT / RIGHT   decrease / increase horizontal distance by 0.25 m
  DOWN / UP      decrease / increase altitude by 0.25 m
  SHIFT          use a 1.0 m adjustment with an arrow key
  TAB            switch between dry outbound and loaded return flight
  R              replay the same start and terrain seed
  N              advance to the next terrain seed
  SPACE          pause / resume
  .              advance one policy decision while paused
  V              toggle terrain rays
  F3             toggle vehicle debug geometry
  C              toggle CRT effects
  ESC            quit
"""

from __future__ import annotations

import argparse
import ctypes
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pygame
import torch

from play import (
    CAMERA,
    SCALE,
    THEME,
    VIEW_H,
    VIEW_W,
    WIN_H,
    WIN_W,
    Debris,
    EventLog,
    Phosphor,
    StripChart,
    make_scanlines,
    world_to_screen,
)
from play_sample_return import (
    SampleReturnConsole,
    _body_basis,
    _dry_center,
    _payload_attached,
    _phase_name,
    _target_label,
    _target_x,
)
from rocketenv.physics import FUEL, OMEGA, THETA, VX, VY, X, Y
from rocketenv.sample_return import (
    ACTOR_OBSERVATION_DIM,
    TrainingTask,
    make_training_env,
)
from rocketenv.sample_return.ppo import Actor


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "ppo_sample_return_overnight_final.pt"
TASKS = {
    "outbound": TrainingTask.OUTBOUND_LEG,
    "return": TrainingTask.RETURN_LEG,
}


class RecurrentPolicy:
    """The deterministic GRU action path used by policy evaluation."""

    def __init__(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        observation_dim = int(checkpoint["actor_observation_dim"])
        hidden_width = int(checkpoint["hidden_width"])
        action_dim = int(checkpoint["action_dim"])

        if observation_dim != ACTOR_OBSERVATION_DIM:
            raise ValueError(
                f"checkpoint expects {observation_dim} actor observations, "
                f"but the environment currently provides {ACTOR_OBSERVATION_DIM}"
            )
        if action_dim != 2:
            raise ValueError(f"checkpoint action dimension must be 2, got {action_dim}")

        self.actor = Actor(observation_dim, hidden_width, action_dim)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor.eval()
        self.action_dim = action_dim
        self.hidden_state = self.actor.initial_hidden_state(1)
        self.previous_action = torch.zeros((1, action_dim), dtype=torch.float32)
        self.metadata = checkpoint

    def reset(self) -> None:
        self.hidden_state = self.actor.initial_hidden_state(1)
        self.previous_action = torch.zeros(
            (1, self.action_dim), dtype=torch.float32
        )

    @torch.inference_mode()
    def act(self, observation: np.ndarray) -> np.ndarray:
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32
        ).unsqueeze(0)
        mean, _, self.hidden_state = self.actor.distribution_parameters(
            observation_tensor, self.previous_action, self.hidden_state
        )
        squashed = torch.tanh(mean)
        action_tensor = torch.stack(
            ((squashed[..., 0] + 1.0) / 2.0, squashed[..., 1]), dim=-1
        )
        self.previous_action = action_tensor
        return action_tensor.squeeze(0).numpy().astype(np.float32, copy=False)


def update_local_camera(env, start_position: np.ndarray, dt: float, *, snap=False) -> None:
    """Fit the active pad, requested start, and current vehicle in one view."""

    target_x = _target_x(env)
    current = _dry_center(env)
    xs = np.asarray([target_x, start_position[0], current[0]], dtype=float)
    left = max(0.0, float(xs.min()) - 7.0)
    right = min(float(env.cfg.world_w), float(xs.max()) + 7.0)

    terrain_x = np.linspace(left, right, 33)
    terrain_y = np.asarray([env.terrain.height_at(x) for x in terrain_x])
    bottom = float(terrain_y.min()) - 3.0
    top = max(float(terrain_y.max()), float(start_position[1]), float(current[1])) + 6.0

    horizontal_span = max(18.0, right - left)
    vertical_span = max(18.0, top - bottom)
    scale = min(
        SCALE * 2.5,
        (VIEW_W - 64.0) / horizontal_span,
        (VIEW_H - 80.0) / vertical_span,
    )

    blend = 1.0 if snap else min(1.0, 3.0 * dt)
    CAMERA.s += (scale - CAMERA.s) * blend
    CAMERA.cx += ((left + right) / 2.0 - CAMERA.cx) * blend
    CAMERA.cy += ((bottom + top) / 2.0 - CAMERA.cy) * blend


def draw_cross(console, point: np.ndarray, label: str, color) -> None:
    x, y = world_to_screen(float(point[0]), float(point[1]))
    pygame.draw.line(console.screen, color, (x - 5, y - 5), (x + 5, y + 5), 1)
    pygame.draw.line(console.screen, color, (x - 5, y + 5), (x + 5, y - 5), 1)
    console.text(console.f_tiny, label, color, x + 8, y - 7)


def draw_policy_hud(
    console,
    env,
    info: dict,
    task: TrainingTask,
    seed: int,
    distance: float,
    altitude: float,
    accepted_distance: float,
    accepted_altitude: float,
    paused: bool,
    ended: bool,
) -> None:
    loaded = task == TrainingTask.RETURN_LEG
    task_name = "RETURN LEG / LOADED" if loaded else "OUTBOUND LEG / DRY"
    inside = distance <= accepted_distance and altitude <= accepted_altitude

    if ended and info.get("task_success", False):
        status = "POLICY SUCCESS"
        status_color = THEME.signal
    elif ended:
        status = str(info.get("outcome") or "EPISODE FAILED").replace("_", " ")
        status_color = THEME.fault
    elif paused:
        status = "PAUSED"
        status_color = THEME.struct
    else:
        status = "POLICY RUNNING"
        status_color = THEME.signal

    body_x = float(_dry_center(env)[0])
    target_error = body_x - _target_x(env)
    edge_miss = max(0.0, abs(target_error) - float(env.cfg.pad_half_w))
    region = (
        f"MASTERED D<= {accepted_distance:.1f}  H<= {accepted_altitude:.1f} M"
        if inside
        else f"OUTSIDE MASTERED D<= {accepted_distance:.1f}  H<= {accepted_altitude:.1f} M"
    )
    region_color = THEME.struct if inside else THEME.fault

    rect = pygame.Rect(16, 16, 600, 126)
    pygame.draw.rect(console.screen, THEME.field, rect)
    pygame.draw.rect(console.screen, THEME.struct, rect, 1)
    console.text(console.f_lab, "RECURRENT POLICY START-POSITION VIEWER", THEME.struct, 28, 25)
    console.text(console.f_num, task_name, THEME.signal, 28, 47)
    console.text(console.f_lab, f"SEED {seed}", THEME.struct, 420, 51)
    console.text(
        console.f_lab,
        f"START DISTANCE {distance:5.2f} M   TERRAIN CLEARANCE {altitude:5.2f} M",
        THEME.signal,
        28,
        73,
    )
    console.text(console.f_lab, region, region_color, 28, 95)
    console.text(
        console.f_lab,
        f"{status}   TARGET {_target_label(env)}   DX {target_error:+6.2f} M"
        f"   EDGE MISS {edge_miss:4.2f} M",
        status_color,
        28,
        117,
    )


def draw_policy_controls(console) -> None:
    x = VIEW_W + 14
    y = WIN_H - 64
    pygame.draw.rect(
        console.screen,
        THEME.field,
        (VIEW_W + 1, y - 2, WIN_W - VIEW_W - 2, 65),
    )
    console.text(
        console.f_tiny,
        "LEFT/RIGHT DISTANCE   UP/DOWN ALTITUDE   SHIFT = 1 M",
        THEME.struct,
        x,
        y,
    )
    console.text(
        console.f_tiny,
        "TAB TASK   R REPLAY   N NEXT SEED   SPACE PAUSE   . STEP",
        THEME.struct,
        x,
        y + 18,
    )
    console.text(
        console.f_tiny,
        "V RAYS   F3 DEBUG   C CRT   ESC QUIT",
        THEME.struct,
        x,
        y + 36,
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Watch the trained recurrent policy from an adjustable start."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task", choices=tuple(TASKS), default="return")
    parser.add_argument("--distance", type=float, default=5.0)
    parser.add_argument("--altitude", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"policy checkpoint not found: {checkpoint_path}")
    if args.distance < 0.0 or args.altitude < 0.0:
        raise ValueError("distance and altitude must be nonnegative")

    policy = RecurrentPolicy(checkpoint_path)
    accepted_distance = float(policy.metadata.get("accepted_distance", 0.0))
    accepted_altitude = float(policy.metadata.get("accepted_altitude", 0.0))
    maximum_distance = float(policy.metadata.get("target_distance", 75.0))
    maximum_altitude = float(policy.metadata.get("target_altitude", 18.0))

    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("ROCKET GNC - TRAINED GRU POLICY")
    clock = pygame.time.Clock()
    console = SampleReturnConsole(screen)
    phosphor = Phosphor((VIEW_W, VIEW_H))
    debris = Debris()
    scanlines = make_scanlines(WIN_W, WIN_H)
    charts = [
        StripChart("ALT M", 0.0, max(20.0, maximum_altitude + 2.0)),
        StripChart("VY M/S", -25.0, 25.0, zero_line=True),
    ]
    log = EventLog()

    task = TASKS[args.task]
    seed = args.seed
    distance = min(float(args.distance), maximum_distance)
    altitude = min(float(args.altitude), maximum_altitude)
    env = make_training_env(task=task, action_repeat=4)

    observation = np.empty(ACTOR_OBSERVATION_DIM, dtype=np.float32)
    info: dict = {}
    action = np.zeros(2, dtype=np.float32)
    start_position = np.zeros(2, dtype=float)
    trail = deque(maxlen=3600)
    ep_reward = 0.0
    t_sim = 0.0
    tick = 0
    ended = False
    paused = False
    step_once = False
    stamp: tuple[str, bool] | None = None
    crt_on = True
    show_rays = False
    show_debug = False
    frame_count = 0

    def reset_episode() -> None:
        nonlocal observation, info, action, start_position, trail
        nonlocal ep_reward, t_sim, tick, ended, step_once, stamp

        observation, info = env.reset(
            seed=seed,
            options={
                "task": task,
                "spawn_distance_from_target": distance,
                "spawn_altitude": altitude,
            },
        )
        policy.reset()
        action = np.zeros(2, dtype=np.float32)
        start_position = _dry_center(env.mission_env).copy()
        trail = deque([tuple(start_position)], maxlen=3600)
        ep_reward = 0.0
        t_sim = 0.0
        tick = 0
        ended = False
        step_once = False
        stamp = None
        for chart in charts:
            chart.reset()
        log.reset()
        phosphor.clear()
        debris.clear()
        update_local_camera(env.mission_env, start_position, env.mission_env.cfg.dt, snap=True)
        mode = "LOADED RETURN" if task == TrainingTask.RETURN_LEG else "DRY OUTBOUND"
        log.post(0.0, f"{mode} - SEED {seed}")
        log.post(0.0, f"START D {distance:.2f} H {altitude:.2f}")

    reset_episode()

    try:
        running = True
        while running:
            frame_count += 1
            tick += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    modifier_step = 1.0 if event.mod & pygame.KMOD_SHIFT else 0.25
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        reset_episode()
                    elif event.key == pygame.K_n:
                        seed += 1
                        reset_episode()
                    elif event.key == pygame.K_TAB:
                        task = (
                            TrainingTask.RETURN_LEG
                            if task == TrainingTask.OUTBOUND_LEG
                            else TrainingTask.OUTBOUND_LEG
                        )
                        reset_episode()
                    elif event.key == pygame.K_LEFT:
                        distance = max(0.0, distance - modifier_step)
                        reset_episode()
                    elif event.key == pygame.K_RIGHT:
                        distance = min(maximum_distance, distance + modifier_step)
                        reset_episode()
                    elif event.key == pygame.K_DOWN:
                        altitude = max(0.0, altitude - modifier_step)
                        reset_episode()
                    elif event.key == pygame.K_UP:
                        altitude = min(maximum_altitude, altitude + modifier_step)
                        reset_episode()
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_PERIOD:
                        step_once = True
                    elif event.key == pygame.K_v:
                        show_rays = not show_rays
                    elif event.key == pygame.K_c:
                        crt_on = not crt_on
                    elif event.key == pygame.K_F3:
                        show_debug = not show_debug

            stepped = False
            if not ended and (not paused or step_once):
                action = policy.act(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                physics_steps = int(info["physics_steps_this_decision"])
                t_sim += physics_steps * env.mission_env.cfg.dt
                ep_reward += float(reward)
                stepped = True
                step_once = False

                body_center = _dry_center(env.mission_env)
                trail.append(tuple(body_center))
                nose, _ = _body_basis(float(env.mission_env.state[THETA]))
                body_height = float(env.mission_env.vehicle.body_height)
                body_base = body_center - nose * body_height / 2.0
                clearance = body_base[1] - env.mission_env.terrain.height_at(
                    float(body_base[0])
                )
                charts[0].push(clearance)
                charts[1].push(env.mission_env.state[VY])

                if info["auto_advanced_sampling_steps"] > 0:
                    policy.reset()

                if terminated or truncated:
                    ended = True
                    success = bool(info["task_success"])
                    outcome = str(info.get("outcome") or "TARGET PAD REACHED")
                    if success:
                        stamp = ("TARGET PAD REACHED - POLICY SUCCESS", False)
                    elif info.get("decision_timeout", False):
                        stamp = ("POLICY DEADLINE - NO LANDING", True)
                    else:
                        stamp = (f"POLICY FAILED - {outcome.replace('_', ' ')}", True)
                    log.post(t_sim, stamp[0], fault=stamp[1])
                    if not success:
                        debris.burst(
                            float(env.mission_env.state[X]),
                            max(float(env.mission_env.state[Y] - env.mission_env.cfg.L), 0.2),
                            math.hypot(
                                float(env.mission_env.state[VX]),
                                float(env.mission_env.state[VY]),
                            ),
                        )

            raw_env = env.mission_env
            camera_dt = (
                int(info.get("physics_steps_this_decision", 4)) * raw_env.cfg.dt
                if stepped
                else raw_env.cfg.dt
            )
            update_local_camera(raw_env, start_position, camera_dt)
            screen.fill(THEME.field)
            screen.set_clip(pygame.Rect(0, 0, VIEW_W, VIEW_H))
            console.draw_graticule(raw_env.cfg)
            console.draw_target_corridor(raw_env)
            console.draw_terrain(raw_env)
            console.draw_crater_landmarks(raw_env)
            console.draw_sample_deposit(raw_env)
            target_label = _target_label(raw_env)
            console.draw_named_pad(
                raw_env, raw_env.base_x, "BASE", target_label == "BASE"
            )
            console.draw_named_pad(
                raw_env, raw_env.sample_x, "SAMPLE", target_label == "SAMPLE"
            )
            console.draw_trail(list(trail))
            draw_cross(console, start_position, "START", THEME.struct)
            if ended:
                end_color = THEME.signal if info.get("task_success", False) else THEME.fault
                draw_cross(console, _dry_center(raw_env), "END", end_color)
            if show_rays and not ended:
                console.draw_rays(raw_env)
            if crt_on:
                phosphor.decay()
            phosphor_layer = phosphor if crt_on else None
            console.draw_mission_vehicle(raw_env, action, phosphor_layer, tick)
            console.draw_guidance_vectors(raw_env, action)
            debris.update_and_draw(
                screen, phosphor_layer, raw_env.cfg.dt * 4.0, raw_env.cfg.g
            )
            if crt_on:
                screen.blit(phosphor.surf, (0, 0))
            if show_debug:
                console.draw_debug_overlay(raw_env, action)
            draw_policy_hud(
                console,
                raw_env,
                info,
                task,
                seed,
                distance,
                altitude,
                accepted_distance,
                accepted_altitude,
                paused,
                ended,
            )
            screen.set_clip(None)
            console.draw_panel(
                raw_env,
                action,
                ep_reward,
                t_sim,
                seed,
                raw_env.state[FUEL] <= 0.0,
                charts,
                log,
                "GRU MEAN POLICY",
                f"D {distance:.2f} / H {altitude:.2f}",
            )
            draw_policy_controls(console)
            if stamp is not None:
                console.draw_stamp(*stamp)
            if crt_on:
                screen.blit(scanlines, (0, 0))
            pygame.display.flip()

            if args.max_frames and frame_count >= args.max_frames:
                running = False
            clock.tick(15)
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
