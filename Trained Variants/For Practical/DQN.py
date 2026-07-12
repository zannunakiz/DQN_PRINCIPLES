# ====================================================
# DQN.py - Core Logic, Environment, Agent, Rendering
# ====================================================

import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
import csv
import json
import os
import time
from datetime import datetime
from constant import *

try:
    pygame.font.init()
    ALL_FONT = pygame.font.Font(None, 16)
    ALL_FONT_COLOR = (128, 128, 128)
except Exception:
    ALL_FONT = None
    ALL_FONT_COLOR = (128, 128, 128)


# ====================================================
# Dynamic Obstacle Class
# ====================================================
class DynamicObstacle:
    def __init__(self, data):
        self.x = data["x"]
        self.y = data["y"]
        self.width = data["width"]
        self.height = data["height"]
        self.start_x = data.get("start_x", data["x"])
        self.start_y = data.get("start_y", data["y"])
        self.end_x = data.get("end_x", data["x"] + 100)
        self.end_y = data.get("end_y", data["y"])
        self.seconds = data.get("seconds", 2.0)
        self.progress = 0.0
        self.direction = 1
        self.is_dynamic = data.get("is_dynamic", False)
        self.moving = True
        self.last_update = time.time()

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self):
        if not self.is_dynamic or not self.moving:
            return

        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time

        if dt > 0.1:
            dt = 0.1

        total_distance = math.hypot(
            self.end_x - self.start_x, self.end_y - self.start_y
        )
        if total_distance == 0:
            return

        speed = total_distance / self.seconds
        distance_to_move = speed * dt
        self.progress += (distance_to_move / total_distance) * self.direction

        if self.progress >= 1.0:
            self.progress = 1.0
            self.direction = -1
        elif self.progress <= 0.0:
            self.progress = 0.0
            self.direction = 1

        self.x = self.start_x + (self.end_x - self.start_x) * self.progress
        self.y = self.start_y + (self.end_y - self.start_y) * self.progress


# ====================================================
# Environment
# ====================================================
class Environment:
    def __init__(self, obstacle_file=None):
        self.dynamic_indices = []
        self.instant_move = False  # Set True for Headless, False for UI

        self.obstacle_rects, self.dynamic_obstacles = self.load_obstacles(obstacle_file)

        self.wall_rects = [
            pygame.Rect(
                ARENA_X - WALL_THICKNESS,
                ARENA_Y + WALL_HEIGHT,
                WALL_WIDTH + 2 * WALL_THICKNESS,
                WALL_THICKNESS,
            ),
            pygame.Rect(
                ARENA_X - WALL_THICKNESS,
                ARENA_Y - WALL_THICKNESS,
                WALL_THICKNESS,
                WALL_HEIGHT + 2 * WALL_THICKNESS,
            ),
            pygame.Rect(
                ARENA_X + WALL_WIDTH,
                ARENA_Y - WALL_THICKNESS,
                WALL_THICKNESS,
                WALL_HEIGHT + 2 * WALL_THICKNESS,
            ),
        ]

        self.all_edges = self.calculate_edges()
        self.reset()

    def _load_default_obstacles(self, obstacle_rects):
        obs_x = globals().get("OBS_X", globals().get("DEFAULT_OBS_X", 200))
        obs_y = globals().get("OBS_Y", globals().get("DEFAULT_OBS_Y", 200))
        obs_size = globals().get("OBS_SIZE", globals().get("DEFAULT_OBS_SIZE", 50))

        rect = pygame.Rect(obs_x, obs_y, obs_size, obs_size)
        obstacle_rects.append(rect)
        self.obstacle_data = [
            {
                "x": obs_x,
                "y": obs_y,
                "width": obs_size,
                "height": obs_size,
                "name": "Default",
            }
        ]
        print("Using default obstacle")

    def load_obstacles(self, obstacle_file):
        obstacle_rects = []
        dynamic_obstacles = []
        self.dynamic_indices = []

        if obstacle_file and os.path.exists(obstacle_file):
            try:
                with open(obstacle_file, "r") as f:
                    data = json.load(f)

                if "obstacles" in data and data["obstacles"]:
                    for obs in data["obstacles"]:
                        if obs.get("is_dynamic", False):
                            dyn = DynamicObstacle(obs)
                            dynamic_obstacles.append(dyn)
                            self.dynamic_indices.append(len(obstacle_rects))
                            obstacle_rects.append(dyn.get_rect())
                        else:
                            rect = pygame.Rect(
                                obs["x"], obs["y"], obs["width"], obs["height"]
                            )
                            obstacle_rects.append(rect)

                    print(
                        f"Loaded {len(obstacle_rects)} obstacles ({len(dynamic_obstacles)} dynamic)"
                    )
                    self.obstacle_data = data["obstacles"]
                else:
                    self._load_default_obstacles(obstacle_rects)
            except Exception as e:
                print(f"Error loading obstacles: {e}")
                self._load_default_obstacles(obstacle_rects)
        else:
            self._load_default_obstacles(obstacle_rects)

        return obstacle_rects, dynamic_obstacles

    def calculate_edges(self):
        edges = []
        for rect in self.obstacle_rects:
            edges.extend(
                [
                    ((rect.left, rect.top), (rect.right, rect.top)),
                    ((rect.right, rect.top), (rect.right, rect.bottom)),
                    ((rect.right, rect.bottom), (rect.left, rect.bottom)),
                    ((rect.left, rect.bottom), (rect.left, rect.top)),
                ]
            )

        edges.extend(
            [
                (
                    (ARENA_X, ARENA_Y + WALL_HEIGHT),
                    (ARENA_X + WALL_WIDTH, ARENA_Y + WALL_HEIGHT),
                ),
                ((ARENA_X, ARENA_Y), (ARENA_X, ARENA_Y + WALL_HEIGHT)),
                (
                    (ARENA_X + WALL_WIDTH, ARENA_Y),
                    (ARENA_X + WALL_WIDTH, ARENA_Y + WALL_HEIGHT),
                ),
            ]
        )
        return edges

    def reset(self):
        self.agent_x = START_X
        self.agent_y = START_Y
        self.agent_angle = 0.0
        self.steps = 0
        self.done = False
        self.termination_reason = "Max Steps"
        self.success = False
        self.collision = False
        self.turning = False
        self.turn_direction = 0
        self.turn_progress = 0
        self.moving = False
        self.move_direction = 0
        self.move_progress = 0
        self.pending_action = None
        self.prev_action = None

        for dyn in self.dynamic_obstacles:
            dyn.progress = 0.0
            dyn.direction = 1
            dyn.x = dyn.start_x
            dyn.y = dyn.start_y
            dyn.moving = True
            dyn.last_update = time.time()

        for i, dyn in enumerate(self.dynamic_obstacles):
            if i < len(self.dynamic_indices):
                self.obstacle_rects[self.dynamic_indices[i]] = dyn.get_rect()

        self.all_edges = self.calculate_edges()
        return self.get_state()

    def step(self, action=None):
        if self.done:
            return self.get_state(), 0, True, {}

        # Update Dynamic Obstacles
        for i, dyn in enumerate(self.dynamic_obstacles):
            dyn.update()
            if i < len(self.dynamic_indices):
                self.obstacle_rects[self.dynamic_indices[i]] = dyn.get_rect()

        self.all_edges = self.calculate_edges()
        current_sensors = get_sensor_values(
            self.agent_x, self.agent_y, self.agent_angle, self.all_edges
        )
        center_sensor_clear = current_sensors[2] >= 0.99

        reward = 0
        info = {
            "success": self.success,
            "collision": self.collision,
            "termination_reason": self.termination_reason,
            "obstacle_count": len(self.obstacle_rects),
            "step_completed": False,
            "action": action,
        }

        if self.instant_move:
            # HEADLESS MODE: Instant execution
            if action is not None:
                prev_action_before = self.prev_action
                if action == 0:
                    self.agent_x += math.sin(math.radians(self.agent_angle)) * MOVE_DIST
                    self.agent_y -= math.cos(math.radians(self.agent_angle)) * MOVE_DIST
                    reward = STEP_PENALTY
                    if center_sensor_clear:
                        reward += ASCENT_REWARD
                elif action == 1:
                    if self.agent_angle <= MIN_ANGLE:
                        reward = STEP_PENALTY + TURN_LIMIT_PENALTY
                    else:
                        self.agent_angle -= TURN_ANGLE
                        reward = STEP_PENALTY
                elif action == 2:
                    if self.agent_angle >= MAX_ANGLE:
                        reward = STEP_PENALTY + TURN_LIMIT_PENALTY
                    else:
                        self.agent_angle += TURN_ANGLE
                        reward = STEP_PENALTY

                if (
                    prev_action_before in [1, 2]
                    and action in [1, 2]
                    and action != prev_action_before
                ):
                    reward += REPEAT_TURN_PENALTY

                self.prev_action = action
                self.agent_angle = max(MIN_ANGLE, min(MAX_ANGLE, self.agent_angle))
                self.steps += 1
                info["step_completed"] = True
                info["action"] = action
        else:
            # UI MODE: Smooth sub-stepping
            if self.turning:
                remaining = TURN_ANGLE - self.turn_progress
                turn_step = min(TURNING_RANGE, remaining)
                self.agent_angle += self.turn_direction * turn_step
                self.agent_angle = max(MIN_ANGLE, min(MAX_ANGLE, self.agent_angle))
                self.turn_progress += turn_step
                if self.turn_progress >= TURN_ANGLE:
                    self.turning = False
                    self.steps += 1
                    reward = STEP_PENALTY
                    info["step_completed"] = True
                    info["action"] = self.pending_action
                    self.pending_action = None
            elif self.moving:
                remaining = MOVE_DIST - self.move_progress
                move_step = min(STEP_RANGE, remaining)
                rad = math.radians(self.agent_angle)
                self.agent_x += math.sin(rad) * move_step * self.move_direction
                self.agent_y -= math.cos(rad) * move_step * self.move_direction
                self.move_progress += move_step
                if self.move_progress >= MOVE_DIST:
                    self.moving = False
                    self.steps += 1
                    reward = STEP_PENALTY
                    if self.pending_action == 0 and center_sensor_clear:
                        reward += ASCENT_REWARD
                    info["step_completed"] = True
                    info["action"] = self.pending_action
                    self.pending_action = None
            else:
                if action in [0, 1, 2]:
                    prev_action_before = self.prev_action
                    if action == 0:
                        self.moving, self.move_direction, self.move_progress = (
                            True,
                            1,
                            0,
                        )
                        self.pending_action = action
                        self.prev_action = action
                    elif action == 1:
                        if self.agent_angle <= MIN_ANGLE:
                            reward = STEP_PENALTY + TURN_LIMIT_PENALTY
                            self.steps += 1
                            info["step_completed"] = True
                            info["action"] = action
                        else:
                            if (
                                prev_action_before in [1, 2]
                                and action in [1, 2]
                                and action != prev_action_before
                            ):
                                reward += REPEAT_TURN_PENALTY
                            self.turning, self.turn_direction, self.turn_progress = (
                                True,
                                -1,
                                0,
                            )
                            self.pending_action = action
                            self.prev_action = action
                    elif action == 2:
                        if self.agent_angle >= MAX_ANGLE:
                            reward = STEP_PENALTY + TURN_LIMIT_PENALTY
                            self.steps += 1
                            info["step_completed"] = True
                            info["action"] = action
                        else:
                            if (
                                prev_action_before in [1, 2]
                                and action in [1, 2]
                                and action != prev_action_before
                            ):
                                reward += REPEAT_TURN_PENALTY
                            self.turning, self.turn_direction, self.turn_progress = (
                                True,
                                1,
                                0,
                            )
                            self.pending_action = action
                            self.prev_action = action

        # Collisions
        agent_rect = self.get_agent_rect()
        if any(agent_rect.colliderect(wall) for wall in self.wall_rects) or any(
            agent_rect.colliderect(obs) for obs in self.obstacle_rects
        ):
            self.done, self.collision, self.termination_reason = True, True, "Collision"
            reward += COLLISION_PENALTY
        elif agent_rect.top <= FINISH_Y:
            self.done, self.success, self.termination_reason = True, True, "Success"
            reward += SUCCESS_REWARD
        elif self.steps >= MAX_STEPS and not self.done:
            self.done, self.termination_reason = True, "Timeout"
            reward += TIMEOUT_PENALTY

        info.update(
            {
                "success": self.success,
                "collision": self.collision,
                "termination_reason": self.termination_reason,
            }
        )
        return self.get_state(), reward, self.done, info

    def get_agent_rect(self):
        return pygame.Rect(
            self.agent_x - AGENT_SIZE // 2,
            self.agent_y - AGENT_SIZE // 2,
            AGENT_SIZE,
            AGENT_SIZE,
        )

    def get_state(self):
        angle = round(self.agent_angle / TURN_ANGLE) * TURN_ANGLE
        angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
        compass = float((angle + 90) // TURN_ANGLE + 1)
        sensors = get_sensor_values(
            self.agent_x, self.agent_y, self.agent_angle, self.all_edges
        )
        prev_input = 0.0 if self.prev_action is None else float(self.prev_action + 1)
        return np.array([compass] + sensors + [prev_input], dtype=np.float32)


# ====================================================
# Sensors
# ====================================================
def get_sensor_values(agent_x, agent_y, agent_angle, edges):
    sensors = []
    for rel_angle in SENSOR_ANGLES:
        rad = math.radians(agent_angle + rel_angle)
        end_x = agent_x + math.sin(rad) * MAX_RANGE
        end_y = agent_y - math.cos(rad) * MAX_RANGE
        min_dist = MAX_RANGE
        for p3, p4 in edges:
            dist = line_intersection((agent_x, agent_y), (end_x, end_y), p3, p4)
            if dist is not None and dist < min_dist:
                min_dist = dist
        sensors.append(round(max(0.0, min(1.0, min_dist / MAX_RANGE)), 2))
    return sensors


def line_intersection(p1, p2, p3, p4):
    x1, y1, x2, y2, x3, y3, x4, y4 = *p1, *p2, *p3, *p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    if 0 <= t <= 1 and 0 <= u <= 1:
        return math.hypot(x1 + t * (x2 - x1) - x1, y1 + t * (y2 - y1) - y1)
    return None


# ====================================================
# DQN & Replay Buffer
# ====================================================
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        s, a, r, ns, d = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(np.array(a)).unsqueeze(1),
            torch.FloatTensor(np.array(r)).unsqueeze(1),
            torch.FloatTensor(np.array(ns)),
            torch.FloatTensor(np.array(d)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ====================================================
# Agent
# ====================================================
class Agent:
    def __init__(self, load_model=False, model_path=None, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.policy_net = DQN(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(self.device)
        self.target_net = DQN(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(self.device)

        if load_model and model_path and os.path.exists(model_path):
            self.policy_net.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.loss, self.avg_q = 0.0, 0.0
        print(f"Agent using device: {self.device}")

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        with torch.no_grad():
            q = self.policy_net(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            self.avg_q = q.mean().item()
            return q.argmax().item()

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return
        s, a, r, ns, d = self.memory.sample(BATCH_SIZE)
        s, a, r, ns, d = (
            s.to(self.device),
            a.to(self.device),
            r.to(self.device),
            ns.to(self.device),
            d.to(self.device),
        )

        q = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            target = r + GAMMA * self.target_net(ns).max(1)[0].unsqueeze(1) * (1 - d)

        loss = nn.MSELoss()(q, target)
        self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ====================================================
# Visualization (For UI)
# ====================================================
def draw_environment(
    screen,
    env,
    agent_obj,
    episode_num,
    total_reward,
    last_action,
    clock,
    action_names,
    slow_motion,
):
    screen.fill(WHITE)
    for wall in env.wall_rects:
        pygame.draw.rect(screen, BLACK, wall)
    pygame.draw.rect(screen, GREEN, (ARENA_X, FINISH_Y, WALL_WIDTH, FINISH_THICKNESS))

    for dyn in env.dynamic_obstacles:
        pygame.draw.line(
            screen,
            (200, 200, 255),
            (dyn.start_x + dyn.width // 2, dyn.start_y + dyn.height // 2),
            (dyn.end_x + dyn.width // 2, dyn.end_y + dyn.height // 2),
            2,
        )

    for i, obs_rect in enumerate(env.obstacle_rects):
        color = BLACK if i in env.dynamic_indices else BLACK
        pygame.draw.rect(screen, color, obs_rect)
        pygame.draw.rect(
            screen, DARK_GRAY if i in env.dynamic_indices else BLACK, obs_rect, 2
        )

    for rel_angle in SENSOR_ANGLES:
        rad = math.radians(env.agent_angle + rel_angle)
        end_x = env.agent_x + math.sin(rad) * MAX_RANGE
        end_y = env.agent_y - math.cos(rad) * MAX_RANGE
        pygame.draw.line(screen, GREEN, (env.agent_x, env.agent_y), (end_x, end_y), 2)

    agent_rect = env.get_agent_rect()
    pygame.draw.rect(screen, BLUE, agent_rect)
    pygame.draw.rect(screen, BLACK, agent_rect, 2)

    head_x = env.agent_x + math.sin(math.radians(env.agent_angle)) * (
        AGENT_SIZE // 2 + 15
    )
    head_y = env.agent_y - math.cos(math.radians(env.agent_angle)) * (
        AGENT_SIZE // 2 + 15
    )
    pygame.draw.line(screen, YELLOW, (env.agent_x, env.agent_y), (head_x, head_y), 3)

    status_text, status_color = "RUNNING", BLUE
    if env.done:
        if env.success:
            status_text, status_color = "✓ SUCCESS!", GREEN
        elif env.collision:
            status_text, status_color = "✗ COLLISION!", RED
        else:
            status_text, status_color = "⊗ TIMEOUT", ORANGE

    left_texts = [
        f"Episode: {episode_num}",
        f"Step: {env.steps}/{MAX_STEPS}",
        f"Reward: {total_reward:.2f}",
        f"Action: {action_names.get(last_action, 'None')}",
        f"Epsilon: {agent_obj.epsilon:.3f}",
        f"Loss: {agent_obj.loss:.4f}",
        f"Slow-mo: {'ON' if slow_motion else 'OFF'}",
    ]

    for i, text in enumerate(left_texts):
        screen.blit(
            ALL_FONT.render(text, True, ALL_FONT_COLOR),
            (ARENA_X + 10, ARENA_Y + 10 + i * 18),
        )
