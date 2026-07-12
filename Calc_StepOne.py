import json
import math
import os
import random
import sys

import pygame

from constant import *


class StepOneSimulator:
    def __init__(self):
        self.obstacles = []
        self.agent_x = START_X
        self.agent_y = START_Y
        self.agent_angle = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.last_log = None
        self.step_count = 0
        self.collision = False
        self.success = False
        self.done = False
        self.freeze_after_step = False
        self.animating = False
        self.animation_progress = 0.0
        self.animation_duration = 12
        self.animation_action = None
        self.animation_start_x = self.agent_x
        self.animation_start_y = self.agent_y
        self.animation_start_angle = self.agent_angle
        self.animation_target_x = self.agent_x
        self.animation_target_y = self.agent_y
        self.animation_target_angle = self.agent_angle
        self.pending_state_before = None
        self.pending_state_after = None
        self.pending_action_name = "None"
        self.sensor_values = []
        self.load_obstacles()

        if not self.obstacles:
            self.add_obstacle(OBS_X, OBS_Y, OBS_SIZE, OBS_SIZE, "Central Obstacle")

    def add_obstacle(self, x, y, width=40, height=40, name="Obstacle"):
        x = round(x / 10) * 10
        y = round(y / 10) * 10
        rect = pygame.Rect(x, y, width, height)
        self.obstacles.append(
            {"id": len(self.obstacles) + 1, "rect": rect, "name": name}
        )
        self.save_obstacles()

    def remove_obstacle_at(self, pos):
        for idx in range(len(self.obstacles) - 1, -1, -1):
            obstacle = self.obstacles[idx]
            if obstacle["rect"].collidepoint(pos):
                del self.obstacles[idx]
                self.save_obstacles()
                return True
        return False

    def set_agent_start(self, x, y):
        self.agent_x = x
        self.agent_y = y
        self.save_obstacles()

    def load_obstacles(self):
        path = os.path.join(os.path.dirname(__file__), OBSTACLE_FILE)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self.obstacles = []
                for item in data.get("obstacles", []):
                    rect = pygame.Rect(
                        item["x"], item["y"], item["width"], item["height"]
                    )
                    self.obstacles.append(
                        {
                            "id": item.get("id", len(self.obstacles) + 1),
                            "rect": rect,
                            "name": item.get("name", "Obstacle"),
                        }
                    )
                if data.get("agent"):
                    self.agent_x = data["agent"].get("x", START_X)
                    self.agent_y = data["agent"].get("y", START_Y)
                return
            except Exception as exc:
                print(f"Tidak bisa membaca obstacle config: {exc}")
        self.obstacles = []

    def save_obstacles(self):
        path = os.path.join(os.path.dirname(__file__), OBSTACLE_FILE)
        payload = {
            "version": "1.0",
            "obstacles": [
                {
                    "id": item["id"],
                    "x": item["rect"].x,
                    "y": item["rect"].y,
                    "width": item["rect"].width,
                    "height": item["rect"].height,
                    "name": item["name"],
                    "is_dynamic": False,
                }
                for item in self.obstacles
            ],
            "agent": {"x": self.agent_x, "y": self.agent_y},
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=4)

    def _build_edges(self):
        edges = []
        for obstacle in self.obstacles:
            rect = obstacle["rect"]
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

    def _get_sensor_values(self, agent_x, agent_y, agent_angle, edges):
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

    def get_sensor_values(self):
        return self._get_sensor_values(
            self.agent_x, self.agent_y, self.agent_angle, self._build_edges()
        )

    def get_state(self, prev_action=None):
        angle = round(self.agent_angle / TURN_ANGLE) * TURN_ANGLE
        angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
        compass = float((angle + 90) // TURN_ANGLE + 1)
        sensors = self.get_sensor_values()
        prev_input = 0.0 if prev_action is None else 1.0
        return [compass] + sensors + [prev_input]

    def _clamp_agent_position(self, x, y):
        x = max(ARENA_X + 10, min(ARENA_X + WALL_WIDTH - 10, x))
        y = max(ARENA_Y + 10, min(ARENA_Y + WALL_HEIGHT - 10, y))
        return x, y

    def _evaluate_result(self, x, y, angle):
        x, y = self._clamp_agent_position(x, y)
        agent_rect = pygame.Rect(
            x - AGENT_SIZE // 2, y - AGENT_SIZE // 2, AGENT_SIZE, AGENT_SIZE
        )
        if any(agent_rect.colliderect(wall_rect) for wall_rect in self.wall_rects()):
            return True, False, COLLISION_PENALTY
        if any(agent_rect.colliderect(obs["rect"]) for obs in self.obstacles):
            return True, False, COLLISION_PENALTY
        if y <= FINISH_Y + 10:
            return False, True, SUCCESS_REWARD
        return False, False, 0.0

    def start_action(self):
        if self.freeze_after_step or self.animating:
            return False

        action = random.choice([0, 1, 2])
        self.step_count += 1
        self.last_action = action
        self.collision = False
        self.success = False
        self.done = False
        self.last_reward = 0.0
        self.freeze_after_step = False
        self.animating = True
        self.animation_progress = 0.0
        self.animation_action = action
        self.animation_start_x = self.agent_x
        self.animation_start_y = self.agent_y
        self.animation_start_angle = self.agent_angle

        if action == 0:
            rad = math.radians(self.agent_angle)
            self.animation_target_x = self.agent_x + math.sin(rad) * MOVE_DIST
            self.animation_target_y = self.agent_y - math.cos(rad) * MOVE_DIST
            self.animation_duration = max(
                8, int(math.ceil(MOVE_DIST / max(1.0, STEP_RANGE)))
            )
            self.pending_action_name = "Forward"
        elif action == 1:
            self.animation_target_x = self.agent_x
            self.animation_target_y = self.agent_y
            self.animation_target_angle = self.agent_angle - TURN_ANGLE
            self.pending_action_name = "Turn Left"
        else:
            self.animation_target_x = self.agent_x
            self.animation_target_y = self.agent_y
            self.animation_target_angle = self.agent_angle + TURN_ANGLE
            self.animation_duration = max(
                8, int(math.ceil(abs(TURN_ANGLE) / max(1.0, TURNING_RANGE)))
            )
            self.pending_action_name = "Turn Right"

        self.animation_target_angle = max(
            MIN_ANGLE, min(MAX_ANGLE, self.animation_target_angle)
        )
        self.pending_state_before = self.get_state(prev_action=None)
        self.pending_state_after = self._build_state_for(
            self.animation_target_x,
            self.animation_target_y,
            self.animation_target_angle,
            prev_action=action,
        )
        self.sensor_values = self.get_sensor_values()
        return True

    def _build_state_for(self, x, y, angle, prev_action=None):
        angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
        angle = round(angle / TURN_ANGLE) * TURN_ANGLE
        compass = float((angle + 90) // TURN_ANGLE + 1)
        sensors = self._get_sensor_values(x, y, angle, self._build_edges())
        prev_input = 0.0 if prev_action is None else 1.0
        return [compass] + sensors + [prev_input]

    def update_animation(self):
        if not self.animating:
            return False

        self.animation_progress += 1.0 / self.animation_duration
        if self.animation_progress >= 1.0:
            self.animation_progress = 1.0
            self.agent_x = self.animation_target_x
            self.agent_y = self.animation_target_y
            self.agent_angle = self.animation_target_angle
            self.agent_x, self.agent_y = self._clamp_agent_position(
                self.agent_x, self.agent_y
            )
            self.agent_angle = max(MIN_ANGLE, min(MAX_ANGLE, self.agent_angle))
            self.collision, self.success, self.last_reward = self._evaluate_result(
                self.agent_x, self.agent_y, self.agent_angle
            )
            self.done = self.collision or self.success
            self.freeze_after_step = True
            self.animating = False
            self.last_log = {
                "action": self.last_action,
                "action_name": self.pending_action_name,
                "reward": self.last_reward,
                "state_before": self.pending_state_before,
                "state_after": self.pending_state_after,
                "collision": self.collision,
                "success": self.success,
                "done": self.done,
            }
            self.sensor_values = self.get_sensor_values()
            return True

        t = self.animation_progress
        self.agent_x = (
            self.animation_start_x
            + (self.animation_target_x - self.animation_start_x) * t
        )
        self.agent_y = (
            self.animation_start_y
            + (self.animation_target_y - self.animation_start_y) * t
        )
        self.agent_angle = (
            self.animation_start_angle
            + (self.animation_target_angle - self.animation_start_angle) * t
        )
        self.agent_x, self.agent_y = self._clamp_agent_position(
            self.agent_x, self.agent_y
        )
        self.agent_angle = max(MIN_ANGLE, min(MAX_ANGLE, self.agent_angle))
        self.sensor_values = self.get_sensor_values()
        return False

    def wall_rects(self):
        return [
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

    def reset(self):
        self.agent_x = START_X
        self.agent_y = START_Y
        self.agent_angle = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.last_log = None
        self.step_count = 0
        self.collision = False
        self.success = False
        self.done = False
        self.freeze_after_step = False
        self.animating = False
        self.animation_progress = 0.0
        self.animation_action = None
        self.animation_start_x = self.agent_x
        self.animation_start_y = self.agent_y
        self.animation_start_angle = self.agent_angle
        self.animation_target_x = self.agent_x
        self.animation_target_y = self.agent_y
        self.animation_target_angle = self.agent_angle
        self.pending_state_before = None
        self.pending_state_after = None
        self.pending_action_name = "None"
        self.sensor_values = []

    def format_value(self, value):
        if isinstance(value, float):
            return f"{value:,.2f}".replace(",", "~").replace(".", ",").replace("~", ".")
        if value is None:
            return "None"
        return str(value)

    def print_log(self):
        if not self.last_log:
            return
        state_before = self.last_log["state_before"]
        state_after = self.last_log["state_after"]
        action = self.last_log["action"]
        action_name = self.last_log["action_name"]
        reward = self.last_log["reward"]

        print("\n" + "=" * 60)
        print("A) States")
        print("State (Step 1):")
        print(f"{'Input':<12} {'Value':<10} {'Normalized':<12}")
        print(
            f"{'Angle':<12} {self.format_angle(self._angle_from_compass(state_before[0])):<10} {self.format_value(state_before[0])}"
        )
        for idx, sensor in enumerate(state_before[1:8], start=1):
            print(
                f"{'Sensor ' + str(idx):<12} {self.format_value(sensor):<10} {self.format_value(sensor)}"
            )
        print(f"{'Prev Action':<12} {'None':<10} {self.format_value(0.0)}")
        print()
        print("Next State (Step 2):")
        print(f"{'Input':<12} {'Value':<10} {'Normalized':<12}")
        print(
            f"{'Angle':<12} {self.format_angle(self._angle_from_compass(state_after[0])):<10} {self.format_value(state_after[0])}"
        )
        for idx, sensor in enumerate(state_after[1:8], start=1):
            print(
                f"{'Sensor ' + str(idx):<12} {self.format_value(sensor):<10} {self.format_value(sensor)}"
            )
        print(f"{'Prev Action':<12} {action_name:<10} {self.format_value(1.0)}")
        print()
        print("B) Action and Reward")
        print(f"Step 1: Action (a) = exploration = {action} ({action_name})")
        print(f"Step 1: reward (r) = {self.format_value(reward)}")
        print()
        print("C) Replay Buffer")
        print("Stored Replay Buffer:")
        print(f"s_t\t\t: {self.format_state_vector(state_before)} (state step 1)")
        print(f"a_(t )\t\t: {action} (action step 1)")
        print(f"r_(t )\t\t: {self.format_value(reward)} (reward step 1)")
        print(f"s_(t+1)\t: {self.format_state_vector(state_after)} (state step 2)")
        print(
            f"d_t\t\t: {'1' if self.done else '0'} (episode {'berakhir' if self.done else 'belum berakhir'})"
        )
        print()
        print("D) Check Replay Buffer Size D(size)")
        print(
            "Pada simulasi ini Batch Size yang di set pada hyperparameter DQN adalah sebesar 64."
        )
        print(
            "Mengartikan pelatihan akan dimulai apabila Replay Buffer Size ≥ Batch Size"
        )
        print(f"D(size) = 1")
        print("D(size) ≥ 64: Start Training")
        print("D(size) < 64: No Training, continue gathering information")
        print("Dikarenakan pada step ini D(size) = 1 maka No Training.")
        print("=" * 60)

    def _angle_to_compass(self, angle):
        angle = round(angle / TURN_ANGLE) * TURN_ANGLE
        angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
        return float((angle + 90) // TURN_ANGLE + 1)

    def format_angle(self, angle):
        return f"{int(angle)}°"

    def format_state_vector(self, state):
        values = []
        for value in state:
            values.append(self.format_value(value))
        return "[" + ", ".join(values) + "]"

    def _angle_from_compass(self, compass_value):
        return (compass_value - 1) * TURN_ANGLE - 90


class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Calc Step One - DQN Simulator")
        self.clock = pygame.time.Clock()
        self.sim = StepOneSimulator()
        self.font = pygame.font.SysFont("consolas", 14)
        self.small_font = pygame.font.SysFont("consolas", 12)
        self.button_rect = pygame.Rect(180, HEIGHT - 60, 140, 40)
        self.reset_button_rect = pygame.Rect(340, HEIGHT - 60, 140, 40)
        self.selected_obstacle_index = None
        self.dragging = False
        self.resizing = False
        self.resize_edge = None
        self.drag_offset = (0, 0)
        self.running = True

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.button_rect.collidepoint(event.pos):
                            if (
                                not self.sim.freeze_after_step
                                and not self.sim.animating
                            ):
                                self.sim.start_action()
                        elif self.reset_button_rect.collidepoint(event.pos):
                            self.sim.reset()
                            self.selected_obstacle_index = None
                            self.dragging = False
                            self.resizing = False
                            self.resize_edge = None
                        elif self._is_inside_arena(event.pos):
                            obstacle_index = self._get_obstacle_index(event.pos)
                            if obstacle_index is not None:
                                self.selected_obstacle_index = obstacle_index
                                rect = self.sim.obstacles[obstacle_index]["rect"]
                                handle = self._get_resize_handle(event.pos, rect)
                                if handle:
                                    self.resizing = True
                                    self.resize_edge = handle
                                else:
                                    self.dragging = True
                                    self.drag_offset = (
                                        event.pos[0] - rect.x,
                                        event.pos[1] - rect.y,
                                    )
                            else:
                                self.selected_obstacle_index = None
                                self.dragging = False
                                self.resizing = False
                                self.resize_edge = None
                                self.sim.add_obstacle(
                                    event.pos[0], event.pos[1], 40, 40, "Obstacle"
                                )
                    elif event.button == 3:
                        if self._is_inside_arena(event.pos):
                            obstacle_index = self._get_obstacle_index(event.pos)
                            if obstacle_index is not None:
                                del self.sim.obstacles[obstacle_index]
                                self.selected_obstacle_index = None
                                self.sim.save_obstacles()
                            else:
                                self.sim.set_agent_start(event.pos[0], event.pos[1])
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False
                        self.resizing = False
                        self.resize_edge = None
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging and self.selected_obstacle_index is not None:
                        rect = self.sim.obstacles[self.selected_obstacle_index]["rect"]
                        new_x = round((event.pos[0] - self.drag_offset[0]) / 10) * 10
                        new_y = round((event.pos[1] - self.drag_offset[1]) / 10) * 10
                        rect.x = max(
                            ARENA_X, min(ARENA_X + WALL_WIDTH - rect.width, new_x)
                        )
                        rect.y = max(
                            ARENA_Y, min(ARENA_Y + WALL_HEIGHT - rect.height, new_y)
                        )
                        self.sim.save_obstacles()
                    elif self.resizing and self.selected_obstacle_index is not None:
                        rect = self.sim.obstacles[self.selected_obstacle_index]["rect"]
                        x, y = event.pos
                        x = round(x / 10) * 10
                        y = round(y / 10) * 10
                        if "left" in self.resize_edge:
                            new_left = min(rect.right - 20, x)
                            rect.width = max(20, rect.right - new_left)
                            rect.x = new_left
                        if "right" in self.resize_edge:
                            rect.width = max(20, x - rect.x)
                        if "top" in self.resize_edge:
                            new_top = min(rect.bottom - 20, y)
                            rect.height = max(20, rect.bottom - new_top)
                            rect.y = new_top
                        if "bottom" in self.resize_edge:
                            rect.height = max(20, y - rect.y)
                        rect.x = max(
                            ARENA_X, min(ARENA_X + WALL_WIDTH - rect.width, rect.x)
                        )
                        rect.y = max(
                            ARENA_Y, min(ARENA_Y + WALL_HEIGHT - rect.height, rect.y)
                        )
                        self.sim.save_obstacles()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.sim.reset()
                        self.selected_obstacle_index = None
                        self.dragging = False
                        self.resizing = False
                        self.resize_edge = None
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False

            completed = self.sim.update_animation()
            if completed:
                self.sim.print_log()

            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit(0)

    def _is_inside_arena(self, pos):
        return (
            ARENA_X <= pos[0] <= ARENA_X + WALL_WIDTH
            and ARENA_Y <= pos[1] <= ARENA_Y + WALL_HEIGHT
        )

    def _get_obstacle_index(self, pos):
        for index in range(len(self.sim.obstacles) - 1, -1, -1):
            if self.sim.obstacles[index]["rect"].collidepoint(pos):
                return index
        return None

    def _get_resize_handle(self, pos, rect):
        margin = 10
        if abs(pos[0] - rect.left) < margin and abs(pos[1] - rect.top) < margin:
            return "top-left"
        if abs(pos[0] - rect.right) < margin and abs(pos[1] - rect.top) < margin:
            return "top-right"
        if abs(pos[0] - rect.left) < margin and abs(pos[1] - rect.bottom) < margin:
            return "bottom-left"
        if abs(pos[0] - rect.right) < margin and abs(pos[1] - rect.bottom) < margin:
            return "bottom-right"
        if abs(pos[0] - rect.left) < margin:
            return "left"
        if abs(pos[0] - rect.right) < margin:
            return "right"
        if abs(pos[1] - rect.top) < margin:
            return "top"
        if abs(pos[1] - rect.bottom) < margin:
            return "bottom"
        return None

    def draw(self):
        self.screen.fill(WHITE)

        pygame.draw.rect(
            self.screen, BLACK, (ARENA_X, ARENA_Y, WALL_WIDTH, WALL_HEIGHT), 2
        )
        for wall in self.sim.wall_rects():
            pygame.draw.rect(self.screen, BLACK, wall)

        for index, obstacle in enumerate(self.sim.obstacles):
            rect = obstacle["rect"]
            color = (255, 220, 120) if index == self.selected_obstacle_index else BLACK
            border = (
                (80, 80, 80) if index != self.selected_obstacle_index else (255, 140, 0)
            )
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, border, rect, 2)

        pygame.draw.rect(
            self.screen, GREEN, (ARENA_X, FINISH_Y, WALL_WIDTH, FINISH_THICKNESS)
        )

        sensor_values = self.sim.get_sensor_values()
        for idx, sensor_value in enumerate(sensor_values):
            rel_angle = SENSOR_ANGLES[idx]
            rad = math.radians(self.sim.agent_angle + rel_angle)
            end_x = self.sim.agent_x + math.sin(rad) * MAX_RANGE
            end_y = self.sim.agent_y - math.cos(rad) * MAX_RANGE
            color = (
                (0, 180, 0)
                if sensor_value > 0.7
                else (220, 140, 0) if sensor_value > 0.3 else (220, 50, 50)
            )
            pygame.draw.line(
                self.screen,
                color,
                (self.sim.agent_x, self.sim.agent_y),
                (end_x, end_y),
                2,
            )
            label = self.small_font.render(f"{sensor_value:.2f}", True, BLACK)
            self.screen.blit(label, (end_x - 12, end_y - 10))

        agent_rect = pygame.Rect(
            self.sim.agent_x - AGENT_SIZE // 2,
            self.sim.agent_y - AGENT_SIZE // 2,
            AGENT_SIZE,
            AGENT_SIZE,
        )
        pygame.draw.rect(self.screen, BLUE, agent_rect)
        pygame.draw.rect(self.screen, BLACK, agent_rect, 2)

        head_x = self.sim.agent_x + math.sin(math.radians(self.sim.agent_angle)) * (
            AGENT_SIZE // 2 + 8
        )
        head_y = self.sim.agent_y - math.cos(math.radians(self.sim.agent_angle)) * (
            AGENT_SIZE // 2 + 8
        )
        pygame.draw.line(
            self.screen,
            YELLOW,
            (self.sim.agent_x, self.sim.agent_y),
            (head_x, head_y),
            3,
        )

        panel_rect = pygame.Rect(WIDTH - 160, 20, 140, 165)
        pygame.draw.rect(self.screen, LIGHT_BLUE, panel_rect)
        pygame.draw.rect(self.screen, BLACK, panel_rect, 2)
        title = self.font.render("Sensors", True, BLACK)
        self.screen.blit(title, (panel_rect.x + 10, panel_rect.y + 8))
        for idx, sensor_value in enumerate(sensor_values):
            label = self.small_font.render(
                f"S{idx + 1}: {sensor_value:.2f}", True, BLACK
            )
            self.screen.blit(label, (panel_rect.x + 10, panel_rect.y + 32 + idx * 15))

        pygame.draw.rect(self.screen, LIGHT_BLUE, self.button_rect)
        pygame.draw.rect(self.screen, BLACK, self.button_rect, 2)
        label = self.font.render("RUN", True, BLACK)
        self.screen.blit(label, (self.button_rect.x + 48, self.button_rect.y + 12))

        pygame.draw.rect(self.screen, LIGHT_BLUE, self.reset_button_rect)
        pygame.draw.rect(self.screen, BLACK, self.reset_button_rect, 2)
        reset_label = self.font.render("RESET", True, BLACK)
        self.screen.blit(
            reset_label, (self.reset_button_rect.x + 38, self.reset_button_rect.y + 12)
        )

        if self.sim.freeze_after_step:
            status = self.font.render("Step completed. Agent frozen.", True, RED)
            self.screen.blit(status, (10, HEIGHT - 95))


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


if __name__ == "__main__":
    App().run()
