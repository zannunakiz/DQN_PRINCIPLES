# ====================================================
# OBSTACLE MODIFIER TOOL - Dynamic Obstacles Support
# No buttons, just click to edit
# Left click empty space = Add static obstacle
# Left click obstacle = Select/Edit (drag to move, drag corners to resize)
# Right click empty space 2x = Create dynamic obstacle (start/end points)
# Right click obstacle = Delete
# Left click dynamic obstacle = Freeze for editing (input seconds)
# Press S = Save
# ====================================================

import pygame
import json
import os
import math
import time

# ====================================================
# Configuration
# ====================================================

from constant import *

# Window settings
WINDOW_WIDTH = WIDTH
WINDOW_HEIGHT = HEIGHT

DEFAULT_OBSTACLE = {
    "x": OBS_X,
    "y": OBS_Y,
    "width": OBS_SIZE,
    "height": OBS_SIZE,
    "name": "Central Obstacle",
}

# ====================================================
# Dynamic Obstacle Class
# ====================================================


class DynamicObstacle:
    def __init__(self, x, y, width, height, name="Dynamic"):
        self.id = None
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.name = name
        self.start_x = x
        self.start_y = y
        self.end_x = x + 100
        self.end_y = y
        self.seconds = 2.0
        self.is_dynamic = True
        self.progress = 0.0
        self.direction = 1
        self.moving = False
        self.frozen = False  # New: freeze state for editing
        self.last_update = time.time()

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, dt):
        if not self.moving or self.frozen:
            return

        # Calculate distance to move
        total_distance = math.hypot(
            self.end_x - self.start_x, self.end_y - self.start_y
        )
        if total_distance == 0:
            return

        speed = total_distance / self.seconds  # pixels per second
        distance_to_move = speed * dt

        # Move along path
        self.progress += (distance_to_move / total_distance) * self.direction

        # Check boundaries
        if self.progress >= 1.0:
            self.progress = 1.0
            self.direction = -1
        elif self.progress <= 0.0:
            self.progress = 0.0
            self.direction = 1

        # Interpolate position
        self.x = self.start_x + (self.end_x - self.start_x) * self.progress
        self.y = self.start_y + (self.end_y - self.start_y) * self.progress

    def to_dict(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "name": self.name,
            "is_dynamic": self.is_dynamic,
            "start_x": self.start_x,
            "start_y": self.start_y,
            "end_x": self.end_x,
            "end_y": self.end_y,
            "seconds": self.seconds,
        }

    @staticmethod
    def from_dict(data):
        obs = DynamicObstacle(
            data["x"],
            data["y"],
            data["width"],
            data["height"],
            data.get("name", "Dynamic"),
        )
        obs.id = data.get("id")
        obs.is_dynamic = data.get("is_dynamic", True)
        obs.start_x = data.get("start_x", data["x"])
        obs.start_y = data.get("start_y", data["y"])
        obs.end_x = data.get("end_x", data["x"] + 100)
        obs.end_y = data.get("end_y", data["y"])
        obs.seconds = data.get("seconds", 2.0)
        return obs


# ====================================================
# Obstacle Editor
# ====================================================


class ObstacleEditor:
    def __init__(self):
        # Initialize attributes
        self.snap_grid = 10
        self.obstacles = []  # Can be dict (static) or DynamicObstacle
        self.selected_obstacle = None
        self.dragging = False
        self.drag_offset = (0, 0)
        self.resizing = False
        self.resize_edge = None
        self.next_id = 1
        self.hover_obstacle = None
        self.hover_handle = None

        # Dynamic obstacle creation
        self.right_click_count = 0
        self.right_click_pos = None
        self.show_start_indicator = False
        self.start_indicator_pos = None

        # Input handling for seconds
        self.input_active = False
        self.input_obstacle = None
        self.input_text = ""
        self.input_rect = None

        # Load existing obstacles
        self.load_obstacles()

        # Add default obstacle if empty
        if not self.obstacles:
            self.add_static_obstacle(
                DEFAULT_OBSTACLE["x"],
                DEFAULT_OBSTACLE["y"],
                DEFAULT_OBSTACLE["width"],
                DEFAULT_OBSTACLE["height"],
                DEFAULT_OBSTACLE["name"],
            )

        # Update next_id
        self.next_id = (
            max(
                [
                    obs.get("id", 0) if isinstance(obs, dict) else obs.id
                    for obs in self.obstacles
                ],
                default=0,
            )
            + 1
        )

    def add_static_obstacle(self, x, y, width, height, name="Obstacle"):
        """Add a new static obstacle"""
        x = round(x / self.snap_grid) * self.snap_grid
        y = round(y / self.snap_grid) * self.snap_grid

        obstacle = {
            "id": self.next_id,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "name": name,
            "is_dynamic": False,
        }
        self.obstacles.append(obstacle)
        self.next_id += 1
        self.selected_obstacle = obstacle
        return obstacle

    def add_dynamic_obstacle(self, start_x, start_y, end_x, end_y):
        """Add a new dynamic obstacle"""
        width, height = 40, 40
        start_x = round(start_x / self.snap_grid) * self.snap_grid
        start_y = round(start_y / self.snap_grid) * self.snap_grid
        end_x = round(end_x / self.snap_grid) * self.snap_grid
        end_y = round(end_y / self.snap_grid) * self.snap_grid

        obs = DynamicObstacle(
            start_x, start_y, width, height, f"Dynamic {self.next_id}"
        )
        obs.id = self.next_id
        obs.end_x = end_x
        obs.end_y = end_y
        obs.moving = True
        obs.frozen = False
        self.obstacles.append(obs)
        self.next_id += 1
        self.selected_obstacle = obs
        return obs

    def remove_obstacle(self, obstacle):
        """Remove an obstacle"""
        if obstacle in self.obstacles:
            self.obstacles.remove(obstacle)
            if self.selected_obstacle == obstacle:
                self.selected_obstacle = None
            # If the removed obstacle was the input obstacle, close input
            if self.input_active and self.input_obstacle == obstacle:
                self.input_active = False
                self.input_obstacle = None
                self.input_text = ""

    def get_obstacle_at(self, pos):
        """Get obstacle at position"""
        x, y = pos
        for obs in reversed(self.obstacles):
            if isinstance(obs, dict):
                rect = pygame.Rect(obs["x"], obs["y"], obs["width"], obs["height"])
            else:
                rect = obs.get_rect()
            hit_rect = rect.inflate(8, 8)
            if hit_rect.collidepoint(x, y):
                return obs
        return None

    def get_resize_edge(self, pos, obstacle):
        """Get which edge/corner is being resized"""
        if isinstance(obstacle, dict):
            rect = pygame.Rect(
                obstacle["x"], obstacle["y"], obstacle["width"], obstacle["height"]
            )
        else:
            rect = obstacle.get_rect()

        x, y = pos
        margin = 15

        corner_margin = 18
        if abs(x - rect.left) < corner_margin and abs(y - rect.top) < corner_margin:
            return "top-left"
        elif abs(x - rect.right) < corner_margin and abs(y - rect.top) < corner_margin:
            return "top-right"
        elif (
            abs(x - rect.left) < corner_margin and abs(y - rect.bottom) < corner_margin
        ):
            return "bottom-left"
        elif (
            abs(x - rect.right) < corner_margin and abs(y - rect.bottom) < corner_margin
        ):
            return "bottom-right"
        elif abs(x - rect.left) < margin:
            return "left"
        elif abs(x - rect.right) < margin:
            return "right"
        elif abs(y - rect.top) < margin:
            return "top"
        elif abs(y - rect.bottom) < margin:
            return "bottom"
        return None

    def get_handle_rects(self, obstacle):
        """Get all handle rectangles for drawing"""
        if isinstance(obstacle, dict):
            rect = pygame.Rect(
                obstacle["x"], obstacle["y"], obstacle["width"], obstacle["height"]
            )
        else:
            rect = obstacle.get_rect()

        handle_size = 10
        handles = {
            "top-left": pygame.Rect(
                rect.left - handle_size // 2,
                rect.top - handle_size // 2,
                handle_size,
                handle_size,
            ),
            "top-right": pygame.Rect(
                rect.right - handle_size // 2,
                rect.top - handle_size // 2,
                handle_size,
                handle_size,
            ),
            "bottom-left": pygame.Rect(
                rect.left - handle_size // 2,
                rect.bottom - handle_size // 2,
                handle_size,
                handle_size,
            ),
            "bottom-right": pygame.Rect(
                rect.right - handle_size // 2,
                rect.bottom - handle_size // 2,
                handle_size,
                handle_size,
            ),
            "top": pygame.Rect(
                rect.centerx - handle_size // 2,
                rect.top - handle_size // 2,
                handle_size,
                handle_size,
            ),
            "bottom": pygame.Rect(
                rect.centerx - handle_size // 2,
                rect.bottom - handle_size // 2,
                handle_size,
                handle_size,
            ),
            "left": pygame.Rect(
                rect.left - handle_size // 2,
                rect.centery - handle_size // 2,
                handle_size,
                handle_size,
            ),
            "right": pygame.Rect(
                rect.right - handle_size // 2,
                rect.centery - handle_size // 2,
                handle_size,
                handle_size,
            ),
        }
        return handles

    def get_handle_at(self, pos, obstacle):
        """Check if position is over a handle"""
        handles = self.get_handle_rects(obstacle)
        for handle_name, handle_rect in handles.items():
            if handle_rect.collidepoint(pos):
                return handle_name
        return None

    def handle_event(self, event):
        """Handle mouse events"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos

            # Check if clicked inside arena
            if not (
                ARENA_X <= pos[0] <= ARENA_X + WALL_WIDTH
                and ARENA_Y <= pos[1] <= ARENA_Y + WALL_HEIGHT
            ):
                return

            # LEFT CLICK - Select/Edit or Add Static
            if event.button == 1:
                # If input is active, handle text input
                if self.input_active:
                    # Clicking outside the input box closes it
                    if not self.input_rect or not self.input_rect.collidepoint(pos):
                        self.input_active = False
                        self.input_obstacle = None
                        self.input_text = ""
                        # Unfreeze the obstacle
                        if isinstance(self.input_obstacle, DynamicObstacle):
                            self.input_obstacle.frozen = False
                    return

                # Cancel dynamic creation on left click
                self.right_click_count = 0
                self.show_start_indicator = False

                # First check if clicking on selected obstacle's handle
                if self.selected_obstacle:
                    handle = self.get_handle_at(pos, self.selected_obstacle)
                    if handle:
                        self.resizing = True
                        self.resize_edge = handle
                        return

                # Check if clicking on obstacle
                obs = self.get_obstacle_at(pos)
                if obs:
                    self.selected_obstacle = obs
                    # If dynamic obstacle, freeze it and show input
                    if isinstance(obs, DynamicObstacle):
                        obs.frozen = True  # Freeze the obstacle
                        self.start_input(obs)
                    else:
                        handle = self.get_handle_at(pos, obs)
                        if handle:
                            self.resizing = True
                            self.resize_edge = handle
                        else:
                            self.dragging = True
                            self.drag_offset = (pos[0] - obs["x"], pos[1] - obs["y"])
                else:
                    # Click outside - unfreeze any frozen dynamic obstacle
                    for o in self.obstacles:
                        if isinstance(o, DynamicObstacle) and o.frozen:
                            o.frozen = False
                    # Add static obstacle at click position
                    self.add_static_obstacle(
                        pos[0] - 25, pos[1] - 25, 50, 50, f"Obstacle {self.next_id}"
                    )

            # RIGHT CLICK - Delete or Create Dynamic
            elif event.button == 3:
                # If input is active, close it
                if self.input_active:
                    self.input_active = False
                    self.input_obstacle = None
                    self.input_text = ""
                    # Unfreeze the obstacle
                    if isinstance(self.input_obstacle, DynamicObstacle):
                        self.input_obstacle.frozen = False
                    return

                # Check if clicking on obstacle
                obs = self.get_obstacle_at(pos)
                if obs:
                    # If it's a dynamic obstacle, unfreeze it before deletion
                    if isinstance(obs, DynamicObstacle):
                        obs.frozen = False
                    self.remove_obstacle(obs)
                    self.right_click_count = 0
                    self.show_start_indicator = False
                else:
                    # Double right-click to create dynamic obstacle (No timeout limit)
                    if self.right_click_count == 0:
                        self.right_click_count = 1
                        self.right_click_pos = pos
                        # Show start indicator
                        self.show_start_indicator = True
                        self.start_indicator_pos = pos
                    elif self.right_click_count == 1:
                        # Second click - create dynamic obstacle
                        if self.right_click_pos:
                            # Use first click as start, second as end
                            start_x, start_y = self.right_click_pos
                            end_x, end_y = pos
                            # Minimum distance check
                            if math.hypot(end_x - start_x, end_y - start_y) > 20:
                                self.add_dynamic_obstacle(
                                    start_x, start_y, end_x, end_y
                                )
                        self.right_click_count = 0
                        self.show_start_indicator = False

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
                self.resizing = False
                self.resize_edge = None

        elif event.type == pygame.MOUSEMOTION:
            pos = event.pos

            # Update hover
            if not self.dragging and not self.resizing and not self.input_active:
                if self.selected_obstacle:
                    handle = self.get_handle_at(pos, self.selected_obstacle)
                    if handle:
                        self.hover_handle = handle
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZENWSE)
                        self.hover_obstacle = None
                        return

                self.hover_handle = None
                self.hover_obstacle = self.get_obstacle_at(pos)
                if self.hover_obstacle:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                else:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

            # Drag obstacle
            if self.dragging and self.selected_obstacle:
                x = pos[0] - self.drag_offset[0]
                y = pos[1] - self.drag_offset[1]
                x = round(x / self.snap_grid) * self.snap_grid
                y = round(y / self.snap_grid) * self.snap_grid

                if isinstance(self.selected_obstacle, dict):
                    x = max(
                        ARENA_X,
                        min(ARENA_X + WALL_WIDTH - self.selected_obstacle["width"], x),
                    )
                    y = max(
                        ARENA_Y,
                        min(
                            ARENA_Y + WALL_HEIGHT - self.selected_obstacle["height"], y
                        ),
                    )
                    self.selected_obstacle["x"] = x
                    self.selected_obstacle["y"] = y
                else:
                    x = max(
                        ARENA_X,
                        min(ARENA_X + WALL_WIDTH - self.selected_obstacle.width, x),
                    )
                    y = max(
                        ARENA_Y,
                        min(ARENA_Y + WALL_HEIGHT - self.selected_obstacle.height, y),
                    )
                    self.selected_obstacle.x = x
                    self.selected_obstacle.y = y

            # Resize obstacle
            if self.resizing and self.selected_obstacle:
                obs = self.selected_obstacle
                x, y = pos
                x = round(x / self.snap_grid) * self.snap_grid
                y = round(y / self.snap_grid) * self.snap_grid

                if isinstance(obs, dict):
                    if "left" in self.resize_edge:
                        new_width = obs["x"] + obs["width"] - x
                        if new_width >= 20:
                            obs["x"] = x
                            obs["width"] = new_width
                    if "right" in self.resize_edge:
                        new_width = x - obs["x"]
                        if new_width >= 20:
                            obs["width"] = new_width
                    if "top" in self.resize_edge:
                        new_height = obs["y"] + obs["height"] - y
                        if new_height >= 20:
                            obs["y"] = y
                            obs["height"] = new_height
                    if "bottom" in self.resize_edge:
                        new_height = y - obs["y"]
                        if new_height >= 20:
                            obs["height"] = new_height
                else:
                    if "left" in self.resize_edge:
                        new_width = obs.x + obs.width - x
                        if new_width >= 20:
                            obs.x = x
                            obs.width = new_width
                    if "right" in self.resize_edge:
                        new_width = x - obs.x
                        if new_width >= 20:
                            obs.width = new_width
                    if "top" in self.resize_edge:
                        new_height = obs.y + obs.height - y
                        if new_height >= 20:
                            obs.y = y
                            obs.height = new_height
                    if "bottom" in self.resize_edge:
                        new_height = y - obs.y
                        if new_height >= 20:
                            obs.height = new_height

        # Keyboard events for input
        elif event.type == pygame.KEYDOWN and self.input_active:
            if event.key == pygame.K_RETURN:
                # Confirm input
                try:
                    seconds = float(self.input_text)
                    if seconds > 0:
                        self.input_obstacle.seconds = seconds
                        # Unfreeze the obstacle after setting time
                        if isinstance(self.input_obstacle, DynamicObstacle):
                            self.input_obstacle.frozen = False
                        print(f"Set dynamic obstacle to {seconds} seconds")
                except ValueError:
                    pass
                self.input_active = False
                self.input_obstacle = None
                self.input_text = ""
            elif event.key == pygame.K_ESCAPE:
                # Unfreeze on escape
                if isinstance(self.input_obstacle, DynamicObstacle):
                    self.input_obstacle.frozen = False
                self.input_active = False
                self.input_obstacle = None
                self.input_text = ""
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            else:
                if event.unicode.isdigit() or event.unicode == ".":
                    self.input_text += event.unicode

    def start_input(self, obstacle):
        """Start input for dynamic obstacle seconds"""
        self.input_active = True
        self.input_obstacle = obstacle
        self.input_text = str(obstacle.seconds)

    def draw_input_box(self, screen):
        """Draw input box for seconds"""
        if not self.input_active or not self.input_obstacle:
            return

        # Position near the obstacle
        if isinstance(self.input_obstacle, DynamicObstacle):
            rect = self.input_obstacle.get_rect()
            x = rect.centerx - 50
            y = rect.bottom + 10

            # Draw background
            self.input_rect = pygame.Rect(x, y, 100, 30)
            pygame.draw.rect(screen, WHITE, self.input_rect)
            pygame.draw.rect(screen, BLACK, self.input_rect, 2)

            # Draw label
            font = pygame.font.Font(None, 14)
            label = font.render("Seconds:", True, BLACK)
            screen.blit(label, (x, y - 18))

            # Draw text
            text_surf = font.render(self.input_text, True, BLACK)
            screen.blit(text_surf, (x + 5, y + 7))

            # Draw cursor
            if int(time.time() * 2) % 2 == 0:
                cursor_x = x + 5 + text_surf.get_width()
                pygame.draw.line(
                    screen, BLACK, (cursor_x, y + 5), (cursor_x, y + 25), 2
                )

    def draw_start_indicator(self, screen):
        """Draw indicator for dynamic obstacle start point"""
        if not self.show_start_indicator or not self.start_indicator_pos:
            return

        x, y = self.start_indicator_pos

        # Draw a simple box at the start position
        box_size = 10
        box_rect = pygame.Rect(x - box_size // 2, y - box_size // 2, box_size, box_size)
        pygame.draw.rect(screen, (0, 200, 0), box_rect)
        pygame.draw.rect(screen, (0, 100, 0), box_rect, 2)

        # Draw text
        font = pygame.font.Font(None, 20)
        label = font.render("Right click another area", True, (0, 150, 0))
        label_rect = label.get_rect(center=(x, y - 20))

        # Background for text
        bg_rect = label_rect.inflate(10, 4)
        pygame.draw.rect(screen, WHITE, bg_rect)
        pygame.draw.rect(screen, (0, 150, 0), bg_rect, 1)
        screen.blit(label, label_rect)

    def draw(self, screen):
        """Draw the environment"""
        screen.fill(WHITE)

        # Draw U-Shaped walls
        wall_rects = [
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

        for wall in wall_rects:
            pygame.draw.rect(screen, BLACK, wall)

        # Draw arena border
        pygame.draw.rect(screen, BLACK, (ARENA_X, ARENA_Y, WALL_WIDTH, WALL_HEIGHT), 2)

        # Draw grid
        for x in range(ARENA_X, ARENA_X + WALL_WIDTH, 20):
            pygame.draw.line(
                screen, (240, 240, 240), (x, ARENA_Y), (x, ARENA_Y + WALL_HEIGHT), 1
            )
        for y in range(ARENA_Y, ARENA_Y + WALL_HEIGHT, 20):
            pygame.draw.line(
                screen, (240, 240, 240), (ARENA_X, y), (ARENA_X + WALL_WIDTH, y), 1
            )

        # Draw dynamic obstacle paths (start/end markers)
        for obs in self.obstacles:
            if isinstance(obs, DynamicObstacle):
                # Draw path line
                pygame.draw.line(
                    screen,
                    (200, 200, 255),
                    (obs.start_x + obs.width // 2, obs.start_y + obs.height // 2),
                    (obs.end_x + obs.width // 2, obs.end_y + obs.height // 2),
                    2,
                )

        # Draw obstacles
        for obs in self.obstacles:
            if isinstance(obs, dict):
                rect = pygame.Rect(obs["x"], obs["y"], obs["width"], obs["height"])
                # Determine color
                if obs == self.selected_obstacle:
                    color = RED
                    border_color = RED
                    border_width = 3
                elif obs == self.hover_obstacle:
                    color = ORANGE
                    border_color = ORANGE
                    border_width = 2
                else:
                    color = BLACK
                    border_color = BLACK
                    border_width = 1

                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, border_color, rect, border_width)

                # Draw resize handles if selected
                if obs == self.selected_obstacle and not isinstance(
                    obs, DynamicObstacle
                ):
                    handles = self.get_handle_rects(obs)
                    for handle_name, handle_rect in handles.items():
                        if handle_name == self.hover_handle:
                            color = GREEN
                        else:
                            color = BLUE
                        pygame.draw.rect(screen, color, handle_rect)
                        pygame.draw.rect(screen, BLACK, handle_rect, 1)

            else:  # DynamicObstacle
                rect = obs.get_rect()
                # Update position if moving and not frozen
                if obs.moving and not obs.frozen:
                    current_time = time.time()
                    dt = current_time - getattr(obs, "_last_update", current_time)
                    obs._last_update = current_time
                    obs.update(dt)

                # Draw dynamic obstacle
                if obs == self.selected_obstacle:
                    color = RED
                elif obs == self.hover_obstacle:
                    color = ORANGE
                elif obs.frozen:
                    color = (255, 200, 100)  # Yellow-orange for frozen state
                else:
                    color = (0, 100, 200)  # Blue

                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLACK, rect, 2)

                # Draw resize handles if selected
                if obs == self.selected_obstacle:
                    handles = self.get_handle_rects(obs)
                    for handle_name, handle_rect in handles.items():
                        if handle_name == self.hover_handle:
                            color = GREEN
                        else:
                            color = BLUE
                        pygame.draw.rect(screen, color, handle_rect)
                        pygame.draw.rect(screen, BLACK, handle_rect, 1)

        # Draw start indicator for dynamic obstacle creation
        self.draw_start_indicator(screen)

        # Draw input box
        self.draw_input_box(screen)

        # Draw help text
        font = pygame.font.Font(None, 14)
        help_texts = [
            "Left Click (empty) = Add static obstacle",
            "Right Click x2 (empty) = Create dynamic obstacle",
            "Left Click (dynamic) = Freeze & Set seconds",
            "Right Click (obstacle) = Delete",
            "S = Save, ESC = Quit",
        ]
        for i, text in enumerate(help_texts):
            surf = font.render(text, True, (180, 180, 180))
            screen.blit(surf, (10, HEIGHT - 20 - i * 18))

    def save_obstacles(self, filename=OBSTACLE_FILE):
        """Save obstacles to file"""
        obstacle_list = []
        for obs in self.obstacles:
            if isinstance(obs, dict):
                obstacle_list.append(obs)
            else:
                obstacle_list.append(obs.to_dict())

        data = {
            "version": "1.0",
            "obstacles": obstacle_list,
            "arena": {
                "width": WALL_WIDTH,
                "height": WALL_HEIGHT,
                "x": ARENA_X,
                "y": ARENA_Y,
            },
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        print(f"✓ Saved {len(obstacle_list)} obstacles to {filename}")
        return True

    def load_obstacles(self, filename=OBSTACLE_FILE):
        """Load obstacles from file"""
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)

                if "obstacles" in data:
                    self.obstacles = []
                    for obs_data in data["obstacles"]:
                        if obs_data.get("is_dynamic", False):
                            obs = DynamicObstacle.from_dict(obs_data)
                            obs.moving = True
                            obs.frozen = False
                            obs._last_update = time.time()
                            self.obstacles.append(obs)
                        else:
                            self.obstacles.append(obs_data)
                    return True
            except Exception as e:
                print(f"Error loading obstacles: {e}")
        return False


# ====================================================
# Main Function
# ====================================================


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Obstacle Editor - Dynamic Obstacles")
    clock = pygame.time.Clock()

    editor = ObstacleEditor()

    running = True

    print("=" * 60)
    print("OBSTACLE EDITOR - DYNAMIC OBSTACLES")
    print("=" * 60)
    print("Controls:")
    print("  Left Click (empty)     = Add static obstacle")
    print("  Right Click x2 (empty) = Create dynamic obstacle")
    print("  Left Click (dynamic)   = Freeze & Set seconds")
    print("  Drag                   = Move selected obstacle")
    print("  Drag BLUE handles      = Resize obstacle")
    print("  Right Click (obstacle) = Delete obstacle")
    print("  S key                  = Save to file")
    print("  ESC key                = Quit")
    print("=" * 60)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    editor.save_obstacles()
                elif event.key == pygame.K_ESCAPE:
                    running = False
            editor.handle_event(event)

        editor.draw(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
