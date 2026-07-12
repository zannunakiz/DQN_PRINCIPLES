# ====================================================
# main_Load_Model.py - VISUALIZATION LOAD MODEL MODE
# Refactored to use core logic from DQN.py
# With Fast Forward, Slow Mo, and Pause Controls
# ====================================================

import pygame
import torch
import numpy as np
import os
import sys
import math
import time

from constant import *

# Use the variant script directory as base for local models and obstacle files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = (
    LOAD_MODEL if os.path.isabs(LOAD_MODEL) else os.path.join(BASE_DIR, LOAD_MODEL)
)
LOCAL_OBSTACLE_PATH = (
    OBSTACLE_FILE
    if os.path.isabs(OBSTACLE_FILE)
    else os.path.join(BASE_DIR, OBSTACLE_FILE)
)

# Import core components from DQN.py
from DQN import Environment, Agent, get_sensor_values, line_intersection

# ====================================================
# Visualization Functions (Local override for extra UI features)
# ====================================================


def draw_environment(
    screen,
    env,
    episode_num,
    total_reward,
    last_action,
    clock,
    action_names,
    fps_counter,
    speed_mode,
    is_paused,
    show_sensor_info,
    q_values=None,
):
    screen.fill(WHITE)

    # Draw U-Shaped Walls
    for wall in env.wall_rects:
        pygame.draw.rect(screen, BLACK, wall)

    # Finish line
    pygame.draw.rect(screen, GREEN, (ARENA_X, FINISH_Y, WALL_WIDTH, FINISH_THICKNESS))

    # Draw dynamic obstacle paths (if any exist from DQN.py Environment)
    if hasattr(env, "dynamic_obstacles"):
        for dyn in env.dynamic_obstacles:
            pygame.draw.line(
                screen,
                (200, 200, 255),
                (dyn.start_x + dyn.width // 2, dyn.start_y + dyn.height // 2),
                (dyn.end_x + dyn.width // 2, dyn.end_y + dyn.height // 2),
                2,
            )

    # Draw obstacles (static and dynamic)
    if hasattr(env, "dynamic_indices"):
        for i, obs_rect in enumerate(env.obstacle_rects):
            is_dynamic = i in env.dynamic_indices
            color = (0, 100, 200) if is_dynamic else (BLACK)
            pygame.draw.rect(screen, color, obs_rect)
            pygame.draw.rect(screen, DARK_GRAY if is_dynamic else BLACK, obs_rect, 2)
    else:
        for obs_rect in env.obstacle_rects:
            pygame.draw.rect(screen, BLACK, obs_rect)
            pygame.draw.rect(screen, DARK_GRAY, obs_rect, 2)

    # Sensor rays
    sensor_colors = [GREEN] * len(SENSOR_ANGLES)
    for rel_angle, color in zip(SENSOR_ANGLES, sensor_colors):
        abs_angle = env.agent_angle + rel_angle
        rad = math.radians(abs_angle)
        end_x = env.agent_x + math.sin(rad) * MAX_RANGE
        end_y = env.agent_y - math.cos(rad) * MAX_RANGE
        pygame.draw.line(screen, color, (env.agent_x, env.agent_y), (end_x, end_y), 2)

    # Agent body
    agent_rect = env.get_agent_rect()
    pygame.draw.rect(screen, BLUE, agent_rect)
    pygame.draw.rect(screen, BLACK, agent_rect, 2)

    # Agent heading direction
    heading_rad = math.radians(env.agent_angle)
    head_x = env.agent_x + math.sin(heading_rad) * (AGENT_SIZE // 2 + 15)
    head_y = env.agent_y - math.cos(heading_rad) * (AGENT_SIZE // 2 + 15)
    pygame.draw.line(screen, YELLOW, (env.agent_x, env.agent_y), (head_x, head_y), 4)

    # Status indicator
    status_text = "RUNNING"
    status_color = BLUE
    if env.done:
        if env.success:
            status_text = "✓ SUCCESS!"
            status_color = GREEN
        elif env.collision:
            status_text = "✗ COLLISION!"
            status_color = RED
        else:
            status_text = "⊗ TIMEOUT"
            status_color = ORANGE

    # Speed indicator
    speed_text = {
        "normal": "▶ Normal (1x)",
        "fast": "⏩ Fast (3x)",
        "slow": "⏪ Slow (0.5x)",
    }.get(speed_mode, "Normal")

    if is_paused:
        speed_text = "⏸ PAUSED"

    # UI Text Info
    small_font = pygame.font.Font(None, 16)
    angle_value = round(env.agent_angle / TURN_ANGLE) * TURN_ANGLE
    angle_value = max(MIN_ANGLE, min(MAX_ANGLE, angle_value))
    compass = int((angle_value + 90) // TURN_ANGLE + 1)
    sensors = get_sensor_values(
        env.agent_x, env.agent_y, env.agent_angle, env.all_edges
    )

    # Display stats on left side
    stats_y = ARENA_Y + 10
    texts = [
        f"Steps: {env.steps}/{MAX_STEPS}",
        f"Reward: {total_reward:.2f}",
        f"Action: {action_names.get(last_action, 'None')}",
        f"Status: {status_text}",
    ]

    for i, text in enumerate(texts):
        surf = small_font.render(text, True, BLACK)
        screen.blit(surf, (ARENA_X + 5, stats_y + i * 20))

    prev_input = 0.0 if env.prev_action is None else float(env.prev_action + 1)
    input_values = [compass] + sensors + [prev_input]

    # Input values in the upper-right of the arena
    input_text = f"Input: [{', '.join([f'{v:.2f}' if isinstance(v, float) else str(v) for v in input_values])}]"
    input_surf = small_font.render(input_text, True, DARK_GRAY)
    input_x = ARENA_X + WALL_WIDTH - input_surf.get_width() - 10
    input_y = ARENA_Y + 10
    screen.blit(input_surf, (input_x, input_y))
    large_font = pygame.font.Font(None, 22)  # or 30, or whatever size you want
    if show_sensor_info:
        # Draw per-sensor labels and values at the end of each sensor ray
        for idx, rel_angle in enumerate(SENSOR_ANGLES):
            abs_angle = env.agent_angle + rel_angle
            rad = math.radians(abs_angle)
            end_x = env.agent_x + math.sin(rad) * MAX_RANGE
            end_y = env.agent_y - math.cos(rad) * MAX_RANGE
            label = f"{idx + 1}: {sensors[idx]:.2f}"
            label_surf = large_font.render(label, True, ORANGE)

            label_x = env.agent_x + math.sin(rad) * (MAX_RANGE * 0.85)
            label_y = env.agent_y - math.cos(rad) * (MAX_RANGE * 0.85)

            # Buat background hitam di belakang teks
            text_rect = label_surf.get_rect()
            text_rect.topleft = (label_x, label_y)
            pygame.draw.rect(
                screen, (0, 0, 0), text_rect.inflate(4, 4)
            )  # inflate untuk padding

            screen.blit(label_surf, (label_x, label_y))

        info_text = ""
        info_surf = small_font.render(info_text, True, DARK_GRAY)
        screen.blit(info_surf, (input_x, input_y + input_surf.get_height() + 6))

    # Display Q-values below the sensor input values
    if q_values is not None:
        q_text = f"Q: [{', '.join([f'{q:.2f}' for q in q_values])}]"
        q_surf = small_font.render(q_text, True, DARK_GRAY)
        q_x = ARENA_X + WALL_WIDTH - q_surf.get_width() - 10
        q_y = input_y + input_surf.get_height() + 6
        screen.blit(q_surf, (q_x, q_y))

    # Controls info (bottom right)
    controls = [
        "Controls:",
        "1: Normal  |  2: Fast  |  3: Slow",
        "P: Pause  |  SPACE: Reset",
        "R: Reset Arena  |  I: Toggle Sensors",
        "ESC: Quit",
    ]
    controls_y = ARENA_Y + WALL_HEIGHT - 70
    for i, text in enumerate(controls):
        surf = small_font.render(text, True, DARK_GRAY)
        screen.blit(surf, (ARENA_X + 5, controls_y + i * 16))


# ====================================================
# Main Visualization
# ====================================================


def visualize_model():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DQN Robot Navigation - Visualization")
    clock = pygame.time.Clock()

    print("=" * 60)
    print("VISUALIZATION MODE - Load Model & Custom Obstacles")
    print("U-Shaped Arena with custom obstacles")
    print("Controls:")
    print("  1 - Normal Speed (1x)")
    print("  2 - Fast Forward (3x)")
    print("  3 - Slow Motion (0.5x)")
    print("  P - Pause/Resume")
    print("  SPACE - Reset Episode")
    print("  R - Reset Arena")
    print("  I - Toggle Sensor Info")
    print("  ESC - Quit")
    print("=" * 60)

    # Load environment with custom obstacles from the variant folder
    env = Environment(LOCAL_OBSTACLE_PATH)
    env.instant_move = True  # Match headless training mode for exact model behavior

    model_path = LOCAL_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"ERROR: Model file '{model_path}' not found!")
        print("Please train the model first.")
        sys.exit(1)

    # Load agent (from DQN.py)
    agent = Agent(load_model=True, model_path=model_path)
    agent.epsilon = 0.0  # Pure greedy policy for visualization (no random exploration)
    print(f"Model loaded successfully from {model_path}")

    action_names = {0: "Forward", 1: "Turn Left", 2: "Turn Right"}

    # Speed control variables
    speed_modes = {
        "normal": 1.0,
        "fast": 3.0,
        "slow": 0.5,
        "super_slow": 0.1,
    }
    current_speed = "normal"
    is_paused = False
    show_sensor_info = True

    # Episode variables
    episode_num = 1
    total_reward = 0
    last_action = None
    running = True
    fps_counter = 0
    frame_count = 0

    state = env.reset()

    # Statistics
    episode_stats = {
        "total_episodes": 0,
        "success_count": 0,
        "collision_count": 0,
        "timeout_count": 0,
    }

    # Frame delay for speed control
    speed_frames = 0
    frame_step_counter = 0  # Counter untuk melacak step per frame

    # Waktu untuk dynamic obstacle update
    last_dynamic_update = time.time()
    dynamic_time_accumulator = 0.0

    # Store last known Q-values to display during transitions
    last_q_values = None

    print(
        f"Loaded {len(env.obstacle_rects)} obstacles ({len(env.dynamic_obstacles)} dynamic)"
    )
    print("Starting visualization...")

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Keyboard shortcuts
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    current_speed = "normal"
                    print("Speed: Normal (1x)")
                elif event.key == pygame.K_2:
                    current_speed = "fast"
                    print("Speed: Fast Forward (3x)")
                elif event.key == pygame.K_3:
                    current_speed = "slow"
                    print("Speed: Slow Motion (0.5x)")
                elif event.key == pygame.K_4:
                    current_speed = "super_slow"
                    print("Speed: Super Slow Motion (0.25x)")
                elif event.key == pygame.K_p:
                    is_paused = not is_paused
                    print("Paused" if is_paused else "Resumed")
                    if is_paused:
                        last_dynamic_update = time.time()  # Reset timer saat pause
                elif event.key == pygame.K_SPACE:
                    state = env.reset()
                    total_reward = 0
                    last_action = None
                    last_dynamic_update = time.time()
                    dynamic_time_accumulator = 0.0
                    last_q_values = None
                    print("Episode reset")
                elif event.key == pygame.K_r:
                    env = Environment(LOCAL_OBSTACLE_PATH)
                    env.instant_move = False
                    state = env.reset()
                    total_reward = 0
                    last_action = None
                    last_dynamic_update = time.time()
                    dynamic_time_accumulator = 0.0
                    last_q_values = None
                    print("Arena reset")
                elif event.key == pygame.K_i:
                    show_sensor_info = not show_sensor_info
                    print(f"Sensor info: {'VISIBLE' if show_sensor_info else 'HIDDEN'}")
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # Skip update if paused
        if is_paused:
            draw_environment(
                screen,
                env,
                episode_num,
                total_reward,
                last_action,
                clock,
                action_names,
                fps_counter,
                current_speed,
                is_paused,
                show_sensor_info,
                last_q_values,  # Show frozen Q-values during pause
            )
            pygame.display.flip()
            clock.tick(FPS)
            continue

        # Speed control
        speed_multiplier = speed_modes[current_speed]

        # Helper function to get action and q_values manually
        def get_action_and_q():
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values_tensor = agent.policy_net(state_tensor)
                q_vals = q_values_tensor.cpu().numpy().flatten()
                act = q_vals.argmax().item()
                return act, q_vals

        # Update dynamic obstacles based on speed multiplier
        # Get current time and calculate dt with speed multiplier
        current_time = time.time()
        dt = current_time - last_dynamic_update
        last_dynamic_update = current_time

        # Apply speed multiplier to dt (fast = 3x, slow = 0.5x)
        dt_scaled = dt * speed_multiplier

        # Update dynamic obstacles with scaled dt
        if not env.done:
            for dyn in env.dynamic_obstacles:
                if dyn.is_dynamic and dyn.moving:
                    # Update each dynamic obstacle with scaled dt
                    dyn.last_update = current_time  # Bypass internal timer
                    # Manually update position with scaled dt
                    total_distance = math.hypot(
                        dyn.end_x - dyn.start_x, dyn.end_y - dyn.start_y
                    )
                    if total_distance > 0:
                        speed = total_distance / dyn.seconds
                        distance_to_move = speed * dt_scaled
                        dyn.progress += (
                            distance_to_move / total_distance
                        ) * dyn.direction

                        if dyn.progress >= 1.0:
                            dyn.progress = 1.0
                            dyn.direction = -1
                        elif dyn.progress <= 0.0:
                            dyn.progress = 0.0
                            dyn.direction = 1

                        dyn.x = dyn.start_x + (dyn.end_x - dyn.start_x) * dyn.progress
                        dyn.y = dyn.start_y + (dyn.end_y - dyn.start_y) * dyn.progress

                        # Update obstacle rect
                        for i, rect_idx in enumerate(env.dynamic_indices):
                            if i < len(env.dynamic_obstacles):
                                env.obstacle_rects[rect_idx] = dyn.get_rect()

            # Recalculate edges after dynamic updates
            env.all_edges = env.calculate_edges()

        # Agent movement with speed control
        if speed_multiplier >= 1.0:
            # Fast forward: take multiple steps per frame
            steps_per_frame = int(speed_multiplier)
            for _ in range(steps_per_frame):
                if not env.done:
                    if env.turning or env.moving:
                        action = None
                        # Keep last_q_values frozen during transition
                        q_values_for_display = last_q_values
                    else:
                        action, q_vals = get_action_and_q()
                        last_q_values = q_vals  # Store latest Q-values
                        q_values_for_display = q_vals
                        last_action = action
                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    state = next_state

                    if done:
                        episode_stats["total_episodes"] += 1
                        if info["success"]:
                            episode_stats["success_count"] += 1
                            print(
                                f"✓ Episode {episode_num}: SUCCESS! Steps={env.steps}, Reward={total_reward:.2f}"
                            )
                        elif info["collision"]:
                            episode_stats["collision_count"] += 1
                            print(
                                f"✗ Episode {episode_num}: COLLISION at step {env.steps}, Reward={total_reward:.2f}"
                            )
                        else:
                            episode_stats["timeout_count"] += 1
                            print(
                                f"⊗ Episode {episode_num}: TIMEOUT, Reward={total_reward:.2f}"
                            )

                        episode_num += 1
                        state = env.reset()
                        total_reward = 0
                        last_action = None
                        last_q_values = None  # Reset Q-values for new episode
                        break
        else:
            # Slow motion: take 1 step every N frames
            speed_frames += 1
            frames_per_step = int(1.0 / speed_multiplier)

            if speed_frames >= frames_per_step:
                speed_frames = 0
                if not env.done:
                    if env.turning or env.moving:
                        action = None
                        # Keep last_q_values frozen during transition
                        q_values_for_display = last_q_values
                    else:
                        action, q_vals = get_action_and_q()
                        last_q_values = q_vals  # Store latest Q-values
                        q_values_for_display = q_vals
                        last_action = action
                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    state = next_state

                    if done:
                        episode_stats["total_episodes"] += 1
                        if info["success"]:
                            episode_stats["success_count"] += 1
                            print(
                                f"✓ Episode {episode_num}: SUCCESS! Steps={env.steps}, Reward={total_reward:.2f}"
                            )
                        elif info["collision"]:
                            episode_stats["collision_count"] += 1
                            print(
                                f"✗ Episode {episode_num}: COLLISION at step {env.steps}, Reward={total_reward:.2f}"
                            )
                        else:
                            episode_stats["timeout_count"] += 1
                            print(
                                f"⊗ Episode {episode_num}: TIMEOUT, Reward={total_reward:.2f}"
                            )

                        episode_num += 1
                        state = env.reset()
                        total_reward = 0
                        last_action = None
                        last_q_values = None  # Reset Q-values for new episode

        # Get Q-values for display
        # If agent is in transition, use frozen last_q_values
        if env.turning or env.moving:
            q_values_for_display = last_q_values
        else:
            # If not in transition, try to get fresh Q-values
            try:
                _, q_vals = get_action_and_q()
                last_q_values = q_vals
                q_values_for_display = q_vals
            except Exception:
                q_values_for_display = last_q_values

        # Draw environment
        draw_environment(
            screen,
            env,
            episode_num,
            total_reward,
            last_action,
            clock,
            action_names,
            fps_counter,
            current_speed,
            is_paused,
            show_sensor_info,
            q_values_for_display,  # Always show Q-values (frozen or fresh)
        )
        pygame.display.flip()

        # Update FPS counter
        frame_count += 1
        if frame_count % 30 == 0:
            fps_counter = int(clock.get_fps())

        clock.tick(FPS)

    # Final statistics
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Total Episodes: {episode_stats['total_episodes']}")
    if episode_stats["total_episodes"] > 0:
        print(
            f"Success Rate: {episode_stats['success_count']/episode_stats['total_episodes']*100:.1f}%"
        )
        print(
            f"Collision Rate: {episode_stats['collision_count']/episode_stats['total_episodes']*100:.1f}%"
        )
        print(
            f"Timeout Rate: {episode_stats['timeout_count']/episode_stats['total_episodes']*100:.1f}%"
        )
    print("=" * 60)

    pygame.quit()


if __name__ == "__main__":
    visualize_model()
