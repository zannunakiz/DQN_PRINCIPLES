import pygame
import sys
import math
from constant import *
from DQN import Environment, get_sensor_values


def draw_scene(screen, env, show_sensors=True):
    screen.fill(WHITE)

    # Draw U-shaped walls
    for wall in env.wall_rects:
        pygame.draw.rect(screen, BLACK, wall)

    # Draw obstacles
    for obs_rect in env.obstacle_rects:
        pygame.draw.rect(screen, BLACK, obs_rect)
        pygame.draw.rect(screen, DARK_GRAY, obs_rect, 2)

    # Draw sensor rays and values
    sensors = get_sensor_values(
        env.agent_x, env.agent_y, env.agent_angle, env.all_edges
    )
    sensor_font = pygame.font.Font(None, 18)

    large_font = pygame.font.Font(None, 22)
    for i, rel_angle in enumerate(SENSOR_ANGLES):
        rad = math.radians(env.agent_angle + rel_angle)
        end_x = env.agent_x + math.sin(rad) * MAX_RANGE
        end_y = env.agent_y - math.cos(rad) * MAX_RANGE

        # Draw sensor ray (always visible)
        pygame.draw.line(screen, GREEN, (env.agent_x, env.agent_y), (end_x, end_y), 1)

        # Draw sensor value label (toggle with I key)
        if show_sensors:
            label = f"{i + 1}: {sensors[i]:.2f}"
            label_surf = large_font.render(label, True, ORANGE)
            label_x = env.agent_x + math.sin(rad) * (MAX_RANGE * 0.85)
            label_y = env.agent_y - math.cos(rad) * (MAX_RANGE * 0.85)
            text_rect = label_surf.get_rect()
            text_rect.topleft = (label_x, label_y)
            pygame.draw.rect(screen, BLACK, text_rect.inflate(4, 4))
            screen.blit(label_surf, (label_x, label_y))

    # Draw agent
    agent_rect = env.get_agent_rect()
    pygame.draw.rect(screen, BLUE, agent_rect)
    pygame.draw.rect(screen, BLACK, agent_rect, 2)

    # Draw heading line
    heading_rad = math.radians(env.agent_angle)
    head_x = env.agent_x + math.sin(heading_rad) * (AGENT_SIZE // 2 + 12)
    head_y = env.agent_y - math.cos(heading_rad) * (AGENT_SIZE // 2 + 12)
    pygame.draw.line(screen, YELLOW, (env.agent_x, env.agent_y), (head_x, head_y), 3)

    # Input display in the upper-right
    font = pygame.font.Font(None, 16)
    sensors = get_sensor_values(
        env.agent_x, env.agent_y, env.agent_angle, env.all_edges
    )
    angle = round(env.agent_angle / TURN_ANGLE) * TURN_ANGLE
    angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
    compass = int((angle + 90) // TURN_ANGLE + 1)
    prev_input = 0.0 if env.prev_action is None else float(env.prev_action + 1)
    input_values = [compass] + sensors + [prev_input]
    input_text = f"Input: [{', '.join([f'{v:.2f}' if isinstance(v, float) else str(v) for v in input_values])}]"
    input_surf = font.render(input_text, True, DARK_GRAY)
    input_x = ARENA_X + WALL_WIDTH - input_surf.get_width() - 10
    input_y = ARENA_Y + 10
    screen.blit(input_surf, (input_x, input_y))

    # Draw info text at bottom
    info_texts = [
        f"Controls: UP=Forward, LEFT=Turn Left, RIGHT=Turn Right, R=Reset",
        f"Sensors: {'VISIBLE' if show_sensors else 'HIDDEN'} (Press I to toggle)",
        f"Steps: {env.steps}/{MAX_STEPS}",
    ]
    for i, text in enumerate(info_texts):
        surf = font.render(text, True, ORANGE)
        screen.blit(surf, (10, HEIGHT - 60 + i * 18))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Manual Control - Press I to toggle sensor values")
    clock = pygame.time.Clock()

    env = Environment(OBSTACLE_FILE)
    env.instant_move = True
    state = env.reset()

    show_sensors = True
    running = True

    print("=" * 60)
    print("MANUAL CONTROL MODE")
    print("Controls:")
    print("  UP ARROW    - Move Forward")
    print("  LEFT ARROW  - Turn Left")
    print("  RIGHT ARROW - Turn Right")
    print("  R - Reset Episode")
    print("  I - Toggle Sensor Values (Show/Hide)")
    print("  ESC - Quit")
    print("=" * 60)

    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0  # Forward
                elif event.key == pygame.K_LEFT:
                    action = 1  # Turn Left
                elif event.key == pygame.K_RIGHT:
                    action = 2  # Turn Right
                elif event.key == pygame.K_r:
                    state = env.reset()
                    action = None
                    print("Episode reset")
                elif event.key == pygame.K_i:
                    show_sensors = not show_sensors
                    print(f"Sensor values: {'VISIBLE' if show_sensors else 'HIDDEN'}")
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if action is not None and not env.done:
            state, _, done, info = env.step(action)
            if done:
                status = (
                    "SUCCESS"
                    if info["success"]
                    else ("COLLISION" if info["collision"] else "TIMEOUT")
                )
                print(f"Episode ended: {status} (Steps: {env.steps})")
                state = env.reset()

        draw_scene(screen, env, show_sensors)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
