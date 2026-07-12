# ====================================================
# main_Train_UI.py - Visual Training Loop
# ====================================================

import pygame
import csv
import os
from constant import *
from DQN import Environment, Agent, draw_environment


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DQN Training - UI Mode")
    clock = pygame.time.Clock()

    env = Environment(OBSTACLE_FILE)
    env.instant_move = False  # UI uses smooth sub-stepping
    agent_obj = Agent()

    action_names = {0: "Forward", 1: "Turn Left", 2: "Turn Right"}
    episode_num, success_count = 0, 0
    slow_motion = False
    slow_fps = max(5, FPS // 6)

    csv_file = open("training_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    header = [
        "Episode",
        "Reward",
        "Steps",
        "Loss",
        "Avg_Q",
        "Epsilon",
        "Success",
        "Collision",
        "Termination",
        "Obstacle_Count",
        "Dynamic_Count",
    ]
    csv_writer.writerow(header)

    state = env.reset()
    total_reward, last_action = 0, None
    running = True

    print("=" * 60)
    print("TRAINING STARTED (UI MODE)")
    print("Controls:")
    print("  1 - Toggle Slow Motion")
    print("  S - Save Model Manually")
    print("=" * 60)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    slow_motion = not slow_motion
                    print(f"Slow Motion: {'ON' if slow_motion else 'OFF'}")
                elif event.key == pygame.K_s:
                    agent_obj.save_model(os.path.join("models", "DQN-manual.pth"))

        action = None if (env.turning or env.moving) else agent_obj.select_action(state)
        last_action = action

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if info.get("step_completed", False):
            agent_obj.memory.push(
                state, info.get("action", action), reward, next_state, done
            )

        state = next_state
        agent_obj.update()

        draw_environment(
            screen,
            env,
            agent_obj,
            episode_num,
            total_reward,
            last_action,
            clock,
            action_names,
            slow_motion,
        )
        pygame.display.flip()
        clock.tick(slow_fps if slow_motion else FPS)

        if done:
            episode_num += 1

            # Update statistics
            if info["success"]:
                success_count += 1
                status_text = "SUCCESS"
                status_symbol = "✓"
            elif info["collision"]:
                status_text = "COLLISION"
                status_symbol = "✗"
            else:
                status_text = "TIMEOUT"
                status_symbol = "⊗"

            success_rate = success_count / episode_num * 100 if episode_num > 0 else 0

            # Console log
            print(
                f"Episode {episode_num:4d} | "
                f"{status_symbol} {status_text:<9} | "
                f"Steps: {env.steps:3d} | "
                f"Reward: {total_reward:7.2f} | "
                f"Eps: {agent_obj.epsilon:.3f} | "
                f"Loss: {agent_obj.loss:.4f} | "
                f"Success: {success_count} ({success_rate:5.1f}%) | "
                f"Dynamic: {len(env.dynamic_obstacles)}"
            )

            # Write to CSV
            csv_writer.writerow(
                [
                    episode_num,
                    total_reward,
                    env.steps,
                    agent_obj.loss,
                    agent_obj.avg_q,
                    agent_obj.epsilon,
                    int(info["success"]),
                    int(info["collision"]),
                    info["termination_reason"],
                    len(env.obstacle_rects),
                    len(env.dynamic_obstacles),
                ]
            )
            csv_file.flush()

            if episode_num % TARGET_UPDATE == 0:
                agent_obj.update_target_network()

            if INDEPENDENT_BASED:
                if info["success"]:
                    print(
                        "✔ Episode succeeded; running independent evaluation with epsilon=0..."
                    )
                    eval_agent = Agent(device=agent_obj.device)
                    eval_agent.policy_net.load_state_dict(
                        agent_obj.policy_net.state_dict()
                    )
                    eval_agent.target_net.load_state_dict(
                        eval_agent.policy_net.state_dict()
                    )
                    eval_agent.epsilon = 0.0
                    eval_env = Environment(OBSTACLE_FILE)
                    eval_env.instant_move = env.instant_move

                    eval_state = eval_env.reset()
                    eval_done = False
                    while not eval_done:
                        eval_action = eval_agent.select_action(eval_state)
                        _, _, eval_done, eval_info = eval_env.step(eval_action)
                        eval_state = eval_env.get_state()

                    if eval_info.get("success"):
                        best_path = LOAD_MODEL
                        agent_obj.save_model(best_path)
                        print(f"💾 [SAVED BEST] Model saved as {best_path}")
            else:
                if episode_num % MODEL_SAVE_INTERVAL == 0:
                    save_path = os.path.join("models", f"DQN-{episode_num}.pth")
                    agent_obj.save_model(save_path)
                    print(
                        f"💾 [SAVED] Model saved at Episode {episode_num} -> {save_path}"
                    )

            agent_obj.decay_epsilon()
            state, total_reward = env.reset(), 0

    if not INDEPENDENT_BASED:
        agent_obj.save_model(os.path.join("models", f"DQN-{episode_num}.pth"))
    csv_file.close()
    pygame.quit()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total Episodes: {episode_num}")
    if episode_num > 0:
        print(f"Success Rate: {success_count/episode_num*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
