import os
import time
import csv
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from constant import *
from DQN import Environment, Agent


def draw_path_plot(env, agent_positions, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(ARENA_X - 50, ARENA_X + WALL_WIDTH + 50)
    ax.set_ylim(ARENA_Y - 50, ARENA_Y + WALL_HEIGHT + 50)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    arena_rect = Rectangle(
        (ARENA_X, ARENA_Y),
        WALL_WIDTH,
        WALL_HEIGHT,
        fill=False,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(arena_rect)

    for wall in env.wall_rects:
        rect = Rectangle(
            (wall.x, wall.y),
            wall.width,
            wall.height,
            facecolor="black",
            edgecolor="black",
        )
        ax.add_patch(rect)

    for obs in env.obstacle_rects:
        rect = Rectangle(
            (obs.x, obs.y),
            obs.width,
            obs.height,
            facecolor="gray",
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(rect)
        ax.text(
            obs.x + obs.width / 2,
            obs.y + obs.height / 2,
            "O",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )

    finish_rect = Rectangle(
        (ARENA_X, FINISH_Y),
        WALL_WIDTH,
        FINISH_THICKNESS,
        facecolor="green",
        edgecolor="darkgreen",
        alpha=0.8,
    )
    ax.add_patch(finish_rect)
    ax.text(
        ARENA_X + WALL_WIDTH / 2,
        FINISH_Y + FINISH_THICKNESS / 2,
        "FINISH",
        ha="center",
        va="center",
        fontsize=12,
        color="white",
        fontweight="bold",
    )

    if agent_positions:
        positions = np.array(agent_positions, dtype=float)
        ax.plot(positions[:, 0], positions[:, 1], color="blue", linewidth=2, label="Agent Path")
        ax.plot(positions[0, 0], positions[0, 1], "go", markersize=8, label="Start")
        ax.plot(positions[-1, 0], positions[-1, 1], "rx", markersize=10, label="End")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_training_and_plot():
    print("=" * 60)
    print("HEADLESS TRAINING MODE WITH BEST-MODEL PATH PLOT")
    print("=" * 60)

    env = Environment(OBSTACLE_FILE)
    env.instant_move = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(device=device)

    episode_rewards, success_count, collision_count, timeout_count = [], 0, 0, 0

    training_episodes = int(os.getenv("TRAINING_EPISODES_OVERRIDE", TRAINING_EPISODES))
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
    ]
    csv_writer.writerow(header)

    start_time = time.time()

    for episode in range(1, training_episodes + 1):
        state = env.reset()
        total_reward, episode_steps = 0, 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            if info.get("step_completed", False):
                episode_steps += 1
                agent.memory.push(state, info.get("action", action), reward, next_state, done)

            state = next_state
            agent.update()
            if done:
                break

        if info["success"]:
            success_count += 1
        elif info["collision"]:
            collision_count += 1
        elif info["termination_reason"] == "Timeout":
            timeout_count += 1

        episode_rewards.append(total_reward)
        avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))

        csv_writer.writerow(
            [
                episode,
                total_reward,
                episode_steps,
                agent.loss,
                agent.avg_q,
                agent.epsilon,
                int(info["success"]),
                int(info["collision"]),
                info["termination_reason"],
                len(env.obstacle_rects),
            ]
        )
        csv_file.flush()

        status_symbol = "✓" if info["success"] else ("✗" if info["collision"] else "⊗")
        status_text = (
            "SUCCESS"
            if info["success"]
            else ("COLLISION" if info["collision"] else "TIMEOUT")
        )

        print(
            f"Ep {episode:4d}/{training_episodes} | {status_symbol} {status_text:<9} | "
            f"Steps: {episode_steps:3d} | Rew: {total_reward:7.2f} | Avg100: {avg_reward:7.2f} | "
            f"Eps: {agent.epsilon:.3f} | Loss: {agent.loss:.4f} | "
            f"SuccCount: {success_count} | SuccRate: {success_count/episode*100:5.1f}%"
        )

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        if INDEPENDENT_BASED:
            if info["success"]:
                eval_agent = Agent(device=agent.device)
                eval_agent.policy_net.load_state_dict(agent.policy_net.state_dict())
                eval_agent.target_net.load_state_dict(eval_agent.policy_net.state_dict())
                eval_agent.epsilon = 0.0
                eval_env = Environment(OBSTACLE_FILE)
                eval_env.instant_move = env.instant_move

                eval_state = eval_env.reset()
                eval_done = False
                eval_positions = [(eval_env.agent_x, eval_env.agent_y)]
                while not eval_done:
                    eval_action = eval_agent.select_action(eval_state)
                    _, _, eval_done, eval_info = eval_env.step(eval_action)
                    eval_state = eval_env.get_state()
                    eval_positions.append((eval_env.agent_x, eval_env.agent_y))

                if eval_info.get("success"):
                    best_path = LOAD_MODEL
                    agent.save_model(best_path)
                    print(f"💾 [SAVED BEST] Model saved as {best_path}")
        else:
            if episode % MODEL_SAVE_INTERVAL == 0:
                save_path = os.path.join("models", f"DQN-{episode}.pth")
                agent.save_model(save_path)
                print(f"💾 [SAVED] Model saved at Episode {episode} -> {save_path}")

        agent.decay_epsilon()

    if not INDEPENDENT_BASED:
        final_path = os.path.join("models", f"DQN-{training_episodes}.pth")
        agent.save_model(final_path)
        print(f"💾 [FINAL] Final model saved -> {final_path}")

    csv_file.close()

    model_path = LOAD_MODEL if os.path.exists(LOAD_MODEL) else None
    if model_path is None and os.path.exists("models"):
        model_candidates = [
            os.path.join("models", f)
            for f in os.listdir("models")
            if f.endswith(".pth")
        ]
        if model_candidates:
            model_path = sorted(model_candidates)[-1]

    if model_path is not None and os.path.exists(model_path):
        eval_agent = Agent(device=device)
        eval_agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        eval_agent.target_net.load_state_dict(eval_agent.policy_net.state_dict())
        eval_agent.epsilon = 0.0
        eval_env = Environment(OBSTACLE_FILE)
        eval_env.instant_move = True

        eval_state = eval_env.reset()
        eval_done = False
        eval_positions = [(eval_env.agent_x, eval_env.agent_y)]
        while not eval_done:
            eval_action = eval_agent.select_action(eval_state)
            _, _, eval_done, eval_info = eval_env.step(eval_action)
            eval_state = eval_env.get_state()
            eval_positions.append((eval_env.agent_x, eval_env.agent_y))

        save_path = os.path.join("plots", "best_model_path.png")
        draw_path_plot(eval_env, eval_positions, "Best Model Path", save_path)
        print(f"📊 Best model path plot saved to: {save_path}")
    else:
        print("⚠ No model file found; skipping best-model path plot")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total Time: {total_time:.2f}s ({total_time/training_episodes:.4f}s per episode)")
    print(
        f"Total Success: {success_count} | Total Collision: {collision_count} | Total Timeout: {timeout_count}"
    )
    print(f"Success Rate: {success_count/training_episodes*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_training_and_plot()
