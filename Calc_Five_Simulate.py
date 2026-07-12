import os
import time
import numpy as np
import torch

from constant import *
from DQN import Environment, Agent
import DQN as dqn_module


def run_five_step_simulation(episodes=1000, max_steps_per_episode=5):
    print("=" * 70)
    print("FIVE-STEP DQN SIMULATION")
    print("=" * 70)
    print(f"Episodes       : {episodes}")
    print(f"Max steps/ep   : {max_steps_per_episode}")
    print(f"Obstacle file  : {OBSTACLE_FILE}")
    print("Mode           : Headless, reward follows standard DQN env")
    print("" * 70)

    dqn_module.MAX_STEPS = max_steps_per_episode

    env = Environment(OBSTACLE_FILE)
    env.instant_move = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(device=device)

    start_time = time.time()
    
    # Store initial Q-values for comparison
    first_episode_q_values = None
    last_episode_q_values = None
    first_episode_action = None
    last_episode_action = None
    
    # Store all Q-values history
    q_values_history = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        episode_steps = 0

        with torch.no_grad():
            initial_q_values = agent.policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
            initial_q_values = initial_q_values.cpu().numpy().flatten()
            
            # Store first episode Q-values
            if episode == 1:
                first_episode_q_values = initial_q_values.copy()
                first_episode_action = int(np.argmax(initial_q_values))

        while episode_steps < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            if info.get("step_completed", False):
                episode_steps += 1
                agent.memory.push(state, action, reward, next_state, done)
                state = next_state
                agent.update()

            if done:
                break

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        agent.decay_epsilon()
        
        # Store Q-values for this episode
        q_values_history.append({
            'episode': episode,
            'q_values': initial_q_values.copy(),
            'best_action': int(np.argmax(initial_q_values)),
            'reward': total_reward,
            'steps': episode_steps,
            'status': info.get('termination_reason', 'Unknown')
        })

        print(
            f"Ep {episode:3d}/{episodes} | Initial state Q-values = {np.round(initial_q_values.tolist(), 4)}"
        )
        print(
            f"    -> Reward: {total_reward:7.2f} | Steps: {episode_steps:2d} | "
            f"Status: {info.get('termination_reason', 'Unknown')} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    elapsed = time.time() - start_time
    
    # Get final episode Q-values
    if q_values_history:
        last_episode_q_values = q_values_history[-1]['q_values']
        last_episode_action = q_values_history[-1]['best_action']
    
    # ==================== PRINT SUMMARY ====================
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    print(f"Simulation finished in {elapsed:.2f}s")
    print(f"Total episodes: {episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print("-" * 70)
    
    # Q-value comparison
    print("\nQ-VALUE DEVELOPMENT (Initial State):")
    print("-" * 70)
    print(f"{'':<20} {'Q0 (Forward)':<15} {'Q1 (Turn Left)':<15} {'Q2 (Turn Right)':<15} {'Best Action':<15}")
    print("-" * 70)
    
    action_names = ["Forward", "Turn Left", "Turn Right"]
    
    if first_episode_q_values is not None:
        print(f"{'Episode 1':<20} {first_episode_q_values[0]:<15.4f} {first_episode_q_values[1]:<15.4f} {first_episode_q_values[2]:<15.4f} {action_names[first_episode_action]:<15}")
    
    if last_episode_q_values is not None:
        print(f"{f'Episode {episodes}':<20} {last_episode_q_values[0]:<15.4f} {last_episode_q_values[1]:<15.4f} {last_episode_q_values[2]:<15.4f} {action_names[last_episode_action]:<15}")
    
    print("-" * 70)
    
    # Calculate and show gap
    if first_episode_q_values is not None and last_episode_q_values is not None:
        gap = last_episode_q_values - first_episode_q_values
        print("\nQ-VALUE CHANGE (Gap):")
        print("-" * 70)
        print(f"{'Action':<20} {'Initial (Ep1)':<15} {'Final (Ep' + str(episodes) + ')':<15} {'Change (Gap)':<15}")
        print("-" * 70)
        
        for i, action_name in enumerate(action_names):
            print(f"{action_name:<20} {first_episode_q_values[i]:<15.4f} {last_episode_q_values[i]:<15.4f} {gap[i]:<15.4f}")
        
        print("-" * 70)
        
        # Best action comparison
        print("\nBEST ACTION COMPARISON:")
        print("-" * 70)
        print(f"Episode 1 Best Action  : {first_episode_action} ({action_names[first_episode_action]})")
        print(f"Episode {episodes} Best Action  : {last_episode_action} ({action_names[last_episode_action]})")
        
        if first_episode_action == last_episode_action:
            print(f"✓ Best action remained the same: {action_names[first_episode_action]}")
        else:
            print(f"↻ Best action changed from {action_names[first_episode_action]} to {action_names[last_episode_action]}")
        
        # Q-value summary
        print("\nQ-VALUE SUMMARY:")
        print("-" * 70)
        print(f"Initial Q-values (Ep1)   : [{first_episode_q_values[0]:.4f}, {first_episode_q_values[1]:.4f}, {first_episode_q_values[2]:.4f}]")
        print(f"Final Q-values (Ep{episodes})  : [{last_episode_q_values[0]:.4f}, {last_episode_q_values[1]:.4f}, {last_episode_q_values[2]:.4f}]")
        print(f"Q-value Gap              : [{gap[0]:.4f}, {gap[1]:.4f}, {gap[2]:.4f}]")
        print(f"Max Q-value increase     : {max(gap):.4f} (Action {np.argmax(gap)}: {action_names[np.argmax(gap)]})")
        print(f"Min Q-value increase     : {min(gap):.4f} (Action {np.argmin(gap)}: {action_names[np.argmin(gap)]})")
        print("-" * 70)
    
    # Statistics
    print("\nTRAINING STATISTICS:")
    print("-" * 70)
    rewards = [h['reward'] for h in q_values_history]
    steps = [h['steps'] for h in q_values_history]
    success_count = sum(1 for h in q_values_history if h['status'] == 'Success')
    collision_count = sum(1 for h in q_values_history if h['status'] == 'Collision')
    timeout_count = sum(1 for h in q_values_history if h['status'] == 'Timeout')
    
    print(f"Total episodes       : {episodes}")
    print(f"Success count        : {success_count}")
    print(f"Collision count      : {collision_count}")
    print(f"Timeout count        : {timeout_count}")
    print(f"Success rate         : {success_count/episodes*100:.1f}%")
    print(f"Average reward       : {sum(rewards)/len(rewards):.4f}")
    print(f"Min reward           : {min(rewards):.4f}")
    print(f"Max reward           : {max(rewards):.4f}")
    print(f"Average steps        : {sum(steps)/len(steps):.2f}")
    print(f"Final epsilon        : {agent.epsilon:.4f}")
    print("-" * 70)
    
    # Q-value evolution trend
    print("\nQ-VALUE EVOLUTION TREND:")
    print("-" * 70)
    
    # Show Q-values at specific intervals
    intervals = [1, episodes//4, episodes//2, episodes*3//4, episodes]
    print(f"{'Episode':<10} {'Q0 (Forward)':<15} {'Q1 (Turn Left)':<15} {'Q2 (Turn Right)':<15} {'Best Action':<15}")
    print("-" * 70)
    
    for ep in intervals:
        if ep <= episodes:
            # Find closest episode
            for record in q_values_history:
                if record['episode'] == ep:
                    q_vals = record['q_values']
                    best_act = record['best_action']
                    print(f"{ep:<10} {q_vals[0]:<15.4f} {q_vals[1]:<15.4f} {q_vals[2]:<15.4f} {action_names[best_act]:<15}")
                    break
            else:
                # If exact episode not found, find nearest
                for record in q_values_history:
                    if record['episode'] >= ep:
                        q_vals = record['q_values']
                        best_act = record['best_action']
                        print(f"{ep}~{record['episode']:<6} {q_vals[0]:<15.4f} {q_vals[1]:<15.4f} {q_vals[2]:<15.4f} {action_names[best_act]:<15}")
                        break
    
    print("-" * 70)
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_five_step_simulation()