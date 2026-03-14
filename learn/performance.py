import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import matplotlib.pyplot as plt
import os
from shuttlecock_env import ShuttlecockEnv

# Register the custom environment with Gymnasium
gym.register(
    id='ShuttlecockEnv-v0',
    entry_point='shuttlecock_env:ShuttlecockEnv',
)

def plot_performance(model_path="ppo_shuttlecock_predictor_tuned"):
    """
    Loads a trained PPO model, evaluates its performance, and generates plots.
    """
    print("Loading trained model...")
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}.zip' not found. Please train the agent first.")
        return

    env = make_vec_env('ShuttlecockEnv-v0', n_envs=1)
    
    num_tests = 50
    all_rewards = []
    all_predicted_pos = []
    all_actual_pos = []
    all_predicted_time = []
    all_actual_time = []
    
    print(f"\nRunning {num_tests} test episodes to collect data for plotting...")

    for i in range(num_tests):
        # The reset for a vectorized environment returns obs and info as lists/tuples
        obs = env.reset()
        
        # model.predict returns (predictions, states), and since n_envs=1,
        # predictions is a single array.
        predicted_values, _ = model.predict(obs, deterministic=True)
        
        predicted_pos = predicted_values[0][:2]
        predicted_time = predicted_values[0][2]
        
        # MODIFIED: Correctly unpack the 4 values returned by make_vec_env's step()
        _, reward, _, info = env.step(predicted_values)

        # The info returned by make_vec_env is a list of dictionaries (one per env)
        env_info = info[0]
        actual_pos = env_info.get('fall_point')
        actual_time = env_info.get('fall_time')
        
        # Store data for plotting
        all_rewards.append(reward[0])
        all_predicted_pos.append(predicted_pos)
        all_actual_pos.append(actual_pos)
        all_predicted_time.append(predicted_time)
        all_actual_time.append(actual_time)

    # --- Plot 1: Reward Distribution Histogram ---
    plt.figure(figsize=(10, 6))
    plt.hist(all_rewards, bins=20, edgecolor='black', color='skyblue')
    plt.title('Reward Distribution Across Test Episodes')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig("reward_distribution.png")
    print("Saved 'reward_distribution.png'")
    plt.show()

    # --- Plot 2: Predicted vs. Actual Values Scatter Plot ---
    all_predicted_pos = np.array(all_predicted_pos)
    all_actual_pos = np.array(all_actual_pos)
    all_predicted_time = np.array(all_predicted_time)
    all_actual_time = np.array(all_actual_time)

    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Agent Prediction Accuracy vs. Actual Values')

    # Scatter plot for Position (X and Y coordinates)
    ax1.scatter(all_actual_pos[:, 0], all_predicted_pos[:, 0], color='blue', label='X Coordinate')
    ax1.scatter(all_actual_pos[:, 1], all_predicted_pos[:, 1], color='orange', label='Y Coordinate')
    ax1.plot([0, 20], [0, 20], 'r--', label='Perfect Prediction')
    ax1.set_title('Predicted vs. Actual Fall Position')
    ax1.set_xlabel('Actual Position (m)')
    ax1.set_ylabel('Predicted Position (m)')
    ax1.set_xlim([0, 20])
    ax1.set_ylim([0, 20])
    ax1.legend()
    ax1.grid(True)

    # Scatter plot for Time
    ax2.scatter(all_actual_time, all_predicted_time, color='green')
    ax2.plot([0, max(all_actual_time)], [0, max(all_actual_time)], 'r--', label='Perfect Prediction')
    ax2.set_title('Predicted vs. Actual Fall Time')
    ax2.set_xlabel('Actual Time (s)')
    ax2.set_ylabel('Predicted Time (s)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("prediction_accuracy.png")
    print("Saved 'prediction_accuracy.png'")
    plt.show()

    print("\nPlotting complete. Check the directory for saved images.")

if __name__ == "__main__":
    plot_performance()