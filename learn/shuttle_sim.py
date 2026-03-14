import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import itertools
from matplotlib.widgets import Button

# Import the custom environment class
from shuttlecock_env import ShuttlecockEnv

# Register the custom environment with Gymnasium
gym.register(
    id='ShuttlecockEnv-v0',
    entry_point='shuttlecock_env:ShuttlecockEnv',
)

def run_training_and_visualization():
    # --- Phase 1: Training ---
    print("Creating environment for training...")
    env = make_vec_env('ShuttlecockEnv-v0', n_envs=1)
    
    print("Initializing PPO agent...")
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=3e-4, 
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1, 
        tensorboard_log="./ppo_shuttlecock_tensorboard/"
    )
    
    total_timesteps = 250000 
    print(f"Training started for {total_timesteps} timesteps...")
    try:
        model.learn(total_timesteps=total_timesteps)
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving model...")
    print("Training finished.")

    model.save("ppo_shuttlecock_predictor_tuned")
    print("Model saved as ppo_shuttlecock_predictor_tuned.zip")

    # --- Phase 2: Prediction and Visualization ---
    print("\nStarting prediction and visualization...")
    # Load the trained model
    model = PPO.load("ppo_shuttlecock_predictor_tuned")
    
    # Create a separate environment instance for visualization
    vis_env = ShuttlecockEnv(render_mode="human")
    
    num_tests = 50
    total_pos_error = 0
    total_time_error = 0
    
    # Initialize the plot outside the loop for persistence
    plt.ion()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    predicted_text = ax.text2D(0.05, 0.9, '', transform=ax.transAxes)
    
    for i in range(num_tests):
        print(f"\n--- Visualizing Test {i+1}/{num_tests} ---")
        
        # Reset the environment to get a new trajectory
        obs, info = vis_env.reset()
        
        # Get the agent's prediction from the initial observation
        predicted_values, _ = model.predict(obs, deterministic=True)
        predicted_pos = predicted_values[:2]
        predicted_time = predicted_values[2]
        
        # --- Visualization Setup for each new test ---
        ax.clear()
        ax.set_xlim([0, 20])
        ax.set_ylim([0, 20])
        ax.set_zlim([0, 20])
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title(f"Agent Prediction vs. Actual Trajectory (Test {i+1})")
        
        x_floor = np.linspace(0, 20, 2)
        y_floor = np.linspace(0, 20, 2)
        X_floor, Y_floor = np.meshgrid(x_floor, y_floor)
        Z_floor = np.zeros_like(X_floor)
        ax.plot_surface(X_floor, Y_floor, Z_floor, color='gray', alpha=0.5, zorder=-1)
        
        # Plot the predicted landing point before the animation starts
        ax.scatter([predicted_pos[0]], [predicted_pos[1]], [0], color='red', marker='o', s=100, label='Predicted Fall Point')
        
        # The actual simulation loop
        trajectory_points = []
        done = False
        while not done:
            action = vis_env.action_space.sample()
            observation, reward, terminated, truncated, info = vis_env.step(action)
            done = terminated or truncated
            trajectory_points.append(observation[:3].copy())
            
            if len(trajectory_points) > 1:
                ax.lines.clear()
                
                traj_array = np.array(trajectory_points)
                ax.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], 'g-', label='Actual Trajectory')
                
                ax.scatter(traj_array[-1, 0], traj_array[-1, 1], traj_array[-1, 2], color='blue', marker='o', s=50, label='Shuttlecock')
                
                ax.scatter([predicted_pos[0]], [predicted_pos[1]], [0], color='red', marker='o', s=100)
                
                ax.legend()
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.01)
        
        actual_pos = vis_env.shuttlecock.position[:2]
        actual_time = vis_env.current_time
        
        pos_error = np.linalg.norm(actual_pos - predicted_pos)
        time_error = abs(actual_time - predicted_time)
        
        total_pos_error += pos_error
        total_time_error += time_error
        
        # Plot a final point for the actual landing position
        ax.scatter([actual_pos[0]], [actual_pos[1]], [0], color='blue', marker='x', s=100,  label='Actual Fall Point')
        ax.legend()
        
        info_text = (
            f"Predicted Point: ({predicted_pos[0]:.2f}, {predicted_pos[1]:.2f})\n"
            f"Actual Point:    ({actual_pos[0]:.2f}, {actual_pos[1]:.2f})\n"
            f"Position Error:  {pos_error:.2f} m\n"
            f"Predicted Time:  {predicted_time:.2f} s\n"
            f"Actual Time:     {actual_time:.2f} s\n"
            f"Time Error:      {time_error:.2f} s"
        )
        predicted_text.set_text(info_text)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        print(f"Final Position: x={actual_pos[0]:.2f}, y={actual_pos[1]:.2f}, Final Time: {actual_time:.2f} s")
        print("Done. Press Enter to visualize the next trajectory.")
        input()
        
    avg_pos_error = total_pos_error / num_tests
    avg_time_error = total_time_error / num_tests

    print("\n--- Evaluation Summary ---")
    print(f"Average Position Error: {avg_pos_error:.2f} meters")
    print(f"Average Time Error:     {avg_time_error:.2f} seconds")

    vis_env.close()
    plt.ioff()
    plt.close(fig)

if __name__ == "__main__":
    run_training_and_visualization()