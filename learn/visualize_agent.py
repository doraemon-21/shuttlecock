import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
import time
from shuttlecock_env import ShuttlecockEnv 
import itertools

# Register the custom environment with Gymnasium
gym.register(
    id='ShuttlecockEnv-v0',
    entry_point='shuttlecock_env:ShuttlecockEnv',
)

def visualize_predictions(model_path="ppo_shuttlecock_predictor"):
    """
    Loads a trained PPO model and visualizes its predictions on a new environment instance.
    """
    print("Loading trained model...")
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}.zip' not found. Please train the agent first.")
        return

    # Create a new environment instance for visualization
    # We will use the 'human' render mode
    vis_env = ShuttlecockEnv(render_mode="human")

    # Define a custom color for the predicted point
    prediction_color = 'magenta'

    # Set up the plot for visualization
    plt.ion()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 20])
    ax.set_zlim([0, 20])
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Agent Prediction vs. Actual Trajectory')

    # Plot the floor
    x_floor = np.linspace(0, 20, 2)
    y_floor = np.linspace(0, 20, 2)
    X_floor, Y_floor = np.meshgrid(x_floor, y_floor)
    Z_floor = np.zeros_like(X_floor)
    ax.plot_surface(X_floor, Y_floor, Z_floor, color='gray', alpha=0.5, zorder=-1)

    # Place a single point for the predicted landing spot
    predicted_point, = ax.plot([], [], [], 'o', color=prediction_color, markersize=10, label='Predicted Point')
    predicted_text = ax.text2D(0.05, 0.9, '', transform=ax.transAxes)

    ax.legend()
    plt.show(block=False)

    num_visualizations = 5
    for i in range(num_visualizations):
        print(f"\nVisualizing Test {i+1}/{num_visualizations}")

        # Reset the environment to get a new trajectory
        obs, info = vis_env.reset()
        
        # Get the agent's prediction from the initial observation
        # MODIFIED: Removed the [0] index
        predicted_values, _ = model.predict(obs, deterministic=True)
        predicted_pos = predicted_values[:2]
        predicted_time = predicted_values[2]

        # The actual simulation loop
        done = False
        trajectory_points = []
        actual_time = 0  # Initialize to capture from info dict
        while not done:
            # The environment's step logic is now used for visualization
            _, _, terminated, truncated, info = vis_env.step(vis_env.action_space.sample())
            done = terminated or truncated
            trajectory_points.append(vis_env.shuttlecock.position.copy())
            
            # Capture the actual time from info dict when episode ends
            if done:
                actual_time = info.get('fall_time', 0)
            
            # Draw the trajectory in real-time
            trajectory_array = np.array(trajectory_points)
            if trajectory_array.shape[0] > 0:
                ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], 'g-')
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.01) # Small delay for animation

        # Plot the predicted landing point
        predicted_point.set_data_3d([predicted_pos[0]], [predicted_pos[1]], [0]) # z=0 for the floor
        
        # Get the actual final position and time from the environment
        actual_pos = vis_env.shuttlecock.position[:2]

        pos_error = np.linalg.norm(actual_pos - predicted_pos)
        time_error = abs(actual_time - predicted_time)

        # Update the info text
        info_text = (
            f"Predicted Point: ({predicted_pos[0]:.2f}, {predicted_pos[1]:.2f})\n"
            f"Actual Point:    ({actual_pos[0]:.2f}, {actual_pos[1]:.2f})\n"
            f"Position Error:  {pos_error:.2f} m\n"
            f"Predicted Time:  {predicted_time:.2f} s\n"
            f"Actual Time:     {actual_time:.2f} s\n"
            f"Time Error:      {time_error:.2f} s"
        )
        predicted_text.set_text(info_text)

        print("Done. Press Enter to visualize the next trajectory.")
        input() # Wait for user input before next visualization

    vis_env.close()
    plt.ioff()
    plt.close(fig)

if __name__ == "__main__":
    visualize_predictions()