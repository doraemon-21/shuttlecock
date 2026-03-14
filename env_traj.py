
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
import time
import itertools # Still imported but not used for color cycling

# A class to represent the shuttlecock
class Shuttlecock:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.mass = 0.005  # ~5 grams
        self.drag_coefficient = 0.6
        self.area = 0.004  # ~4e-3 m^2
        self.air_density = 1.225  # kg/m^3 at sea level

# Our custom 3D Shuttlecock Environment
class ShuttlecockEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -50, -50, -50]),
            high=np.array([20, 20, 20, 50, 50, 50]),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(low=0.1, high=1.0, shape=(1,), dtype=np.float32)

        self.shuttlecock = None
        self.gravity = np.array([0, 0, -9.8])
        self.render_mode = render_mode
        self.current_time = 0.0
        self.dt = 0.01
        self.trajectory = []
        self.episode_step_count = 0
        self.shift_applied = False
        self.fast_forward_mode = False

        self.fig = None
        self.ax = None
        self.current_line = None
        self.shuttlecock_point = None
        self.text_info = None
        
        self.all_drawn_trajectories = [] 
        # MODIFIED: No longer using itertools.cycle for colors
        # self.colors = itertools.cycle(plt.cm.get_cmap('tab10').colors)
        self.current_trajectory_color = None # Will be set in reset

        self.button_ax = None
        self.button = None

    def _get_obs(self):
        return np.concatenate((self.shuttlecock.position, self.shuttlecock.velocity))

    def _get_info(self):
        return {"current_time": self.current_time}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # MODIFIED: Set current trajectory color to bright green
        self.current_trajectory_color = 'limegreen' # Bright green for normal trajectories

        initial_position = [0, self.np_random.uniform(0, 20), self.np_random.uniform(1, 2)]
        
        initial_velocity_x = self.np_random.uniform(20, 45)  
        initial_velocity_y = self.np_random.uniform(-10, 10) 
        initial_velocity_z = self.np_random.uniform(10, 25)

        initial_velocity = np.array([initial_velocity_x, initial_velocity_y, initial_velocity_z])
        
        self.shuttlecock = Shuttlecock(position=initial_position, velocity=initial_velocity)
        self.current_time = 0.0
        self.trajectory = [self.shuttlecock.position.copy()]
        self.episode_step_count = 0
        self.shift_applied = False

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        
        global current_episode_number
        if current_episode_number is not None and (current_episode_number % 5 == 0) and not self.shift_applied:
            if self.episode_step_count > 50:
                shift_vector = np.random.uniform(-1, 1, 3) 
                self.shuttlecock.velocity += shift_vector
                self.shift_applied = True
                print(f"Applying a random shift to trajectory {current_episode_number}!")
        
        self.episode_step_count += 1
        
        velocity_magnitude = np.linalg.norm(self.shuttlecock.velocity)
        drag_force = -0.5 * self.shuttlecock.air_density * velocity_magnitude**2 * \
                     self.shuttlecock.area * self.shuttlecock.drag_coefficient * \
                     (self.shuttlecock.velocity / (velocity_magnitude + 1e-6))
        
        net_force = self.shuttlecock.mass * self.gravity + drag_force
        
        acceleration = net_force / self.shuttlecock.mass
        self.shuttlecock.velocity += acceleration * self.dt
        self.shuttlecock.position += self.shuttlecock.velocity * self.dt
        self.current_time += self.dt
        
        self.trajectory.append(self.shuttlecock.position.copy())
        
        done = self.shuttlecock.position[2] <= 0
        reward = 0.0
        if done:
            reward = 1.0 

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, False, info

    def render(self):
        if self.render_mode == "human":
            if self.fig is None:
                plt.ion()
                self.fig = plt.figure(figsize=(10, 8))
                self.ax = self.fig.add_subplot(111, projection='3d')
                
                self.ax.set_xlim([0, 20])
                self.ax.set_ylim([0, 20])
                self.ax.set_zlim([0, 20])
                
                self.ax.set_xlabel('X Position (m)')
                self.ax.set_ylabel('Y Position (m)')
                self.ax.set_zlabel('Z Position (m)')
                self.ax.set_title('Shuttlecock Trajectories')
                
                x_floor = np.linspace(0, 20, 2)
                y_floor = np.linspace(0, 20, 2)
                X_floor, Y_floor = np.meshgrid(x_floor, y_floor)
                Z_floor = np.zeros_like(X_floor)
                self.ax.plot_surface(X_floor, Y_floor, Z_floor, color='gray', alpha=0.5, zorder=-1)
                
                self_color = 'red' if (current_episode_number % 5 == 0 and self.shift_applied) else self.current_trajectory_color
                self.current_line, = self.ax.plot([], [], [], '-', color=self_color, label='Current Trajectory')
                self.shuttlecock_point, = self.ax.plot([], [], [], 'o', color=self_color, markersize=8, label='Shuttlecock')
                
                self.text_info = self.fig.text(0.02, 0.95, '', transform=self.fig.transFigure)
                self.ax.legend()
                
                self.button_ax = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
                self.button = Button(self.button_ax, 'Fast-Forward')
                self.button.on_clicked(self.toggle_fast_forward)
            
            # Ensure the current line/point color updates if a shift is applied mid-trajectory
            current_line_color = 'red' if (current_episode_number % 5 == 0 and self.shift_applied) else self.current_trajectory_color
            self.current_line.set_color(current_line_color)
            self.shuttlecock_point.set_color(current_line_color)


            trajectory_array = np.array(self.trajectory)
            if trajectory_array.shape[0] > 0:
                self.current_line.set_data_3d(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2])
                self.shuttlecock_point.set_data_3d([self.shuttlecock.position[0]], [self.shuttlecock.position[1]], [self.shuttlecock.position[2]])
            
            info_text = f"Time: {self.current_time:.2f} s\n"
            info_text += f"Position: ({self.shuttlecock.position[0]:.2f}, {self.shuttlecock.position[1]:.2f}, {self.shuttlecock.position[2]:.2f})\n"
            info_text += f"Velocity: ({self.shuttlecock.velocity[0]:.2f}, {self.shuttlecock.velocity[1]:.2f}, {self.shuttlecock.velocity[2]:.2f})"
            self.text_info.set_text(info_text)
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            if not self.fast_forward_mode:
                time.sleep(0.01)

    def toggle_fast_forward(self, event):
        self.fast_forward_mode = not self.fast_forward_mode
        if self.fast_forward_mode:
            self.button.label.set_text('Normal Speed')
            print("Fast-forward mode ON.")
        else:
            self.button.label.set_text('Fast-Forward')
            print("Fast-forward mode OFF.")

    def add_completed_trajectory(self, trajectory_data):
        if self.ax is not None:
            trajectory_array = np.array(trajectory_data)
            
            global current_episode_number
            # MODIFIED: Use bright red for altered trajectories, bright green for normal
            if current_episode_number % 5 == 0:
                line, = self.ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], 
                                     '-', color='red', linewidth=2.5, alpha=0.9, label=f'Trajectory {len(self.all_drawn_trajectories) + 1} (Altered)')
            else:
                line, = self.ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], 
                                     '-', color='limegreen', linewidth=1.5, alpha=0.7, label=f'Trajectory {len(self.all_drawn_trajectories) + 1}')
            
            self.all_drawn_trajectories.append(line)
            
            # Rebuild the legend to show all labels
            handles, labels = self.ax.get_legend_handles_labels()
            self.ax.legend(handles, labels)

    def close(self):
        plt.close(self.fig)
        self.fig = None
        self.ax = None
        plt.ioff()

# Global variable to track episode number
current_episode_number = 0

# --- Main execution block for running the simulation and displaying output ---
if __name__ == "__main__":
    env = ShuttlecockEnv(render_mode="human")
    
    num_episodes = 50
    
    print(f"Starting simulation of {num_episodes} shuttlecock trajectories in a 20x20x20 grid.")
    print("Normal trajectories will be bright green. Every 5th trajectory will be altered and highlighted in bright red.")
    print("Click the 'Fast-Forward' button in the plot window to toggle the animation speed.")

    for episode in range(1, num_episodes + 1):
        current_episode_number = episode
        print(f"\n--- Episode {episode}/{num_episodes} ---")
        
        observation, info = env.reset()
        done = False
        
        env.render() # The initial render sets up the plot and button
        
        while not done:
            action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)
            
            env.render() # The render method now uses the internal fast_forward_mode flag
        
        env.add_completed_trajectory(env.trajectory)
        
        print(f"Episode {episode} completed.")
        print(f"Final Position: x={env.shuttlecock.position[0]:.2f}, y={env.shuttlecock.position[1]:.2f}, z={env.shuttlecock.position[2]:.2f}")
        print(f"Total Flight Time: {env.current_time:.2f} s")
        
    print("\nSimulation of all episodes completed.")
    
    print("All trajectories are now displayed. Close the plot window to exit the program.")
    plt.ioff()
    plt.show(block=True)
    
    env.close()