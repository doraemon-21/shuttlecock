import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
import time
import itertools

# A class to represent the shuttlecock
class Shuttlecock:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.mass = 0.005  # ~5 grams
        self.drag_coefficient = 0.6
        self.area = 0.004  # ~4e-3 m^2
        self.air_density = 1.225  # kg/m^3 at sea level
        # NEW: Spin properties
        self.angular_velocity = np.array([0.0, 0.0, 0.0]) # radians per second

# Our custom 3D Shuttlecock Environment
class ShuttlecockEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -50, -50, -50]),
            high=np.array([20, 20, 20, 50, 50, 50]),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([20, 20, 5]), # Assuming max fall time is 5 seconds
            dtype=np.float32,
        )

        self.shuttlecock = None
        self.gravity = np.array([0, 0, -9.8])
        self.render_mode = render_mode
        self.current_time = 0.0
        self.dt = 0.01
        self.trajectory = []
        self.episode_step_count = 0
        self.shift_applied = False
        self.ff_speed = 1.0

        self.fig = None
        self.ax = None
        self.current_line = None
        self.shuttlecock_point = None
        self.text_info = None
        
        self.all_drawn_trajectories = [] 
        self.colors = itertools.cycle(plt.cm.get_cmap('tab10').colors)
        self.current_trajectory_color = None
        
        self.buttons = {}
        self.final_pos = None
        self.final_time = None

    def _get_obs(self):
        return np.concatenate((self.shuttlecock.position, self.shuttlecock.velocity))

    def _get_info(self):
        return {"current_time": self.current_time}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_trajectory_color = next(self.colors)
        
        initial_position = [0, self.np_random.uniform(0, 20), self.np_random.uniform(1, 2)]
        
        # MODIFIED: Reduced initial velocities to keep trajectories within the grid
        initial_velocity_x = self.np_random.uniform(10, 25)
        initial_velocity_y = self.np_random.uniform(-5, 5)
        initial_velocity_z = self.np_random.uniform(5, 15)

        initial_velocity = np.array([initial_velocity_x, initial_velocity_y, initial_velocity_z])
        
        self.shuttlecock = Shuttlecock(position=initial_position, velocity=initial_velocity)
        self.current_time = 0.0
        self.trajectory = [self.shuttlecock.position.copy()]
        self.episode_step_count = 0
        self.shift_applied = False
        
        self.final_pos = None
        self.final_time = None
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        
        temp_shuttlecock = self.shuttlecock
        temp_time = self.current_time
        temp_step_count = self.episode_step_count
        temp_shift_applied = self.shift_applied

        terminated = False
        truncated = False
        while not terminated and not truncated:
            velocity_magnitude = np.linalg.norm(temp_shuttlecock.velocity)
            drag_force = -0.5 * temp_shuttlecock.air_density * velocity_magnitude**2 * \
                         temp_shuttlecock.area * temp_shuttlecock.drag_coefficient * \
                         (temp_shuttlecock.velocity / (velocity_magnitude + 1e-6))
            
            net_force = temp_shuttlecock.mass * self.gravity + drag_force
            
            acceleration = net_force / temp_shuttlecock.mass
            temp_shuttlecock.velocity += acceleration * self.dt
            temp_shuttlecock.position += temp_shuttlecock.velocity * self.dt
            temp_time += self.dt
            
            temp_step_count += 1
            global current_episode_number
            if current_episode_number is not None and (current_episode_number % 5 == 0) and not temp_shift_applied:
                if temp_step_count > 50:
                    shift_vector = np.random.uniform(-1, 1, 3) 
                    temp_shuttlecock.velocity += shift_vector
                    temp_shift_applied = True
            
            # The episode ends if the shuttlecock leaves the 20x20x20 grid
            terminated = temp_shuttlecock.position[2] <= 0 or \
                         temp_shuttlecock.position[0] < 0 or temp_shuttlecock.position[0] > 20 or \
                         temp_shuttlecock.position[1] < 0 or temp_shuttlecock.position[1] > 20
        
        self.final_pos = temp_shuttlecock.position[:2]
        self.final_time = temp_time

        predicted_pos = action[:2]
        predicted_time = action[2]
        
        pos_error = np.linalg.norm(self.final_pos - predicted_pos)
        time_error = abs(self.final_time - predicted_time)

        reward = -(pos_error + time_error)

        observation = self._get_obs() 
        info = {"fall_point": self.final_pos, "fall_time": self.final_time}

        return observation, reward, terminated, truncated, info

    def render(self):
        print(f"Rendering not supported in this training script. Final position: {self.final_pos}, time: {self.final_time}")
        pass
    
    def toggle_fast_forward(self, event):
        pass

    def add_completed_trajectory(self, trajectory_data):
        pass

    def close(self):
        pass

# Global variable to track episode number
current_episode_number = 0