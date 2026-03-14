import env
import numpy as np
import gymnasium as gym 
from gymnasium import spaces
class ShuttlecockEnv(gym.Env):
    def __init__(self):
        super(ShuttlecockEnv, self).__init__()
        
        # State = [x, y, z, vx, vy, vz]
        self.observation_space = spaces.Box(
            low=-100, high=100, shape=(6,), dtype=np.float32
        )
        
        # Action = predicted change in velocity (delta vx, vy, vz)
        self.action_space = spaces.Box(
            low=-10, high=10, shape=(3,), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)   # call parent reset to handle seeding
        
        self.state = np.array([0, 0, 10, 0, 0, -5], dtype=np.float32)  
        
        # return observation and info (empty dict if no extra info)
        return self.state, {}


    def step(self, action):
        x, y, z, vx, vy, vz = self.state

        # Apply action
        vx += action[0]
        vy += action[1]
        vz += action[2]

        # Gravity
        g = -9.81 * 0.05
        vz += g

        # Update position
        x += vx * 0.05
        y += vy * 0.05
        z += vz * 0.05

        self.state = np.array([x, y, z, vx, vy, vz], dtype=np.float32)

        # Reward: closer to physics = better
        target_vz = -9.81 * 0.05
        reward = -abs(vz - target_vz)

        terminated = z <= 0
        truncated = False  

        return self.state, reward, terminated, truncated, {}



    def render(self, mode="human"):
        print(f"Shuttlecock at {self.state[:3]}")
        

from stable_baselines3 import PPO

env = ShuttlecockEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Save model
model.save("ppo_shuttlecock")

