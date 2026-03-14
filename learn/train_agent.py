import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

# Import the custom environment class
from shuttlecock_env import ShuttlecockEnv

# Register the custom environment with Gymnasium
gym.register(
    id='ShuttlecockEnv-v0',
    entry_point='shuttlecock_env:ShuttlecockEnv',
)

def run_training():
    print("Creating environment...")
    # Using make_vec_env is a good practice for Stable Baselines3, even with n_envs=1
    env = make_vec_env('ShuttlecockEnv-v0', n_envs=1)
    
    print("Initializing PPO agent...")
    # The PPO model learns to output the continuous values for the prediction
    # MODIFIED: Added hyperparameter tuning for better performance
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
    
    # Increased total_timesteps for proper training
    total_timesteps = 250000 
    print(f"Training started for {total_timesteps} timesteps...")
    try:
        model.learn(total_timesteps=total_timesteps)
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving model...")
    print("Training finished.")

    model.save("ppo_shuttlecock_predictor_tuned")
    print("Model saved as ppo_shuttlecock_predictor_tuned.zip")

    # --- Prediction after training ---
    print("\nStarting prediction and evaluation...")
    model = PPO.load("ppo_shuttlecock_predictor_tuned")
    
    num_tests = 50
    total_pos_error = 0
    total_time_error = 0
    
    for i in range(num_tests):
        obs = env.reset()
        predicted_values, _ = model.predict(obs, deterministic=True)
        
        predicted_pos = predicted_values[0][:2]
        predicted_time = predicted_values[0][2]
        
        step_result = env.step(predicted_values)

        if len(step_result) == 5:
            _, _, terminated, truncated, info = step_result
        else:
            _, _, done, info = step_result

        if isinstance(info, (list, tuple)):
            env_info = info[0]
        else:
            env_info = info

        actual_pos = env_info.get('fall_point')
        actual_time = env_info.get('fall_time')
        
        pos_error = np.linalg.norm(actual_pos - predicted_pos)
        time_error = abs(actual_time - predicted_time)

        total_pos_error += pos_error
        total_time_error += time_error
        
        print(f"\nTest {i+1}:")
        print(f"  Predicted Fall Point: ({predicted_pos[0]:.2f}, {predicted_pos[1]:.2f})")
        print(f"  Actual Fall Point:    ({actual_pos[0]:.2f}, {actual_pos[1]:.2f})")
        print(f"  Position Error:       {pos_error:.2f} meters")
        print(f"  Predicted Fall Time:  {predicted_time:.2f} s")
        print(f"  Actual Fall Time:     {actual_time:.2f} s")
        print(f"  Time Error:           {time_error:.2f} s")
    
    avg_pos_error = total_pos_error / num_tests
    avg_time_error = total_time_error / num_tests

    print("\n--- Evaluation Summary ---")
    print(f"Average Position Error: {avg_pos_error:.2f} meters")
    print(f"Average Time Error:     {avg_time_error:.2f} seconds")

if __name__ == "__main__":
    run_training()