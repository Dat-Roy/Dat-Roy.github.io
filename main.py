import gym_cutting_stock
import gymnasium as gym
from student_submissions.s2310xxx.policy23_first_fit import Policy_first_fit

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

if __name__ == "__main__":
    # Reset the environment
    observation, info = env.reset(seed=42)
    # Test
    rd_policy = Policy_first_fit()
    ep = 0
    while ep < NUM_EPISODES:
        action = rd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset(seed=ep)
            print(info)
            ep += 1
env.close()

