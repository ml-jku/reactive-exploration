import sys
import argparse
from src import custom_environments
sys.modules['jbw.environments'] = custom_environments
import gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="JBW-render-verbose-v1")
    args = parser.parse_args()
    env = gym.make(args.env_name)
    obs = env.reset()
    print(obs)
    for i in range(10000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render(mode='matplotlib')
