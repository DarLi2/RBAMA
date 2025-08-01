from src.RBAMA import guard_net
from src.RBAMA.guard_net import Bandit_Pushing
import gymnasium as gym
from src.environments.wrappers.multi_channel import Multi_Channel
from src.environments.wrappers.random_drowning import Random_Drowning
from src.environments.wrappers.partial_observability import Partial_Observability
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train bridge_guard agent.")
    parser.add_argument('env_id', type=str, help="Gym environment ID (e.g., bridge1_v1)")
    parser.add_argument('training_episodes', type=int, help="Number of training episodes")
    parser.add_argument('--use_random_drowning', action='store_true', help='Apply the Random_Drowning wrapper')
    parser.add_argument('--use_partial_observability', action='store_true', help='Apply the Partial_observability wrapper')
    

    args = parser.parse_args()
    env_id = args.env_id
    training_episodes = args.training_episodes

    env = gym.make(env_id)
    env.set_reward_type("waiting")
    env = Multi_Channel(env)
    agent = Bandit_Pushing(env)

    if args.use_random_drowning:
        env = Random_Drowning(env)
    if args.use_partial_observability:
        env = Partial_Observability(env)

    agent_name = f"{env_id}push{training_episodes}"
    agent.train(training_episodes, agent_name)

    # Save the trained waiting net
    guard_net.save_agent(agent, agent_name)


