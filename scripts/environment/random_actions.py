#!/usr/bin/env python3
import gymnasium as gym
import argparse
import ast
from src.environments import registered_versions
from src.environments.wrappers.partial_observability import Partial_Observability
from src.environments.wrappers.multi_channel import Multi_Channel
from src.environments.wrappers.random_drowning import Random_Drowning

"""
Visualizes an environment with random actions, printing the action, reward, and state at each step.
"""

def test_random(env, state_reset=None, random_init="no randomness"):
    """
    Runs an environment with random actions.
    """
    env.set_render_mode("human")
    env.metadata['render_fps'] = 4

    while True:
        if state_reset:
            state, _ = env.reset(state=state_reset)
        else:
            state, _ = env.reset(random_init=random_init)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            env.render()
            # randomly choose an action
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            print(f"Action: {action}, Reward: {reward}, State: \n {state}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run an environment with random actions for testing.")
    
    parser.add_argument('env_id', type=str, help='Gym environment ID')
    parser.add_argument('--random_init', type=str, default='no randomness', help='Random initialization type for reset (e.g., "positions", "no randomness").')
    parser.add_argument('--state_reset', type=ast.literal_eval, help='"Comma-separated values specifying the positions of the agent and each person on the flattened map, following the pattern: [agent_position, position_person_id_1, position_person_id_2, position_person_id_3, position_person_id_4]"')

    # Wrappers
    parser.add_argument('--multi_channel', action='store_true', help='Apply the Multi_Channel wrapper.')
    parser.add_argument('--random_drowning_prob', type=float, help='Apply Random_Drowning wrapper with given probability.')
    parser.add_argument('--partial_observability_window', type=int, help='Apply Partial_Observability wrapper with given window size.')

    args = parser.parse_args()

    env = gym.make(args.env_id)

    if args.multi_channel:
        env = Multi_Channel(env)

    if args.random_drowning_prob is not None:
        env = Random_Drowning(env, prob=args.random_drowning_prob)

    if args.partial_observability_window is not None:
        env = Partial_Observability(env, obs_window_size=args.partial_observability_window)

    test_random(env, state_reset=args.state_reset, random_init=args.random_init)