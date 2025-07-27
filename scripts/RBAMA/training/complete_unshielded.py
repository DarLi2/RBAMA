#!/usr/bin/env python3
from src.environments import registered_versions
import gymnasium as gym
from src.environments.wrappers.multi_channel import Multi_Channel
from src.environments.wrappers.random_drowning import Random_Drowning
from src.environments.wrappers.partial_observability import Partial_Observability
from src.RBAMA import moral_judge
from src.RBAMA import translator
from scripts.RBAMA.training.modular_training import train_guard, train_rescuing, train_instrumental_policy, train_reasoning
from scripts.utils import create_judge
import argparse
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s] %(message)s', datefmt='%H:%M:%S')
    
    parser = argparse.ArgumentParser(description="Train an RBAMA agent in a selected environment with optional wrappers.")

    parser.add_argument('env_id', type=str, help='Gym environment ID')
    parser.add_argument('training_episodes_guard', type=int)
    parser.add_argument('training_episodes_resc', type=int)
    parser.add_argument('training_episodes_instr', type=int)
    parser.add_argument('training_episodes_reas', type=int)
    parser.add_argument('judge', type=str, default='prioR', choices=['prioR', 'prioW', 'onlyR', 'onlyW'],
                        help='Type of moral judge to use (default: prioR)')
    parser.add_argument('--use_CNN', action='store_true', help='Use a CNN')
    parser.add_argument('--use_random_drowning', action='store_true', help='Apply the Random_Drowning wrapper')
    parser.add_argument('--use_partial_observability', action='store_true', help='Apply the Partial_observability wrapper')

    args = parser.parse_args()

    env_id = args.env_id
    env = gym.make(env_id)
    env = Multi_Channel(env)

    if args.use_random_drowning:
        env = Random_Drowning(env)
    if args.use_partial_observability:
        env = Partial_Observability(env)

    judge = create_judge(args.judge)

    guard_net_name =  env_id + "wait" + str(args.training_episodes_guard)
    rescuing_net_name =  env_id + "resc" + str(args.training_episodes_resc)
    instrumental_net_name = env_id +"instr" + str(args.training_episodes_instr)

    train_instrumental_policy(env, args.training_episodes_instr, instrumental_net_name)
    logging.info("Instrumental policy trained: %s", instrumental_net_name)
    train_guard(env, args.training_episodes_guard, guard_net_name)
    logging.info("Guard network trained: %s", guard_net_name)
    train_rescuing(env, args.training_episodes_resc, rescuing_net_name)
    logging.info("Rescuing network trained: %s", rescuing_net_name)
    train_reasoning(env,env_id, judge, rescuing_net_name, guard_net_name, instrumental_net_name, args.training_episodes_resc, args.training_episodes_guard, args.training_episodes_instr, args.training_episodes_reas)
    logging.info("Reasoning trained")