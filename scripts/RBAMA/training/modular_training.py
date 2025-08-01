#!/usr/bin/env python3
from src.RBAMA import RBAMA
from src.RBAMA import rescuing_net
from src.RBAMA import guard_net

def train_rescuing(env, episodes, name):
    env.set_reward_type("rescuing")
    rescuing_subnet = rescuing_net.RescuingCNN(env)
    rescuing_subnet.train(episodes, name)
    rescuing_net.save_agent(rescuing_subnet, name)

def train_guard(env, episodes, name, pb=True):
    env.set_reward_type("waiting")
    waiting_subnet = guard_net.Bandit_Pushing(env)
    waiting_subnet.train(episodes, name, pb=pb)
    guard_net.save_agent(waiting_subnet, name)

def train_instrumental_policy(env, episodes, name, pb=True):
    env.set_reward_type("instrumental")
    instrumental_net = RBAMA.RBAMA_CNN(env)
    instrumental_net.train(episodes, name, env, random_init = "positions", pb=pb)
    RBAMA.save_agent(instrumental_net, name)

def train_policy_shielded(env, guard_net, episodes, name, pb=True):
    env.set_reward_type("instrumental")
    instrumental_net = RBAMA.RBAMA_CNN(env)
    instrumental_net.reasoning_unit.guard_net = guard_net
    instrumental_net.train_shielded(episodes, name, "C", env, random_init = "positions", pb=pb)
    RBAMA.save_agent(instrumental_net, name)

def train_conformance(env, env_id, judge, rescuing_net_name, guard_net_name, instrumental_net_name, training_episodes_rescuing, training_episodes_waiting, training_episodes_instrumental_policy, training_episodes_reasoning):
    rescuing_subnet, _ = rescuing_net.setup_rescuing_agent(rescuing_net_name)
    guard_subnet, _ = guard_net.setup_on_bridge(guard_net_name)
    agent, _ = RBAMA.setup_reasoning_agent(instrumental_net_name)
    
    # Replace the reasoning unit networks with the trained ones
    agent.reasoning_unit.rescuing_net = rescuing_subnet
    agent.reasoning_unit.guard_net = guard_subnet
    
    agent.train_reasons(training_episodes_reasoning, env, judge=judge)
    agent_name =  env_id +"modular"+ "R" + str(training_episodes_rescuing) + "W" + str(training_episodes_waiting) + "I" + str(training_episodes_instrumental_policy) + "R" + str(training_episodes_reasoning)
    RBAMA.save_agent(agent, agent_name)
