#!/usr/bin/env python3
from src.RBAMA import RBAMA
from src.RBAMA import rescuing_net
from src.RBAMA import waiting_net

def train_rescuing(env, episodes, name):
    env.set_reward_type("rescuing")
    rescuing_subnet = rescuing_net.RescuingCNN(env)
    rescuing_subnet.train(episodes, name)
    rescuing_net.save_agent(rescuing_subnet, name)

def train_guard(env, episodes, name, pb=True):
    env.set_reward_type("waiting")
    waiting_subnet = waiting_net.Bandit_Pushing(env)
    waiting_subnet.train(episodes, name, pb=pb)
    waiting_net.save_agent(waiting_subnet, name)

def train_instrumental_policy(env, episodes, name, pb=True):
    env.set_reward_type("instrumental")
    instrumental_net = RBAMA.RBAMA_CNN(env)
    instrumental_net.train(episodes, name, env, random_init = "positions", pb=pb)
    RBAMA.save_agent(instrumental_net, name)

def train_policy_shielded(env, waiting_net, episodes, name, pb=True):
    env.set_reward_type("instrumental")
    instrumental_net = RBAMA.RBAMA_CNN(env)
    instrumental_net.reasoning_unit.waiting_net = waiting_net
    instrumental_net.train_shielded(episodes, name, "C", env, random_init = "positions", pb=pb)
    RBAMA.save_agent(instrumental_net, name)

def train_reasoning(env, env_id, judge, rescuing_net_name, waiting_net_name, instrumental_net_name, training_episodes_rescuing, training_episodes_waiting, training_episodes_instrumental_policy, training_episodes_reasoning):
    rescuing_subnet, _ = rescuing_net.setup_rescuing_agent(rescuing_net_name)
    waiting_subnet, _ = waiting_net.setup_on_bridge(waiting_net_name)
    agent, _ = RBAMA.setup_reasoning_agent(instrumental_net_name)
    agent.reasoning_unit.rescuing_net = rescuing_subnet
    agent.reasoning_unit.waiting_net = waiting_subnet
    agent.train_reasons(training_episodes_reasoning, env, judge=judge)
    agent_name =  env_id +"modular"+ "R" + str(training_episodes_rescuing) + "W" + str(training_episodes_waiting) + "I" + str(training_episodes_instrumental_policy) + "R" + str(training_episodes_reasoning)
    RBAMA.save_agent(agent, agent_name)
