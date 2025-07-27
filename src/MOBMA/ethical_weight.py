import src.MOBMA.CHVI as CHVI
import numpy as np
import gymnasium as gym
from src.MOBMA import ethical_momdp_wrapper
from itertools import product
from src.environments import registered_versions
from src.environments.wrappers.random_drowning import Random_Drowning
import time
import signal
import logging

logger = logging.getLogger(__name__)

"""
original code of R.S. adapted to calculate the ethical weight for instances of the bridge environmnet
"""

def ethical_embedding_state(hull):
    """
    Ethical embedding operation for a single state. Considers the points in the hull of a given state and returns
    the ethical weight that guarantees optimality for the ethical point of the hull

    :param hull: set of 2-D points, coded as a numpy array
    :return: the etical weight w, a positive real number
    """

    w = 0.0

    if len(hull) < 2:
        return w
    else:
        ethically_sorted_hull = hull[hull[:,1].argsort()]

        best_ethical = ethically_sorted_hull[-1]
        second_best_ethical = ethically_sorted_hull[-2]

        individual_delta = second_best_ethical[0] - best_ethical[0]
        ethical_delta = best_ethical[1] - second_best_ethical[1]

        if ethical_delta != 0:
            w = individual_delta/ethical_delta

        return w


def ethical_embedding(hull, epsilon,env, s0):
    """
    Repeats the ethical embedding process for each state in order to select the ethical weight that guarantees
    that all optimal policies are ethical.

    :param hull: the convex-hull-value function storing a partial convex hull for each state. The states are adapted
    to the public civility game.
    :param epsilon: the epsilon positive number considered in order to guarantee ethical optimality (it does not matter
    its value as long as it is greater than 0).
    :return: the desired ethical weight
    """
    w = 0.0
    
    states_entities = CHVI.get_states_of_all_entities(env)
    game_states = product(*states_entities)

    for game_state in game_states:
            game_state = list(game_state)
            starting_state = s0[0]
            if game_state == s0[0]:
                pass
            ethical_embedding_value = ethical_embedding_state(CHVI.get_value_by_index_list(hull, game_state, env))
            new_w = max(w, ethical_embedding_value)
            if new_w > w:
                    max_state= game_state
                    logger.info("Max state: %s, New ethical weight: %s", max_state, new_w)
            w = max(w, ethical_embedding_value)
            if game_state == s0[0]:
                 logger.info("Ethical weight for s0: %s", w+epsilon)

    return w + epsilon



def Ethical_Environment_Designer(env, epsilon, s0, discount_factor=1.0, max_iterations=5):
    """
    Calculates the Ethical Environment Designer in order to guarantee ethical
    behaviours in value alignment problems.


    :param env: Environment of the value alignment problem encoded as an MOMDP
    :param epsilon: any positive number greater than 0. It guarantees the success of the algorithm
    :param discount_factor: discount factor of the environment, to be set at discretion
    :param max_iterations: convergence parameter, the more iterations the more probabilities of precise result
    :return: the ethical weight that solves the ethical embedding problem
    """

    hull = CHVI.partial_convex_hull_value_iteration(env, discount_factor, max_iterations)
    ethical_weight = ethical_embedding(hull, epsilon, env, s0)

    return ethical_weight




