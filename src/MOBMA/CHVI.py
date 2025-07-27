import numpy as np
import src.MOBMA.convexhull as convexhull
import gymnasium as gym
from src.MOBMA.ethical_momdp_wrapper import Ethical_MOMDP_Wrapper
from itertools import product
from src.environments import registered_versions
import logging

logger = logging.getLogger(__name__)

def get_states_of_all_entities(env):
     return [list(env.states[key]) for key in sorted(env.states.keys())]
     
def create_nested_list(shape, fill_value=None):
        if len(shape) == 1:
            return [np.array([]) if fill_value is None else fill_value for _ in range(shape[0])]
        return [create_nested_list(shape[1:], fill_value) for _ in range(shape[0])]

def get_V_indices(state,env):
    indices = []
    for i in range(0,len(state)):
        env_states = sorted(env.states[i])
        indices.append(env_states.index(state[i]))
    return indices 

def get_value_by_index_list(nested_structure, state, env):
    """Access a value in a nested structure given a list of indices."""
    indices = get_V_indices(state, env)
    value = nested_structure
    for idx in indices:
        value = value[idx]
    return value.copy()  

def assign_to_nested_list(nested_list, state, value, env):
    """
    Assign a value to a nested list dynamically based on indices.
    
    :param nested_list: The nested list structure.
    :param indices: A list of indices representing the path to the innermost level.
    :param value: The value to assign.
    """
    indices = get_V_indices(state, env)
    current = nested_list
    for idx in indices[:-1]:  
        current = current[idx]
    current[indices[-1]] = value 

def Q_function_calculator(env, state, V, discount_factor):
    """

    Calculates the (partial convex hull)-value of applying each action to a given state.
    Heavily adapted to the public civility game

    :param env: the environment of the Markov Decision Process
    :param state: the current state
    :param V: value function to see the value of the next state V(s')
    :param discount_factor: discount factor considered, a real number
    :return: the new convex obtained after checking for each action (this is the operation hull of unions)
    """
    hulls = list()
    for action in range(0,5):

            if state == [41, 49, 49, 49, 49, 34] and action == 4:
                 pass
            env.reset(state=state)
            next_state, rewards, terminated, _, _ = env.step(action)
            if rewards[1] != 0:
                 pass

            if not terminated:
                V_state = get_value_by_index_list(V, next_state, env).copy()
            else:
                V_state = np.array([])

            hull_sa = convexhull.translate_hull(rewards, discount_factor, V_state)
            if state == [48, 49, 49, 49, 39] and not np.equal(hull_sa,np.array([0,0])).all():
                 pass
            if len(hull_sa) > 2:
                pass
    
            for point in hull_sa:
                if point[1] > 0:
                     pass
                hulls.append(point)

    hulls = np.unique(np.array(hulls), axis=0)

    new_hull = convexhull.get_hull(hulls)

    return new_hull


def partial_convex_hull_value_iteration(env, discount_factor=1.0, max_iterations=10):
    """
    Partial Convex Hull Value Iteration algorithm adapted from "Convex Hull Value Iteration" from
    Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Calculates the partial convex hull for each state of the MOMDP

    :param env: the environment encoding the MOMDP
    :param discount_factor: discount factor of the environment, to be set at discretion
    :param max_iterations: convergence parameter, the more iterations the more probabilities of precise result
    :return: value function storing the partial convex hull for each state
    """

    states_entities = [list(env.states[key]) for key in sorted(env.states.keys())]

    states_dimensions = [len(state) for state in [state for state in states_entities]]
    V = create_nested_list(states_dimensions)

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        game_states = product(*states_entities)
        for game_state in game_states:
            game_state = list(game_state)
            value_estimation = Q_function_calculator(env, game_state, V, discount_factor)
            assign_to_nested_list(V, game_state, value_estimation, env)
        logger.info("Iterations: %d/%d", iteration, max_iterations)
    return V


if __name__ == "__main__":
    env=gym.make("bridge1_v0_testMO")
    env = Ethical_MOMDP_Wrapper(env, obs = 'CHVI')
    v = partial_convex_hull_value_iteration(env, discount_factor=0.9)




