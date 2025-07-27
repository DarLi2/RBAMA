
#!/usr/bin/env python3
import argparse
from src.MOBMA import MOBMA
import gymnasium as gym
from src.MOBMA.ethical_momdp_wrapper import Ethical_MOMDP_Wrapper
from src.MOBMA.ethical_weight import Ethical_Environment_Designer
import signal
import time
import logging

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout! Ethical_Environment_Designer took too long.")


def main():

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Calculate the ethical weight for an instance of the bridge environment.")
    
    parser.add_argument('env_id', type=str, help="Id of the instance of the bridge environment")
    parser.add_argument('--epsilon', type=float, default=0.1,help="Set the epsilon for the ethical weight calculation")
    parser.add_argument('--discount', type=float,default=0.9, help="Set the discount factor for the environment")
    parser.add_argument('--max_iterations', type=int, default=15, help="Set the maximum number of iterations for the ethical weight calculation")

    args = parser.parse_args()

    env=gym.make(args.env_id)
    env = Ethical_MOMDP_Wrapper(env, obs_config = 'CHVI')
    epsilon = args.epsilon
    discount_factor = args.discount
    max_iterations = args.max_iterations
    s0 = env.reset(random_init = "no randomness")

    timeout = 7200  # timeout after 2 hours
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        start_time = time.time()
        # throws an error if state space was estimated incorrectly
        w_E = Ethical_Environment_Designer(env, epsilon, s0, discount_factor, max_iterations)
        signal.alarm(0)
    except TimeoutException:
        w_E = None
        logger.info("timeout")
    finally:
        end_time = time.time()
        logger.info("Execution time: %.2f seconds", end_time - start_time)

    logger.info("Ethical weight if initial positions are random: %s", w_E)
    logger.info("Execution time: %.4f seconds", end_time - start_time)

if __name__ == "__main__":
    main()
