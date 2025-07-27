from src.environments import bridge
import gymnasium as gym

def register_bridge(   
        env_id, # unique name of the envrionment
         #setting up persons
        ids_moving_persons, #persons that move across the bridges (id sets the bridge which the person crosses; 1:leftest bridge); must not include a number larger than the number of bridges
        pos_static_persons, #persons standing on the map without moving; they directly spawn after a map reset
        drowning, 
        drowning_time, 
        slipping_prob, # person fall into the water when True; they don't when False
        pushed_off_bridge_prob, #persons are pushed off the bridge when True; they don't when False
        respawn_timer, # controls the probablity at which moving persons spawn initially and the speed at which persons resapwn after drowning
        #setting up map and rendering
        render_mode,
        num_bridges, #max number of bridges is 3
        dangerous_spots, #spots where persons might fall into the water
        width, #number of grids; should be at least 3 for 1 bridge; 5 for 2 bridges; and 9 for 3 bridges; must be and odd number
        height, #number of grids; should be at least 5 (2 land lines on top, 2 on the bottom and water line)
        target_location,
        reward_type = 'instrumental'
        ):
    if env_id not in gym.envs.registry:
        gym.register(
        id=env_id,
        entry_point="src.environments.bridge:Bridge",  
        max_episode_steps=100,
        kwargs={
            'ids_moving_persons': ids_moving_persons, 
            'pos_static_persons': pos_static_persons, 
            'drowning': drowning, 
            'drowning_time': drowning_time, 
            'slipping_prob': slipping_prob, 
            'pushed_off_bridge_prob': pushed_off_bridge_prob, 
            'respawn_timer': respawn_timer, 
            'render_mode': render_mode, 
            'num_bridges': num_bridges, 
            'dangerous_spots': dangerous_spots, 
            'width': width, 
            'height': height,
            'target_location': target_location,
            "reward_type": reward_type
        }
)