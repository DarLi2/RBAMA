from itertools import chain, combinations
from collections import deque
import numpy as np
from src.environments import bridge_person

"""translates from abstract action types (moral obligations) to trajectories of primitive actions (one possible understanding of what counts as fulfillment of the moral obligations)"""
class Translator():
    def __init__(self):
        super().__init__()

    def index_to_2d(self, index, width):
        row = index // width
        col = index % width
        return (col, row)
    
    """breadth first search for finding a shortest path to a drowning person"""
    @classmethod
    def shortest_paths_bfs(cls, start, goal, grid_size, env):
        directions = env.directions

        queue = deque([(start, [start])]) 
        visited = {}
        shortest_paths = []
        shortest_length = None
        
        while queue:
            current_position, path = queue.popleft()

            if np.array_equal(current_position, goal):
                path_length = len(path) - 1
                if shortest_length is None:
                    shortest_length = path_length
                if path_length == shortest_length:
                    shortest_paths.append(path[1:]) 
                continue

            for direction in directions:
                neighbor = tuple(np.array(current_position) + direction)

                if env.bridge_map.location_in_grid(neighbor) and env.bridge_map.get_grid_type(neighbor) != env.bridge_map.grid_types["water"]:
                    if neighbor not in visited or len(visited[neighbor]) == len(path):
                        visited[neighbor] = path
                        queue.append((neighbor, path + [neighbor]))

        return shortest_paths
        
    @classmethod
    def next_steps_rescue(cls, env, person_position):

        directions = env.directions
        start_position = env.get_agent_location()
        next_actions = []

        #assumption: the person which has fallen into the water first should first be rescued first
        land_tiles_next_to_person = []

        for direction in directions:
            neighbor = tuple(tuple(person_position)  + direction)
            if 0  < neighbor[0] < env.bridge_map.width and  0 < neighbor[1] < env.bridge_map.height:
                if env.bridge_map.get_grid_type(neighbor) != env.bridge_map.grid_types["water"]:
                    land_tiles_next_to_person.append(neighbor)

        def manhattan_distance(pos1, pos2):
            return abs(pos2[0] - pos1[0]) + abs(pos2[1] - pos1[1])
        
        min_distance = min(manhattan_distance(start_position, pos) for pos in land_tiles_next_to_person)
        closest_land_tiles = [pos for pos in land_tiles_next_to_person if manhattan_distance(start_position, pos) == min_distance]

        shortest_paths = []

        for land_tile in closest_land_tiles:
            shortest_paths.extend(cls.shortest_paths_bfs(start_position, land_tile, env.bridge_map.width, env))

        if any(len(path) > 0 for path in shortest_paths):
            next_steps = {path[0] for path in shortest_paths}
            direction_to_action = {tuple(direction): action for action, direction in env.action_to_direction.items()}
            for next_step in next_steps:
                for direction in  direction_to_action:
                    if np.array_equal(start_position + np.array(direction), next_step):
                        next_actions.append(direction_to_action[direction]) 
        else:
            next_actions = [4]

        return next_actions
    
    @classmethod
    def impermissible(cls, obligation, env):

        action_space = [action for action in range(env.action_space.n)]
        impermissible_actions = [] 

        if 'C' == obligation:
            if env.agent_behind_bridge():
                impermissible_actions = [3]
            elif env.agent_in_front_of_bridge():
                impermissible_actions = [1]
            elif env.agent_on_bridge():
                impermissible_actions = [0,1,2,3]
            return impermissible_actions
        if 'R' == obligation:
            drowning_spots = []
            for person in env.persons:
                if person.status == bridge_person.Status.IN_WATER_AT_RISK:
                    drowning_spots.append(person.position)
            person_to_rescue = drowning_spots[0] 
            morally_permissible_R = cls.next_steps_rescue(env, person_to_rescue)
            for action in action_space:
                if action not in morally_permissible_R:
                    impermissible_actions.append(action)
            return impermissible_actions
