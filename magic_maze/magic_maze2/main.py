import copy
import statistics
import random
import numpy as np
from dataclasses import dataclass

class Maze:
    def __init__(self, locations):
        """
        :param locations: every location of the grid with the template: [value, rewards, next_value]
        :param actions: saves all the used actions
        :param endstates: a list with every endstate coordinates
        """
        self.locations = locations
        self.reward_locations_printed = False

    def print_locations(self):
        """
        Prints the maze lists in a fashionable style
        """
        print("\033[1m {}\033[00m".format("Values:"))
        print_line = "\033[34m {}\033[33m".format("|")
        for key in  self.locations.keys():
            value =  self.locations.get(key)
            print_line += ("\033[91m {}\033[00m" .format(round(value.v)))
            print_line += "\033[34m {}\033[33m".format("|")
            if 3 == key[0]:
                print(print_line)
                print_line = "\033[34m {}\033[33m".format("|")
        print(" ")
        if not self.reward_locations_printed:
            print("\033[1m {}\033[00m".format("Rewards:"))
            print_line = "\033[34m {}\033[33m".format("|")
            for key in self.locations.keys():
                value = self.locations.get(key)
                print_line += ("\033[91m {}\033[00m".format(value.r))
                print_line += "\033[34m {}\033[33m".format("|")
                if 3 == key[0]:
                    print(print_line)
                    print_line = "\033[34m {}\033[33m".format("|")
            print(" ")
            self.reward_locations_printed = True

    def set_locations(self, new_locations):
        self.locations = new_locations

    def get_locations(self):
        return self.locations


class Policy:
    def create_policy_grid(self, locations, discountfactor):
        """
        Creates a grid with arrows pointing towards the point it needs to go, while adding a X to every endstate
        :param locations: every location of the grid with the template: [value, rewards, next_value]
        :param discountfactor: the discountfactor used to calculate the best next value
        :param endstates: a list with every endstate coordinates
        :return: the grid with every arrow towards the next possible value aka the policy grid
        """
        possible_directions = ["←", "→", "↑", "↓"]
        print_line = "\033[34m {}\033[33m".format("|")
        for key in locations.keys():
            value = locations.get(key)
            if value.endstate:
                print_line += ("\033[91m {}\033[00m".format("X"))
            else:
                _, neighbour_values = self.select_action(key, locations, discountfactor)
                highest_index_states = [i for i, x in enumerate(neighbour_values) if x == max(neighbour_values)]
                direction = possible_directions[highest_index_states[0]]
                print_line += ("\033[91m {}\033[00m".format(direction))
                if len(highest_index_states) > 1:
                    for x in highest_index_states[1:]:
                        print_line += ("\033[91m {}\033[00m".format(possible_directions[x]))
            print_line += "\033[34m {}\033[33m".format("|")
            if 3 == key[0]:
                print(print_line)
                print_line = "\033[34m {}\033[33m".format("|")

    def select_action(self, current_coordinates, locations, discountfactor):
        neighbourclosevalues = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        neighbours_scores = []
        neighbours_coordinates = []
        for neighbour in neighbourclosevalues:
            neighbour_coordinates = tuple(np.add(current_coordinates, neighbour).clip(min=0, max=3).tolist())
            neighbour_values = locations.get(neighbour_coordinates)
            neighbours_coordinates.append(neighbour_coordinates)
            if locations.get(current_coordinates).endstate:
                neighbours_scores.append(0)
            else:
                neighbours_scores.append(neighbour_values.r+discountfactor*neighbour_values.v)
        neighbours_scores = neighbours_scores
        return neighbours_coordinates, neighbours_scores

class Agent:
    def __init__(self, m: Maze, pol: Policy):
        """
        :param m: The maze class
        :param pol: The policy class
        """
        self.m = m
        self.pol = pol
        self.discountfactor = 1
        self.gamma = 0.1


    def delta_definer(self):
        highest_delta = 0
        for loc_details in  self.m.locations.values():
            if abs(loc_details.next_v - loc_details.v) > highest_delta:
                highest_delta = abs(loc_details.next_v - loc_details.v)
        return highest_delta

    def value_iteration(self):
        """
        While loops until the delta (see function delta_definer) is above 0.1,
        if so it will loop through every location (with 2 loops) and runs the state_definer function to determine the next value (the 3rd value in grid_locations).
        after that it will determine the delta between the first and third value.
        after that it will put the third value in the first value like updating it value
        and at final it will print it's grid in a fashionable way.

        create a policy grid of the end result of the grid

        determines the next value when placed on a certain location (aka spawn location)
        :param spawn_location: the value used for the PolicyX.select_action function
        """

        self.m.print_locations()
        locations = self.m.get_locations()
        k = 0
        delta = 0.1
        while delta >= 0.1:
            k = k + 1
            print("\033[1m {}\033[00m".format("K=" + str(k)))
            for key in locations.keys():
                value = locations.get(key)
                _, next_value = self.pol.select_action(key, locations, self.discountfactor)
                value.next_v = max(next_value)

            delta = self.delta_definer()
            for key in locations.keys():
                value = locations.get(key)
                value.v = value.next_v
            self.m.set_locations(locations)
            self.m.print_locations()

        policyX.create_policy_grid(locations, self.discountfactor)

    def create_episode(self, start_position=(0,0)):
        """
        copy the position (as list because tuples dont have copy () and to use the np.add)
        runs the while loop until the position is an endstate
        create a random x value to move, if zero then random y value to move
        find the real coordinates by adding the list together eg. [0,3] + [1,0] (right neighbour) = [1,3]
        add this neighbour to the episode list
        :param start_position: creates episodes from this startpoint
        :return: a list with multiple random steps until it reaches the endstates
        """
        position = list(start_position).copy()
        episode = [position] #start position
        while not self.m.locations.get(tuple(position)).endstate:
            moving_x = random.randint(-1, 1)
            moving_y = 0
            if moving_x == 0: #prevents diagonal movements
                moving_y = random.randint(-1, 1)
            position = np.add(position, [moving_x, moving_y]).clip(min=0, max=3).tolist()
            episode.append(position)
        return episode

    def monte_carlo(self, start_position):
        locations = self.m.get_locations()
        run_times = 30000
        temp_locations = copy.deepcopy(locations)
        for x in range(run_times):
            episode = self.create_episode(start_position)
            G = 0
            for index, step in reversed(list(enumerate(episode))):
                if not temp_locations.get(tuple(step)).endstate:
                    G = self.discountfactor * G + temp_locations.get(tuple(episode[index+1])).r
                    if step not in episode[0:index]:
                        temp_locations.get(tuple(step)).g.append(G)
                        # self.m.locations.get(tuple(step)).v = statistics.mean(temp_locations.get(tuple(step)).g)
                        self.m.locations.get(tuple(step)).v = sum(temp_locations.get(tuple(step)).g) / len(temp_locations.get(tuple(step)).g)
        self.m.print_locations()
        policyX.create_policy_grid(locations, self.discountfactor)

    def td_learning(self):
        run_times = 30000
        for x in range(run_times):
            S = tuple(random.sample(range(0, 4), 2))
            while not self.m.locations.get(tuple(S)).endstate:
                next_coordinates, next_value = self.pol.select_action(S, locations, self.discountfactor)
                best_next_step = next_coordinates[next_value.index(max(next_value))]
                r2 = self.m.locations.get(tuple(best_next_step)).r
                v1 = self.m.locations.get(tuple(S)).v
                v2 = self.m.locations.get(tuple(best_next_step)).v
                self.m.locations.get(tuple(S)).v = v1 + self.gamma * (r2 + self.discountfactor*v2-v1)
                    #V(previous_state) = V(previous_state) + learning_rate * (R + discountfactor*V(current_value) - V(previousvalue))
                S = best_next_step
        self.m.print_locations()
        self.pol.create_policy_grid(self.m.locations, self.discountfactor)

    def policy_monte_carlo(self):
        pass

    def sarsa(self):
        pass

    def sarsa_max(self):
        pass

    def double_q_learning(self):
        run_times = 3000
        for x in range(run_times):
            episode = self.create_episode(start_position)
            for step in range(len(episode)):
                pass

@dataclass
class Location:
    x: int
    y: int
    v: int
    r: int
    next_v: int
    g: list
    endstate: bool

if __name__ == "__main__":
    #create the template given the exercise
    locations = {}
    for y in range(4):
      for x in range(4):
        locations[(x, y)] = Location(x=x, y=y, r=-1, endstate=False, v=0, next_v=0, g=[])
    locations[(3, 0)].r = 40
    locations[(3, 1)].r = -10
    locations[(2, 1)].r = -10
    locations[(0, 3)].r = 10
    locations[(1, 3)].r = -2
    locations[(3, 0)].endstate = True
    locations[(0, 3)].endstate = True
    mazeX = Maze(locations)
    policyX = Policy()
    agentX = Agent(mazeX, policyX)

    #algoritms
    agentX.value_iteration()

    start_position = (2,3)
    # agentX.monte_carlo(start_position)
    agentX.td_learning()





