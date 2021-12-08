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
        print_line = ""
        for key in  self.locations.keys():
            value =  self.locations.get(key)
            print_line += ("\033[34m {}\033[33m".format("|") + "\033[91m {}\033[00m" .format(value.v))
            print_line += "\033[34m {}\033[33m".format("|")
            if 3 == key[0]:
                print(print_line)
                print_line = ""
        print(" ")
        if not self.reward_locations_printed:
            print("\033[1m {}\033[00m".format("Rewards:"))
            print_line = ""
            for key in self.locations.keys():
                value = self.locations.get(key)
                print_line += ("\033[34m {}\033[33m".format("|") + "\033[91m {}\033[00m".format(value.r))
                print_line += "\033[34m {}\033[33m".format("|")
                if 3 == key[0]:
                    print(print_line)
                    print_line = ""
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
        print_line = ""
        for key in locations.keys():
            value = locations.get(key)
            if value.endstate:
                print_line += ("\033[34m {}\033[33m".format("|") + "\033[91m {}\033[00m".format("X"))
            else:
                neighbour_values = self.select_action(key, locations, discountfactor)
                # print(neighbour_values)
                highest_index_states = [i for i, x in enumerate(neighbour_values) if x == max(neighbour_values)]
                direction = possible_directions[highest_index_states[0]]
                print_line += ("\033[34m {}\033[33m".format("|") + "\033[91m {}\033[00m".format(direction))
                if len(highest_index_states) > 1:
                    for x in highest_index_states[1:]:
                        print_line += ("\033[91m {}\033[00m".format(possible_directions[x]))
            print_line += "\033[34m {}\033[33m".format("|")
            if 3 == key[0]:
                print(print_line)
                print_line = ""

        # return policygrid

    def select_action(self, current_coordinates, locations, discountfactor):
        neighbourclosevalues = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        neighbours_scores = []
        for neighbour in neighbourclosevalues:
            neighbour_coordinates = tuple(np.add(current_coordinates, neighbour).clip(min=0, max=3).tolist())
            neighbour_values = locations.get(neighbour_coordinates)
            if locations.get(current_coordinates).endstate:
                neighbours_scores.append(0)
            else:
                neighbours_scores.append(neighbour_values.r+discountfactor*neighbour_values.v)
        neighbours_scores = neighbours_scores
        return neighbours_scores

class Agent:
    def __init__(self, m: Maze, pol: Policy):
        """
        :param m: The maze class
        :param pol: The policy class
        """
        self.m = m
        self.pol = pol
        self.discountfactor = 0.9


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
                value.next_v = max(self.pol.select_action(key, locations, self.discountfactor))
            delta = self.delta_definer()
            for key in locations.keys():
                value = locations.get(key)
                value.v = value.next_v
            self.m.set_locations(locations)
            self.m.print_locations()

        policygrid = policyX.create_policy_grid(locations, self.discountfactor)
        # print("\033[1m {}\033[00m".format("Policygrid "))
        # self.maze_valueprinter(policygrid)

    def create_episode(self, start_position):
        position = list(start_position).copy()
        episode = [position]
        while not self.m.locations.get(tuple(position)).endstate:
            moving_x = random.randint(-1, 1)
            if moving_x != -1 or moving_x != 1:
                moving_y = random.randint(-1, 1)
            position = np.add(position, [moving_x, moving_y]).clip(min=0, max=3).tolist()
            episode.append(position)
        return episode

    def monte_carlo(self, start_position):
        locations = self.m.get_locations()
        run_times = 3000
        for x in range(run_times):
            episode = self.create_episode(start_position)
            # print(episode)
            G = 0
            for index, step in reversed(list(enumerate(episode))):
                if not self.m.locations.get(tuple(step)).endstate and episode[index] != episode[-1]:
                    # print("This: ", tuple(episode[index]), " next: ", tuple(episode[index + 1]))
                    G = self.discountfactor * G + self.m.locations.get(tuple(episode[index+1])).r
                    if step not in episode[0:index]:
                        self.m.locations.get(tuple(step)).g.append(G)
                        self.m.locations.get(tuple(step)).v = statistics.mean(self.m.locations.get(tuple(step)).g)
            self.m.print_locations()
        policyX.create_policy_grid(locations, self.discountfactor)


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
    locations[(3, 0)].endstate = True
    locations[(0, 3)].endstate = True
    mazeX = Maze(locations)
    policyX = Policy()
    agentX = Agent(mazeX, policyX)

    #algoritms
    # agentX.value_iteration()
    start_position = (0,1)
    agentX.monte_carlo(start_position)





