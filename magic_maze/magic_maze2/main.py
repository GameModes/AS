import copy
import random
import numpy as np
from dataclasses import dataclass


class Maze:
    def __init__(self, locations):
        """
        :param locations: every location of the grid located in the dataclass location
        :param actions: saves all the used actions
        :param endstates: a list with every endstate coordinates
        """
        self.locations = locations
        self.reward_locations_printed = False

    def print_locations(self):
        """
        Prints the maze grid in a fashionable style and the rewards once (with rewards_locations_printed)
        """
        print("\033[1m {}\033[00m".format("Values:"))
        print_line = "\033[34m {}\033[33m".format("|")
        for key in self.locations.keys():
            value = self.locations.get(key)
            print_line += ("\033[91m {}\033[00m".format(round(value.v)))
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
        :param locations: every location of the grid located in the dataclass location
        :param discountfactor: the discountfactor used to calculate the best next value
        """
        print("\033[1m {}\033[00m".format("Policygrid:"))
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

    def create_control_policy_grid(self,locations):
        """
        The best direction is already calculated within every location, this function just places it on the right place to print it out
        :param locations: every location of the grid located in the dataclass location
        """
        print("\033[1m {}\033[00m".format("Control PolicyGrid:"))
        print_line = "\033[34m {}\033[33m".format("|")
        for key in locations.keys():
            value = locations.get(key)
            if value.endstate:
                print_line += ("\033[91m {}\033[00m".format("X"))
            else:
                direction = max(value.prob, key=value.prob.get)
                print_line += ("\033[91m {}\033[00m".format(direction))
            print_line += "\033[34m {}\033[33m".format("|")
            if 3 == key[0]:
                print(print_line)
                print_line = "\033[34m {}\033[33m".format("|")

    def select_action(self, current_coordinates, locations, discountfactor):
        """
        with the current coordinates if wil look to the values and rewards of the neigbours and returns every neighbours value in a list
        :param current_coordinates: the current coordinates on the maze grid
        :param locations: every location of the grid located in the dataclass location
        :param discountfactor: the factor/impact that the values of others have on calculating the value
        :return: a nested list with the coordinates of the neighbours and their values
        """
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
                neighbours_scores.append(neighbour_values.r + discountfactor * neighbour_values.v)
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

    def direction_converter(self, position, direction):
        """
        converts a direction with it's position (← & (1,0)) to the next position
        """
        direction_dictonary = {"←":[-1, 0], "→":[1, 0], "↑":[0, -1], "↓":[0, 1]}
        next_position = tuple(np.add(position, direction_dictonary.get(direction)).clip(min=0, max=3).tolist())
        return next_position

    def delta_definer(self):
        """
        compares the next_v and v (next value(k=1) and current value(K=0)) with each other to return the highest difference/delta
        """
        highest_delta = 0
        for loc_details in self.m.locations.values():
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
        print("\033[92m\033[1m\033[4m {}\033[00m".format("Value Iteration:"))
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

    def create_episode(self, start_position=(0, 0)):
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
        episode = [position]  # start position
        while not self.m.locations.get(tuple(position)).endstate:
            moving_x = random.randint(-1, 1)
            moving_y = 0
            if moving_x == 0:  # prevents diagonal movements
                moving_y = random.randint(-1, 1)
            position = np.add(position, [moving_x, moving_y]).clip(min=0, max=3).tolist()
            episode.append(position)
        return episode

    def create_control_episode(self, start_position=(0, 0)):
        """
        (montecarlo control uses probabity to explore instead of value so a new episode creator is needed)
        loops until the current position is a endstate
        creates random number to calculate which path is chosen with the probability dictionary
        looks around with random number which direction is chosen
        if so, breaks the for loop and add the step to the episode
        when it reaches the endstate append the last episode with cross to indicate it doesnt move anymore
        :param start_position:
        :return: the episode list with position and the direction
        """
        position = start_position
        episode = []  # start position
        while not self.m.locations.get(position).endstate:
            random_policy_number = round(random.uniform(0, 1), 3)
            for direction, value in self.m.locations.get(position).prob.items():
                if random_policy_number < value:
                    episode.append([position, direction])
                    position = self.direction_converter(position, direction)
                    break
            episode.append([position, "X"])
        return episode


    def monte_carlo(self):
        """
        creates a startpoint with random values
        runs the for loop x amount of times with a variable called run_times
        creates a G value which updates every step from the episode and if that step is the first one than add the G value to the value to the position
        adds the new G values from the positions to the maze grid
        """
        print("\033[92m\033[1m\033[4m {}\033[00m".format("Using monte_carlo:"))
        S = tuple(random.sample(range(0, 4), 2))
        print("\033[92m {}\033[00m".format("Start point: " + str(S)))
        locations = self.m.get_locations()
        run_times = 300000
        temp_locations = copy.deepcopy(locations)
        for x in range(run_times):
            print("\033[92m {}\033[00m".format("Start point: " + str(S)))
            print("\033[92m {}\033[00m".format("Creating Episode..."))
            episode = self.create_episode(S)
            print("\033[92m {}\033[00m".format("Episode Created!"))
            G = 0
            for index, step in reversed(list(enumerate(episode))):
                if not temp_locations.get(tuple(step)).endstate:
                    G = self.discountfactor * G + temp_locations.get(tuple(episode[index + 1])).r
                    if step not in episode[0:index]:
                        temp_locations.get(tuple(step)).g.append(G)
                        # self.m.locations.get(tuple(step)).v = statistics.mean(temp_locations.get(tuple(step)).g)
                        self.m.locations.get(tuple(step)).v = sum(temp_locations.get(tuple(step)).g) / len(
                            temp_locations.get(tuple(step)).g)
        self.m.print_locations()
        policyX.create_policy_grid(locations, self.discountfactor)

    def td_learning(self):
        """
        (in comparison of monte_carlo this algoritm updates every step.)
        create a S start_position
        gets the possible next coordinates and next value with select action function
        calculates the best next coordinates
        gets the values and rewards of the current and next coordinates to use in the formula to calculate the new value
        and changes the startposition to the next position
        """
        print("\033[92m\033[1m\033[4m {}\033[00m".format("Using TD Learning:"))
        run_times = 30000
        for x in range(run_times):
            print("\033[36m {}\033[00m".format("Run time: " + str(x)))
            S = tuple(random.sample(range(0, 4), 2))
            print("\033[92m {}\033[00m".format("Start point: " + str(S)))
            while not self.m.locations.get(tuple(S)).endstate:
                next_coordinates, next_value = self.pol.select_action(S, locations, self.discountfactor)
                best_next_coordinates = next_coordinates[next_value.index(max(next_value))]
                r2 = self.m.locations.get(tuple(best_next_coordinates)).r
                v1 = self.m.locations.get(tuple(S)).v
                v2 = self.m.locations.get(tuple(best_next_coordinates)).v
                self.m.locations.get(tuple(S)).v = v1 + self.gamma * (r2 + self.discountfactor * v2 - v1)
                # V(previous_state) = V(previous_state) + learning_rate * (R + discountfactor*V(current_value) - V(previousvalue))
                S = best_next_coordinates
        self.m.print_locations()
        self.pol.create_policy_grid(self.m.locations, self.discountfactor)

    def policy_monte_carlo(self):
        """
        (uses probilities instead of max values to determine next location)
        - loop with run_times
            - create_random_start_point
            - create a episode using a controlepisode function using probability
            - g = 0
            - reverse loop through it and skip endstate
                - g = discountfactor * g + next_value
                - check if g is in episode, if not
                    add it to the Q table as [direction, g]
                    get index of the highest G with the direction with Max
                    update with the highest value of Q table in prob dictionary with the formula: 1-gamma+gamma/4
                    update the rest of the values in prob dictionary with the formula: gamma/4
        """
        print("\033[92m\033[1m\033[4m {}\033[00m".format("Using Policy Monte Carlo:"))
        run_times = 30
        locations = copy.deepcopy(self.m.locations)
        for x in range(run_times):
            print("\033[36m {}\033[00m".format("Run time: " + str(x)))
            S = tuple(random.sample(range(0, 4), 2))
            print("\033[92m {}\033[00m".format("Start point: " + str(S)))
            print("\033[93m {}\033[00m".format("Creating Episode..."))
            episode = self.create_control_episode(S)
            print("\033[92m {}\033[00m".format("Episode Created!"))
            G = 0
            print("\033[93m {}\033[00m".format("Running through reversed Episode..."))
            for index, step in reversed(list(enumerate(episode))): #template: [state, action]
                if not locations.get(tuple(step[0])).endstate:
                    G = self.discountfactor * G + locations.get(tuple(episode[index + 1][0])).r
                    if step[0] not in episode[0:index]:
                        self.m.locations.get(tuple(step[0])).qtable[step[1]] = G
                        best_direction = max(self.m.locations.get(tuple(step[0])).qtable, key=self.m.locations.get(tuple(step[0])).qtable.get)
                        self.m.locations.get(tuple(step[0])).prob[best_direction] = 1 - self.gamma + self.gamma / 4
                        for probabilities in self.m.locations.get(tuple(step[0])).prob.items():
                            if probabilities[0] != best_direction:
                                self.m.locations.get(tuple(step[0])).prob[probabilities[0]] = self.gamma/4
            print("\033[92m {}\033[00m".format("Done!"))
        self.pol.create_control_policy_grid(self.m.locations, self.discountfactor)

    def sarsa(self):
        pass

    def sarsa_max(self):
        pass


@dataclass
class Location:
    x: int
    y: int
    v: int
    r: int
    next_v: int
    g: list
    prob: dict
    qtable: dict
    endstate: bool


if __name__ == "__main__":
    # create the template given the exercise
    locations = {}
    for y in range(4):
        for x in range(4):
            locations[(x, y)] = Location(x=x, y=y, r=-1, endstate=False, v=0, next_v=0, g=[],
                                         prob={"←": 0.25, "→": 0.5, "↑": 0.75, "↓": 1}, qtable={"←": 0, "→": 0, "↑": 0, "↓": 0})
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

    # algoritms
    # agentX.value_iteration()

    start_position = (2, 3)
    # agentX.monte_carlo()
    agentX.td_learning()
    # agentX.policy_monte_carlo()
