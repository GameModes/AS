import copy
import random
import numpy as np
from dataclasses import dataclass
from statistics import mean

class Policy:
    def __init__(self, discountfactor=0.9):
        self.directions = [(0, -1), (0, 1),(-1, 0), (1, 0)] #directions the player can go: [(up), (under), (left), (right)] or ["↑", "↓", "←", "→"]
        self.discountfactor = discountfactor #the amount of change to the value every k

        #for control:
        self.q = {}
        self.returns = {}
        self.pi = {}
        self.ep = 0.1

    def create_empty_control_values(self):
        """
        creates empty q, returns and pi dictionary and add the possibilities to pi
        """
        for x in range(4):
            for y in range(4):
                for direction in self.directions:
                    self.q[((x, y), direction)] = 0 #start best value is 0
                    self.returns[((x, y), direction)] = [] #start with empty list
                if (x, y) == (0, 3) or (x, y) == (3, 0):  # if the x and y are endstate dont add possibilities
                    self.pi[(x, y)] = [0, 0, 0, 0]
                else:
                    direction_possibilities = [0.025, 0.925, 0.025, 0.025]
                    random.shuffle(direction_possibilities) #to create a randomness so not every point goes the same way
                    self.pi[(x, y)] = direction_possibilities

    def select_action(self, locations, current_state) -> object:
        """
        looks around with the directions and calculates which direction has the best value
        :param locations: the entire grid information
        :param current_state: the current state/coordinates
        :return: the best direction the player can go to
        """
        values = []

        for direction in self.directions:
            neighbour_coordinates = tuple(np.add((current_state.x, current_state.y), direction).clip(min=0, max=3).tolist())
            neighbour_values = locations[neighbour_coordinates]
            value = self.discountfactor * neighbour_values.v + neighbour_values.r
            values.append(value)
        best_value_index = np.argmax(values)
        return self.directions[best_value_index]

    def select_q_action(self, curr_state, epsilon) -> object:
        """
        Calculates the best direction looking at the Q dictionary
        :param curr_state: the current place
        :param epsilon: epsilon value
        :return: the best direction
        """
        if random.random() > epsilon:  # random.random() is number between 0.0 and 1.0 so if epsilon is not hit
            action_values = {direction: self.q[((curr_state.x, curr_state.y), direction)] for direction in self.directions} #creates a dictionary to connect the direction with the actionvalue
            direction = max(action_values, key=action_values.get) #uses the dictionary to get the key with the highest value
        else:
            direction = random.choice(self.directions) #if episolon is hit choose a random direction
        return direction

    def print_pi(self):
        """
        prints out the grid with Pi
        """
        arrow_directions = ["↑", "↓", "←", "→"]
        print_line = "\033[34m Monte carlo Policy: \033[33m\n"
        for y in range(4):
            print_line += "\033[34m {}\033[33m".format("|")
            for x in range(4):
                if (self.pi[(x, y)]) == [0, 0, 0, 0]:
                    print_line += "X"
                else:
                    print_line += arrow_directions[np.argmax(np.array(self.pi[(x, y)]))]
                print_line += "\033[34m {}\033[33m".format("|")
            print_line += "\n"
        print(print_line)

        print("\033[34mDirection Order: ↑, ↓, ←, → \033[33m")
        print_str = ""
        for y in range(4):
            for x in range(4):
                print_str += "\033[34m[\033[33m{}:\033[31m {}\033[34m] ".format((x, y), str(self.pi[(x, y)]))
            print_str += "\n"
        print(print_str)

    def print_Q(self):
        """
            Prints out the entire grid and its Q values, with each coordinate showing all the Q values of each direction.
        """
        arrow_directions = ["↑", "↓", "←", "→"]
        print_line = "\033[34m Sarsa: \033[33m\n"
        for y in range(4):
            print_line += "\033[34m {}\033[33m".format("|")
            for x in range(4):
                if (self.pi[(x, y)]) == [0, 0, 0, 0]:
                    print_line += "X"
                else:
                    all_directions = []
                    for direction in self.directions:
                       all_directions.append(round(self.q[((x, y), direction)], 1))
                    print_line += arrow_directions[np.argmax(np.array(all_directions))]
                print_line += "\033[34m {}\033[33m".format("|")
            print_line += "\n"
        print(print_line)


        print("\033[93mDirection corresponding to each Q value: ↑, ↓, ←, →  \033[35m")
        print_str = "\033[93mQ values: \033[35m\n"
        for y in range(4):
            for x in range(4):
                print_str += "\033[34m[\033[33m{}: ".format(str((x, y)))
                for direction in self.directions:
                    print_str += "\033[31m {}, ".format(round(self.q[((x, y), direction)], 1))
                print_str += "\033[34m]"
            print_str += "\n"
        print(print_str)

    def get_directions(self):
        return self.directions



class Maze:
    def __init__(self, entire_maze):
        """
        :param entire_maze: every location of the grid located in the dataclass location
        :param actions: saves all the used actions
        :param endstates: a list with every endstate coordinates
        """
        self.locations = entire_maze
        self.reward_locations_printed = False

    def print_locations(self):
        """
        Prints the maze grid in a fashionable style and the rewards once (with rewards_locations_printed)
        """

        print("\033[34m {}\033[00m".format("Monte Carlo Non Policy:"))
        print("\033[33m {}\033[00m".format("Values:"))
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

#TODO Document this:
class Agent:
    def __init__(self, m: Maze, pol:Policy, discountfactor=0.9):
        """
        :param m: the maze grid class
        :param pol: the policy class
        """
        self.m = m
        self.pol = pol
        self.discountfactor = discountfactor
        self.gamma = 0.1
        self.epsilon = 0.1

    def monte_carlo_non_policy(self):
        "First visit monte carlo non_policy"
        returns_s = {x: [] for x in self.m.locations} #create empty return dictionary
        for k in range(10000):
            episode = self.non_policy_episode_creator()  # make episode
            G = 0
            for index, element in reversed(list(enumerate(episode))):  # reverse looping through epioside with correct index
                state = self.m.locations[element[0]] #gets the state with the coordinates from the maze grid
                current_step = (state.x, state.y)
                G = self.discountfactor*G + state.r
                if state not in [self.m.locations[move[0]] for move in episode[0: index]]: #to check if it's the first in the episode
                    returns_s[current_step].append(G)
                    self.m.locations[current_step].v = mean(returns_s[current_step])
        self.m.print_locations()

    def monte_carlo_policy(self):
        "First visit monte carlo policy"
        self.pol.create_empty_control_values()
        for k in range(10000):
            episode = self.policy_episode_creator()
            G = 0
            for index, element in reversed(list(enumerate(episode))):  # reverse looping through epioside with correct index
                if index != 0: #not the first one
                    curr_step = (element[0].x, element[0].y) #element[0] = state,  element[1] = best direction, element[2] = reward for moving
                    G = self.discountfactor * G + element[2]
                    if [episode[index][0], episode[index][1]] not in [[temp_element[0], temp_element[1]] for temp_element in episode[0: index]]: # to check if it's the first in the episode
                        #appends every movement to returns
                        self.pol.returns[curr_step, element[1]].append(G)
                        #gets the average value of the movement and save it in q
                        self.pol.q[curr_step, element[1]] = mean(self.pol.returns[curr_step, element[1]])
                        #gets the best value of the movement actions in save it in a_star
                        action_values = {direction: self.pol.q[(curr_step, direction)] for direction in self.pol.get_directions()}
                        a_star_action = max(action_values, key=action_values.get) #https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary

                        #for all a as element A
                        for direction_index in range(len(self.pol.get_directions())):
                            if self.pol.get_directions()[direction_index] == a_star_action: #if the best direction

                                self.pol.pi[curr_step][direction_index] = 1 - self.pol.ep + (self.pol.ep / 4) #add it with the highest chance of occurring
                            else:
                                self.pol.pi[curr_step][direction_index] = (self.pol.ep / 4)
        self.pol.print_pi()

    def sarsa(self):
        """
        Sarsa, create a new episode after every new q values
        """
        S = 0.1
        self.pol.create_empty_control_values()
        for k in range(10000):
            curr_state = self.m.locations[random.choice(list(self.m.locations.keys()))]  # https://www.w3schools.com/python/ref_random_choice.asp
            curr_best_direction = self.pol.select_q_action(curr_state, self.epsilon)
            while curr_state.endstate != True:
                next_coordinates = tuple(np.add(curr_best_direction, (curr_state.x, curr_state.y)).clip(min=0, max=3).tolist())
                next_state = self.m.locations[next_coordinates]
                R = next_state.r
                next_best_direction = self.pol.select_q_action(next_state, self.epsilon)
                self.pol.q[((curr_state.x, curr_state.y), curr_best_direction)] += S * (R + self.discountfactor * self.pol.q[((next_state.x, next_state.y), next_best_direction)] - self.pol.q[((curr_state.x, curr_state.y), curr_best_direction)])
                curr_state = next_state
                curr_best_direction = next_best_direction
        print(self.pol.print_Q())

    def non_policy_episode_creator(self):
        """
        :return: an episode, a list with positions and actions, that chooses his next position with the select_action() until the state is an endstate
        """
        curr_state = self.m.locations[random.choice(list(self.m.locations.keys()))] #https://www.w3schools.com/python/ref_random_choice.asp
        episode = [((curr_state.x, curr_state.y), curr_state.r)]
        while curr_state.endstate is False:
            best_direction = self.pol.select_action(self.m.locations, curr_state)
            if random.random() < self.epsilon:  #random.random() is number between 0.0 and 1.0 so if episilon is hit
                best_direction = random.choice(self.pol.get_directions())
            next_coordinates = tuple(np.add(best_direction, (curr_state.x, curr_state.y)).clip(min=0, max=3).tolist())
            next_state = self.m.locations[next_coordinates]
            episode.append((next_coordinates, next_state.r))
            curr_state = next_state
        return episode

    def policy_episode_creator(self):
        """
        :return: an episode, a list with positions and actions, that chooses his next position with probabilites made with the pi dictionary until the state is an endstate
        """
        curr_state = self.m.locations[random.choice(list(self.m.locations.keys()))] #https://www.w3schools.com/python/ref_random_choice.asp
        episode = []
        while curr_state.endstate is False:
            best_direction_index = np.random.choice([0, 1, 2, 3], p=self.pol.pi[(curr_state.x, curr_state.y)]) #use the pi from policy as probabilities to choose a side
            best_direction = self.pol.get_directions()[best_direction_index]
            next_coordinates = tuple(np.add(best_direction, (curr_state.x, curr_state.y)).clip(min=0, max=3).tolist())
            next_state = self.m.locations[next_coordinates]
            episode.append((curr_state, best_direction, next_state.r))
            curr_state = next_state
        return episode


@dataclass
class Location:
    x: int #x-coordinate
    y: int #y-coordinate
    v: int #value
    r: int #reward
    next_v: int #next possible value
    g: list #a list to hold the G values (used in td_learning)
    endstate: bool

if __name__ == "__main__":
    # create the maze
    entire_maze = {}
    for y in range(4):
        for x in range(4):
            entire_maze[(x, y)] = Location(x=x, y=y, r=-1, endstate=False, v=0, next_v=0, g=[])
    entire_maze[(3, 0)].r = 40
    entire_maze[(3, 1)].r = -10
    entire_maze[(2, 1)].r = -10
    entire_maze[(0, 3)].r = 10
    entire_maze[(1, 3)].r = -2
    entire_maze[(3, 0)].endstate = True
    entire_maze[(0, 3)].endstate = True
    mazeA = Maze(entire_maze)
    mazeB = Maze(entire_maze)
    mazeC = Maze(entire_maze)

    discountfactor = 1
    #create policy
    policyA = Policy(discountfactor)
    policyB = Policy(discountfactor)
    policyC = Policy(discountfactor)

    #create the agent
    AgentA = Agent(mazeA, policyA, discountfactor)
    AgentA.monte_carlo_non_policy()

    AgentB = Agent(mazeB, policyB, discountfactor)
    AgentB.monte_carlo_policy()

    AgentC = Agent(mazeC, policyC, discountfactor)
    AgentC.sarsa()