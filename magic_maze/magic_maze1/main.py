import copy
import numpy as np, random

class Maze:
    def __init__(self, locations, actions, endstates):
        """

        :param locations: every location of the grid with the template: [value, rewards, next_value]
        :param actions: saves all the used actions
        :param endstates: a list with every endstate coordinates
        """
        self.locations = locations
        self.actions = actions
        self.endstates = endstates

    def step(self, state):
        """
        empty for now
        """
        pass

    def get_actions(self):
        """
        gets every action
        """
        return self.actions

class Policy:
    def __init__(self):
        pass

    def create_policy_grid(self, grid_locations, discountfactor, endstates):
        """
        Creates a grid with arrows pointing towards the point it needs to go, while adding a X to every endstate
        :param grid_locations: every location of the grid with the template: [value, rewards, next_value]
        :param discountfactor: the discountfactor used to calculate the best next value
        :param endstates: a list with every endstate coordinates
        :return: the grid with every arrow towards the next possible value aka the policy grid
        """
        possible_directions = ["↑", "↓", "←", "→"]
        neighbours = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        policygrid = copy.deepcopy(grid_locations)
        for row in range(len(grid_locations)):
            for column in range(len(grid_locations)):
                if [row, column] in endstates:
                    policygrid[row][column] = "X"
                else:
                    neighbour_coordinates = [np.add([row, column], neighbour).clip(min=0, max=3).tolist() for neighbour in
                                             neighbours]
                    neighbour_values = []
                    for neighbour in neighbour_coordinates:
                        if neighbour == [row, column]:
                            neighbour_values.append(0)
                        else:
                            neighbour_values.append(grid_locations[neighbour[0]][neighbour[1]][0] + discountfactor * grid_locations[neighbour[0]][neighbour[1]][1])
                    highest_index_states = [i for i, x in enumerate(neighbour_values) if x == max(neighbour_values)]
                    policygrid[row][column] = possible_directions[highest_index_states[0]]
                    if len(highest_index_states) > 1:
                        for x in highest_index_states[1:]:
                            policygrid[row][column] = policygrid[row][column] + str(possible_directions[x])
        return policygrid

    def select_action(self, location, grid_locations, discountfactor):
        """
        (needs refinement) It will search the next best value coordinates when placed on a location
        :param location: The value that is used to determine the begin place
        :param grid_locations: every location of the grid with the template: [value, rewards, next_value]
        :param discountfactor: the discountfactor used to calculate the best next value
        :return: the next location when given a location
        """
        neighbourclosevalues = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        neighbour_coordinates = [np.add(location, neighbour).clip(min=0, max=3).tolist() for neighbour in
                                 neighbourclosevalues]
        neighbour_values = [grid_locations[neighbour[1]][neighbour[0]][0] + discountfactor *
                            grid_locations[neighbour[1]][neighbour[0]][1] for neighbour in neighbour_coordinates]
        highest_states = [i for i, x in enumerate(neighbour_values) if x == max(neighbour_values)]
        if len(highest_states) > 1:
            next_location = neighbour_coordinates[random.choice(highest_states)]
        else:
            next_location = neighbour_coordinates[highest_states[0]]
        return next_location

class Agent:
    def __init__(self, m: Maze, pol: Policy):
        """
        :param m: The maze class
        :param pol: The policy class
        """
        self.m = m
        self.pol = pol

    def action(self):
        pass

    def value(self):
        pass

    def state_definer(self, current_coordinates):
        """
        retrieves every neighbours scores (value used in formula) in a list (unless they are endstates which returns 0)
        :param current_coordinates: the current coordinates that the neighbours needs to be calculated form
        :return: every score of the neighbours in a list
        """
        neighbourclosevalues = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        neighbours_scores = []
        for neighbour in neighbourclosevalues:
            neighbour_coordinates = np.add(current_coordinates, neighbour).clip(min=0, max=3).tolist()
            neighbour_values = self.m.locations[neighbour_coordinates[1]][neighbour_coordinates[0]]
            if current_coordinates in self.m.endstates:
                neighbours_scores.append(0)
            else:
                neighbours_scores.append(neighbour_values[1]+self.discountfactor*neighbour_values[0])
        return neighbours_scores

    def maze_valueprinter(self, locations):
        """
        Prints the maze lists in a fashionable style
        :param locations: every location of the grid with the template: [value, rewards]
        """
        for row in locations:
            print_line = ""
            for column in row:
                print_line += ("\033[34m {}\033[33m".format("|") + "\033[91m {}\033[00m" .format(column[0]))
            print_line += "\033[34m {}\033[33m".format("|")
            print(print_line)
        print(" ")

    def delta_definer(self):
        """
        compares the earlier used locations with the current ones to determine the delta value
        :return: the delta value
        """
        delta = 0
        for row in self.m.locations:
            for column in row:
                if abs(column[2] - column[0]) > delta:
                    delta = abs(column[2] - column[0])
        return delta

    def value_iteration(self, spawn_location):
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
        self.discountfactor = 1
        self.maze_valueprinter(self.m.locations)
        k=0
        delta = 0.1
        while delta >= 0.1:
            k=k+1
            print("\033[1m {}\033[00m".format("K=" + str(k)))
            for row in range(len(self.m.locations)):
                for column in range(len(self.m.locations[row])):
                    self.m.locations[row][column][2] = max(self.state_definer([column, row]))
            delta = self.delta_definer()
            for row in range(len(self.m.locations)):
                for column in range(len(self.m.locations[row])):
                    self.m.locations[row][column][0] = self.m.locations[row][column][2]

            self.maze_valueprinter(self.m.locations)


        policygrid = policyX.create_policy_grid(self.m.locations, self.discountfactor, self.m.endstates)
        print("\033[1m {}\033[00m".format("Policygrid (nog wel wat bugs in)"))
        self.maze_valueprinter(policygrid)

        self.m.actions =  policyX.select_action(spawn_location, self.m.locations, self.discountfactor)
        print("When spawn_location is", spawn_location, "Next Step:", self.m.get_actions())

#every location with his start values with the template: [value, rewards, next_value]
mazeX_locations =   [[[0, -1, 0], [0, -1, 0], [0, -1, 0], [0, 40, 0]],
                    [[0, -1, 0], [0, -1, 0], [0, -10, 0], [0, -10, 0]],
                    [[0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]],
                    [[0, 10, 0], [0, -2, 0], [0, -1, 0], [0, -1, 0]]]

#every endstate, which means if it's in there the game is finished, which means it always has no value/next_value
endstates = [[3, 0], [0, 3]]

agent_spawn_location = [0, 0] #the spawn place to test out the policyX.select_action function

mazeX = Maze(mazeX_locations, None, endstates)
policyX = Policy()

agentX = Agent(mazeX, policyX)
agentX.value_iteration(agent_spawn_location) #runs the value iteration functions (aka the main)
