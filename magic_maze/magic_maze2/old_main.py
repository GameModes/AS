import copy
import statistics

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

    def get_max_values(self):
        max_X = len(self.locations[0])
        max_Y = len(self.locations[1])

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
        # possible_directions = ["←", "→", "↓", "↑"]
        neighbours = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        policygrid = copy.deepcopy(grid_locations)
        for row in range(len(grid_locations)):
            for column in range(len(grid_locations)):
                if [row, column] in endstates:
                    policygrid[row][column] = "X"
                else:
                    # neighbour_coordinates = [np.add([row, column], neighbour).clip(min=0, max=3).tolist() for neighbour in
                    #                          neighbours]
                    # neighbour_values = []
                    # for neighbour in neighbour_coordinates:
                    #     if neighbour == [row, column]:
                    #         neighbour_values.append(0)
                    #     else:
                    #         neighbour_values.append(grid_locations[neighbour[0]][neighbour[1]][0] + discountfactor * grid_locations[neighbour[0]][neighbour[1]][1])

                    neighbour_values = self.get_neighbour_values([row, column], grid_locations, discountfactor, False)
                    # print(neighbour_values)
                    highest_index_states = [i for i, x in enumerate(neighbour_values) if x == max(neighbour_values)]
                    policygrid[row][column] = possible_directions[highest_index_states[0]]
                    if len(highest_index_states) > 1:
                        for x in highest_index_states[1:]:
                            policygrid[row][column] = policygrid[row][column] + str(possible_directions[x])
        return policygrid

    def get_neighbour_values(self, location, grid_locations, discountfactor, use_self_coordinates=True):
        neighbourclosevalues = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        neighbour_coordinates = [np.add(location, neighbour).clip(min=0, max=3).tolist() for neighbour in
                                 neighbourclosevalues]
        neighbour_values = []
        for neighbour in neighbour_coordinates:
            if use_self_coordinates is False and neighbour == location:
                neighbour_values.append(0)
            else:
                #TODO kijk naar Index
                neighbour_values.append(grid_locations[neighbour[0]][neighbour[1]][0] + discountfactor * grid_locations[neighbour[0]][neighbour[1]][1])
                # neighbour_values.append(grid_locations[neighbour[0]][neighbour[1]][0] + discountfactor * grid_locations[neighbour[0]][neighbour[1]][1])

        # neighbour_values = [grid_locations[neighbour[1]][neighbour[0]][0] + discountfactor *
        #                     grid_locations[neighbour[1]][neighbour[0]][1] for neighbour in neighbour_coordinates]
        # print(neighbour_values)
        return neighbour_values

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

        neighbour_values = self.get_neighbour_values(location, grid_locations, discountfactor)

        highest_states_index = [i for i, x in enumerate(neighbour_values) if x == max(neighbour_values)]
        if len(highest_states_index) > 1: #more then 1 highest state
            next_location = neighbour_coordinates[random.choice(highest_states_index)] #random
        else:
            next_location = neighbour_coordinates[highest_states_index[0]]
        return next_location

    def create_episode(self, start_position, endstates):
        position = start_position.copy()
        episode = [position]
        while position not in endstates:
            moving_x = random.randint(-1, 1)
            moving_y = random.randint(-1, 1)
            position = np.add(position, [moving_x, moving_y]).clip(min=0, max=3).tolist()
            episode.append(position)
        return episode

    def monte_carlo(self, grid_locations, start_position, endstates, discountfactor,):
        run_times = 10000

        for x in range(run_times):
            episode = self.create_episode(start_position, endstates)
            episode.reverse()
            G = 0
            for position_index in range(len(episode)):
                position = episode[position_index]
                x_position = position[0]
                y_position = position[1]
                #                                  Get the X     Get the Y    get current_value
                G = discountfactor*G+(grid_locations[x_position][y_position] [1] )
                if position not in episode[0:position_index] and position not in endstates:
                    grid_locations[x_position][y_position][3].append(G)
                    grid_locations[x_position][y_position][0] = round(statistics.mean(grid_locations[x_position][y_position][3]))
            # print(grid_locations)
        return grid_locations

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

    def move(self, current_place, next_place):
        next_place_coordinates = np.add(current_place, next_place).clip(min=0, max=3).tolist()
        return next_place_coordinates


    def state_definer(self, current_coordinates):
        """
        retrieves every neighbours scores (value used in formula) in a list (unless they are endstates which returns 0)
        :param current_coordinates: the current coordinates that the neighbours needs to be calculated form
        :return: every score of the neighbours in a list
        """
        neighbourclosevalues = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        neighbours_scores = []
        for neighbour in neighbourclosevalues:
            neighbour_coordinates = self.move(current_coordinates, neighbour)

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
        print("\033[1m {}\033[00m".format("Policygrid "))
        self.maze_valueprinter(policygrid)

        self.m.actions =  policyX.select_action(spawn_location, self.m.locations, self.discountfactor)
        print("When spawn_location is", spawn_location, "Next Step:", self.m.get_actions())

#every location with his start values with the template: [value, rewards, next_value, G_list]
mazeX_locations =   [[[0, -1, 0, []], [0, -1, 0, []], [0, -1, 0, []], [0, 40, 0, []]],
                    [[0, -1, 0, []], [0, -1, 0, []], [0, -10, 0, []], [0, -10, 0, []]],
                    [[0, -1, 0, []], [0, -1, 0, []], [0, -1, 0, []], [0, -1, 0, []]],
                    [[0, 10, 0, []], [0, -2, 0, []], [0, -1, 0, []], [0, -1, 0, []]]]

#every endstate, which means if it's in there the game is finished, which means it always has no value/next_value
endstates = [[3, 0], [0, 3]]

agent_spawn_location = [2, 0] #the spawn place to test out the policyX.select_action function

mazeX = Maze(mazeX_locations, None, endstates)
policyX = Policy()

agentX = Agent(mazeX, policyX)
# agentX.value_iteration(agent_spawn_location) #runs the value iteration functions (aka the main)
agentX.maze_valueprinter(mazeX_locations)
mazeX_locations = policyX.monte_carlo(mazeX_locations, agent_spawn_location, endstates, discountfactor=1)
agentX.maze_valueprinter(mazeX_locations)