import numpy as np

class Maze:
    def __init__(self, locations, actions, endstates):
        self.locations = locations
        self.actions = actions
        self.endstates = endstates

    def step(self, state):
        pass


class Policy:
    def __init__(self):
        pass

    def select_action(self):
        pass




class Agent:
    def __init__(self, m: Maze, pol: Policy):
        self.m = m
        self.pol = pol

    def action(self):
        pass

    def value(self):
        pass

    def state_definer(self, current_coordinates):
        neighbourclosevalues = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        neighbours_scores = []
        for neighbour in neighbourclosevalues:
            neighbour_coordinates = np.add(current_coordinates, neighbour).clip(min=0, max=3).tolist()
            neighbour_values = self.m.locations[neighbour_coordinates[1]][neighbour_coordinates[0]]
            if current_coordinates in self.m.endstates:
                neighbours_scores.append(0)
            else:
                neighbours_scores.append(neighbour_values[1]+self.discountfactor*neighbour_values[0])
            # print("state: " + str(self.m.locations[neighbour_stats[1]][neighbour_stats[0]]))
        # print(neighbours_scores)
        return max(neighbours_scores)

    def maze_valueprinter(self):
        for row in self.m.locations:
            print_line = ""
            for column in row:
                print_line += ("\033[34m {}\033[33m".format("|") + "\033[91m {}\033[00m" .format(column[0]))
            print_line += "\033[34m {}\033[33m".format("|")
            print(print_line)
        print(" ")

    def delta_definer(self):
        delta = 0
        for row in self.m.locations:
            for column in row:
                if abs(column[2] - column[0]) > delta:
                    delta = abs(column[2] - column[0])
        return delta

    def value_iteration(self, k=8):
        self.discountfactor = 1
        self.maze_valueprinter()
        k=0
        delta = 0.1
        while delta >= 0.1:
            k=k+1
            print("\033[1m {}\033[00m".format("K=" + str(k)))
            for row in range(len(self.m.locations)):
                for column in range(len(self.m.locations[row])):
                    self.m.locations[row][column][2] = self.state_definer([column, row])
            delta = self.delta_definer()
            for row in range(len(self.m.locations)):
                for column in range(len(self.m.locations[row])):
                    self.m.locations[row][column][0] = self.m.locations[row][column][2]

            self.maze_valueprinter()

mazeX_locations =   [[[0, -1, 0], [0, -1, 0], [0, -1, 0], [0, 40, 0]],
                    [[0, -1, 0], [0, -1, 0], [0, -10, 0], [0, -10, 0]],
                    [[0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]],
                    [[0, 10, 0], [0, -2, 0], [0, -1, 0], [0, -1, 0]]]

endstates = [[3, 0], [0, 3]]

mazeX = Maze(mazeX_locations, None, endstates)
policyX = Policy()

agentX = Agent(mazeX, policyX)
agentX.value_iteration()