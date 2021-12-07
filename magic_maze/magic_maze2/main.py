class Maze:
    def __init__(self, locations, endstates):
        """
        :param locations: every location of the grid with the template: [value, rewards, next_value]
        :param actions: saves all the used actions
        :param endstates: a list with every endstate coordinates
        """
        self.locations = locations
        self.endstates = endstates

    def print_locations(self):
        """
        Prints the maze lists in a fashionable style
        :param locations: every location of the grid with the template: [value, rewards]
        """
        for row in self.locations:
            print_line = ""
            for column in row:
                print_line += ("\033[34m {}\033[33m".format("|") + "\033[91m {}\033[00m" .format(column[0]))
            print_line += "\033[34m {}\033[33m".format("|")
            print(print_line)
        print(" ")

    def set_locations(self, new_locations):
        self.locations = new_locations

    def get_locations(self):
        return self.locations

    def get_endstates(self):
        return self.endstates

class Policy:
    def __init__(self, policygrid=[]):
        self.policygrid = policygrid

    def create_policy_grid(self, grid_locations, discountfactor, endstates):
        """
        Creates a grid with arrows pointing towards the point it needs to go, while adding a X to every endstate
        :param grid_locations: every location of the grid with the template: [value, rewards, next_value]
        :param discountfactor: the discountfactor used to calculate the best next value
        :param endstates: a list with every endstate coordinates
        :return: the grid with every arrow towards the next possible value aka the policy grid
        """
        return policygrid

class Agent:
    def __init__(self, m: Maze, pol: Policy):
        """
        :param m: The maze class
        :param pol: The policy class
        """
        self.m = m
        self.pol = pol



