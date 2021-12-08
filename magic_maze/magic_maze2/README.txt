Code Structure:

- Class Maze:
    - init: save the maze locations & endstates
    - print_locations: prints the maze locations in a fashionable way
    - get_locations: get the maze locations
    - set_locations: set the maze locations
    - get_endstates: get the maze endstates
    - set_endstates: set the maze endstates

- Class Policy:
    - init: save the policygrid (empty beginning)
    - create_policygrid: create policygrid from grid_locations and save it

- Class Agent:
    - init: save the Policy class and maze class
    - select_action: with a location and the maze locations get the next step/action with looking at neighbours
    - delta_definer: compare the current value and the next value to calculate the delta for the next K
    - value_iteration{algoritm}: if the world is known then execute an algoritm to update every value of locations
    - monte_carlo{algoritm}: if the world is unknown then create paths/episodes and go back the path to update the values
    - td_learning{algoritm}: using monte_carlo predict the other locations and compare the values with the true value to update the prediction algoritm
    - policy_monte_carlo{algoritm}
    - sarsa{algoritm}:
    - sarsa_max{algoritm}:
    - double_q_learning{algoritm}: