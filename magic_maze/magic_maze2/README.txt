How to run:
    in the main function bottom of the code in magic_maze2\main.py you see the algorithms
    if you delete the hashtag you can use it.
    Also in the main there are most of the initiatives


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
    - value_iteration{algorithm}: if the world is known then execute an algoritm to update every value of locations
    - monte_carlo{algorithm}: if the world is unknown then create paths/episodes and go back the path to update the values
    - td_learning{algorithm}: using monte_carlo predict the other locations and compare the values with the true value to update the prediction algoritm
    - policy_monte_carlo{algorithm}
    - sarsa{algorithm}:
    - sarsa_max{algorithm}:

control_carlo algorithm:
    Initialize
        - empty_policy = [[0,0,0],[0,0,0],[0,0,0]]
        - create probablity (for every location) with the template: [direction, probability]  = {[UP, 0.25], [Left, 0.25], [Down, 0.25], [Right, 0.25]]}
        - create Qtable (for every location) with the template: [direction, G] = {[UP, 0], [Left, 0], [Down, 0], [Right, 0]]}

    Update
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



