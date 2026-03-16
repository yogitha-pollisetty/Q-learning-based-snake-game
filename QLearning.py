
import SnakeEnv as snake_env  # Importing the Snake game environment
import Agent as agent_class  # Importing the Agent class that will perform Q-learning
import numpy as np  # For numerical computations, especially arrays
import random  # For random actions or exploration
import copy  # For deep copying parameter dictionaries
import csv  # For CSV file handling, likely for saving the Q-table
import matplotlib.pyplot as plt  # For visualizing results with plots

def q_learning(agent, env, max_steps, train=True):
    """ Runs Q-learning for one episode. """
    state = env.reset()  # Reset the environment and get the initial state
    agent.init_state(state)  # Initialize agent's state

    # Initialize counters for tracking performance metrics
    total_return, n_apples, n_stops, n_goodsteps = 0.0, 0, 0, 0
    done = False  # Flag indicating whether the episode has finished

    for i in range(max_steps):
        # Select action based on whether training or evaluation mode
        action = agent.select_action(state) if train else agent.select_greedy(state)
        next_state, reward, done, _ = env.step(action)  # Take the action and observe the outcome

        if train:
            # Update the Q-table based on the observed state and reward
            agent.update_Qtable(state, action, reward, next_state)

        state = next_state  # Move to the next state
        total_return += pow(agent.gamma, i) * reward  # Calculate discounted total return

        # Track specific performance metrics based on rewards
        if reward == 10:  # Apple eaten
            n_apples += 1
        elif reward > 0:  # Moving closer to the apple
            n_goodsteps += 1  
        elif reward == -100:  # Snake hit a wall or itself
            n_stops += 1

    # Return various performance metrics after the episode
    return total_return, n_apples, n_stops, n_goodsteps, agent.num_states_visited()

def run_ql(max_runs, max_steps, in_params, qtable_file, display=False, train=False):
    """ Runs Q-learning multiple times and saves the best Q-table. """
    num_runs = max_runs
    results_list = []  # To store results of each run
    best_return = float('-inf')  # Best return observed across runs
    best_qtable = None  # Best Q-table to be saved

    # Displaying Q-learning parameters
    print("\n Running Q-Learning with Parameters:")
    print(f"   Gamma (Discount Factor)  = {in_params['gamma']}")
    print(f"    Alpha (Learning Rate)    = {in_params['alpha']}")
    print(f"    Epsilon (Exploration)    = {in_params['epsilon']}")
    print(f"    Min Epsilon              = {in_params['epsilon_min']}")
    print(f"    Epsilon Decay Rate       = {in_params['epsilon_decay']}")
    print(f"    Num Runs                 = {num_runs}")
    print(f"    Num Steps per Run        = {max_steps}\n")

    for run in range(num_runs):
        # Create deep copy of parameters to prevent modification across runs
        params = copy.deepcopy(in_params)  

        env = snake_env.SnakeEnv()  # Create the Snake environment
        agent = agent_class.Agent(env, params)  # Initialize the agent with the environment and parameters
        env.display = display  # Optionally display the game state

        if not train and qtable_file is not None:
            # If evaluating, load the Q-table from the file
            agent.read_qtable(qtable_file)

        # Run Q-learning for one episode and collect performance metrics
        ret = q_learning(agent, env, max_steps, train=train)
        results_list.append(ret)  # Append the result tuple to the list

        env.close()  # Close the environment after the run
        # Print the statistics for the current run
        print(f"* Run {run}: Return={ret[0]:.3f}, #Apples={ret[1]}, #Stops={ret[2]}, #GoodSteps={ret[3]}, #UniqueStatesVisited={ret[4]}")

        if train and ret[0] > best_return:
            # If training and current return is better than best return, update best Q-table
            best_return = ret[0]
            best_qtable = agent.Q

    # After all runs, compute and print mean statistics
    mean_results = np.mean(results_list, axis=0) if results_list else None
    if mean_results is not None:
        print("\n Mean Statistics Across Runs")
        print(f"   ➤ Mean Return: {mean_results[0]:.3f}, Apples: {mean_results[1]:.1f}, Stops: {mean_results[2]:.1f}, Good Steps: {mean_results[3]:.1f}, Unique States: {mean_results[4]:.1f}")

    if train:
        # If training, save the best Q-table
        agent.Q = best_qtable  
        agent.write_qtable(qtable_file)
        print(f"\n Training complete! Q-table saved as {qtable_file}")

    # Return both raw and mean results from the runs
    return results_list, mean_results

def plot_results(results):
    """
    Generates line plots for different performance metrics based on epsilon values.
    """
    epsilons = results[:, 0]  # Extract epsilon values from the first column of the results

    # Create subplots to display multiple performance metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot mean return vs epsilon
    axes[0, 0].plot(epsilons, results[:, 1], marker='o', linestyle='-', label="Mean Return")
    axes[0, 0].set_title("Mean Return vs Epsilon")
    axes[0, 0].set_xlabel("Epsilon")
    axes[0, 0].set_ylabel("Mean Return")

    # Plot mean apples eaten vs epsilon
    axes[0, 1].plot(epsilons, results[:, 2], marker='s', linestyle='-', label="Mean Apples Eaten")
    axes[0, 1].set_title("Mean Apples Eaten vs Epsilon")
    axes[0, 1].set_xlabel("Epsilon")
    axes[0, 1].set_ylabel("Mean Apples")

    # Plot mean stops vs epsilon
    axes[1, 0].plot(epsilons, results[:, 3], marker='^', linestyle='-', label="Mean Stops")
    axes[1, 0].set_title("Mean Stops vs Epsilon")
    axes[1, 0].set_xlabel("Epsilon")
    axes[1, 0].set_ylabel("Mean Stops")

    # Plot mean unique states visited vs epsilon
    axes[1, 1].plot(epsilons, results[:, 4], marker='x', linestyle='-', label="Mean Unique States Visited")
    axes[1, 1].set_title("Mean Unique States vs Epsilon")
    axes[1, 1].set_xlabel("Epsilon")
    axes[1, 1].set_ylabel("Mean Unique States")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def epsilon_experiment(base_params, qtable_file, num_runs=1, num_steps=1000):
    """
    Run Q-learning experiments with different epsilon values and visualize the impact.
    """
    # List of epsilon values to experiment with for different levels of exploration
    epsilon_values = [0.1, 0.3, 0.5, 0.7, 0.9]  
    results = []  # To store the results for each epsilon value

    print("\n Running Epsilon Experiment...\n")

    for epsilon in epsilon_values:
        # Modify the epsilon value in the parameters for each experiment
        exp_params = base_params.copy()
        exp_params['epsilon'] = epsilon  

        print(f"\n Running with Epsilon = {epsilon}\n")
        # Run Q-learning and collect the mean results for this epsilon
        _, mean_results = run_ql(num_runs, num_steps, exp_params, qtable_file, display=False, train=False)

        # Store the results for plotting
        if mean_results is not None:
            results.append((epsilon, *mean_results))

    # Convert the results list to a numpy array for easier handling in the plot function
    results_array = np.array(results)

    # Plot the results
    plot_results(results_array)

## Train & Save the Q-table
num_runs = 1  # Number of runs
num_steps = 1000  # Number of steps per run

# Set up the Q-learning parameters
params = {
    'gamma': 0.95,  # Discount factor for future rewards
    'alpha': 0.7,  # Learning rate
    'epsilon': 0.6,  # Exploration rate (probability of random action)
    'epsilon_min': 0.01,  # Minimum epsilon value for exploration
    'epsilon_decay': 0.995  # Decay rate for epsilon after each episode
}

qtable_file = "qtable_2025.csv"  # Q-table file to save/load

# Call run_ql() for either training or evaluation
results_list = run_ql(num_runs, num_steps, params, qtable_file, display = True, train = False) # evaluation
#results_list = run_ql(num_runs, num_steps, params, qtable_file, display = False, train = True) # training

# Run the epsilon experiment to analyze the effect of different epsilon values on performance
epsilon_experiment(params, qtable_file, num_runs, num_steps)
