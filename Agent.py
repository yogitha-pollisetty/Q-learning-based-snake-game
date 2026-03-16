
import random
import numpy as np
import csv

class Agent:
    """ 
    An AI agent which controls the snake's movements using Q-learning.
    """
    
    def __init__(self, env, params):
        """
        Initializes the agent with environment and parameters for Q-learning.
        
        Arguments:
        env -- The game environment (SnakeEnv)
        params -- Dictionary containing hyperparameters like gamma, alpha, epsilon, etc.
        """
        self.env = env
        self.action_space = env.action_space  # Number of actions in the environment (4 for Snake Game)
        self.state_space = env.state_space    # Number of state features (12 features for Snake Game)
        self.gamma = params['gamma']          # Discount factor for future rewards
        self.alpha = params['alpha']          # Learning rate
        self.epsilon = params['epsilon']      # Exploration rate
        self.epsilon_min = params['epsilon_min']  # Minimum exploration rate
        self.epsilon_decay = params['epsilon_decay']  # Rate of decay for exploration
        
        # Initialize Q-table as a dictionary to hold state-action pairs and their Q-values
        self.Q = {}

    @staticmethod
    def state_to_str(state_list):
        """ 
        Convert a binary list state representation to a string. 
        
        Arguments:
        state_list -- List representing the state of the game
        
        Returns:
        A string representing the state in binary format.
        """
        return "".join(str(x) for x in state_list)

    def init_state(self, state):
        """ 
        Initialize a state in the Q-table if it does not exist already.
        
        Arguments:
        state -- Current state of the agent
        
        This function ensures that every new state has an associated entry in the Q-table.
        """
        state_key = self.state_to_str(state)
        if state_key not in self.Q:
            self.Q[state_key] = [0.0] * self.action_space  # Initialize Q-values to 0 for all actions

    def select_action(self, state):
        """ 
        Epsilon-greedy action selection. Chooses a random action with probability epsilon, 
        otherwise selects the action with the highest Q-value.
        
        Arguments:
        state -- Current state of the agent
        
        Returns:
        action -- Selected action (integer)
        """
        state_key = self.state_to_str(state)
        if state_key not in self.Q:
            self.Q[state_key] = [0.0] * self.action_space  # Initialize if state is not found

        # Exploration vs. Exploitation: Random action with probability epsilon, otherwise pick the best Q-action
        if random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space)  # Exploration: random action
        else:
            return np.argmax(self.Q[state_key])  # Exploitation: action with max Q-value

    def select_greedy(self, state):
        """ 
        Greedy action selection: Always selects the action with the highest Q-value.
        
        Arguments:
        state -- Current state of the agent
        
        Returns:
        action -- Selected action (integer)
        """
        state_key = self.state_to_str(state)
        if state_key not in self.Q:
            self.Q[state_key] = [0.0] * self.action_space  # Initialize if state is not found
        return np.argmax(self.Q[state_key])  # Return action with the highest Q-value

    def update_Qtable(self, state, action, reward, next_state):
        """ 
        Update the Q-value for the given state-action pair based on reward and next state.
        
        Arguments:
        state -- Current state of the agent
        action -- Action taken by the agent
        reward -- Reward received after taking the action
        next_state -- Next state after performing the action
        
        The Q-value is updated using the Q-learning formula:
        Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a Q(s', a) - Q(s, a))
        """
        state_key = self.state_to_str(state)
        next_state_key = self.state_to_str(next_state)

        if state_key not in self.Q:
            self.Q[state_key] = [0.0] * self.action_space
        if next_state_key not in self.Q:
            self.Q[next_state_key] = [0.0] * self.action_space

        # Find the max Q-value for the next state
        max_next_q = max(self.Q[next_state_key])  
        
        # Store the old Q-value for debugging (optional)
        old_value = self.Q[state_key][action]
        
        # Update the Q-value using the Q-learning equation
        self.Q[state_key][action] += self.alpha * (reward + self.gamma * max_next_q - self.Q[state_key][action])

        # Adjust epsilon after each update to reduce exploration over time
        self.adjust_epsilon()

    def num_states_visited(self):
        """ 
        Returns the number of unique states visited during training.
        
        This is useful for tracking how many different states have been encountered.
        """
        return len(self.Q)

    def write_qtable(self, filepath):
        """ 
        Write the content of the Q-table to an output CSV file with integer states.
        
        Arguments:
        filepath -- The path where the Q-table should be saved
        
        This function stores each state as an integer (converted from binary) 
        along with its action-value pairs.
        """
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            for state, actions in sorted(self.Q.items()):  # Sort states for consistency
                state_int = int(state, 2)  # Convert binary string to integer
                for action, q_value in enumerate(actions):
                    writer.writerow([state_int, action, q_value])  # Store state-action Q-value

    def read_qtable(self, filepath):
        """ 
        Read a Q-table from a CSV file and populate the agent's Q-table.
        
        Arguments:
        filepath -- The path to the CSV file containing the Q-table
        
        This function resets the Q-table and loads values from the file.
        """
        self.Q = {}  # Reset the Q-table before loading new data
        try:
            with open(filepath, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    state_int, action, q_value = int(row[0]), int(row[1]), float(row[2])
                    state_bin = format(state_int, '012b')  # Convert integer back to 12-bit binary string
                    if state_bin not in self.Q:
                        self.Q[state_bin] = [0.0] * self.action_space
                    self.Q[state_bin][action] = q_value
        except FileNotFoundError:
            print(f"Warning: Q-table file '{filepath}' not found. Starting fresh.")

    def adjust_epsilon(self):
        """ 
        Adjust the epsilon value over time by applying epsilon decay.
        
        The epsilon value is gradually reduced to minimize exploration as the agent learns.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Reduce epsilon over time
