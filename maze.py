import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Define the grid world environment
grid_size = 5
goal_state = (4, 4)

# Define obstacles in the grid
obstacles = [(1, 1), (2,1), (3, 1), (4, 3), ()]

actions = ['up', 'down', 'left', 'right']
num_actions = len(actions)

# Parameters for Q-learning
alpha = 0.1      # learning rate
gamma = 0.9      # discount factor
epsilon = 0.2    # exploration rate
num_episodes = 1000

# Initialize the Q-table
q_table = np.zeros((grid_size * grid_size, num_actions))

# Helper functions
def state_to_index(state):
    return state[0] * grid_size + state[1]

def index_to_state(index):
    return divmod(index, grid_size)

def get_next_state(state, action):
    x, y = state
    if action == 'up' and x > 0: x -= 1
    elif action == 'down' and x < grid_size - 1: x += 1
    elif action == 'left' and y > 0: y -= 1
    elif action == 'right' and y < grid_size - 1: y += 1
    return (x, y)

def get_reward(state):
    if state == goal_state:
        return 1
    elif state in obstacles:
        return -1
    else:
        return -0.01  # Small penalty for each move to encourage shorter paths

# Function to draw the grid using matplotlib
def draw_grid(state, path=[]):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(np.arange(0, grid_size, 1))
    ax.set_yticks(np.arange(0, grid_size, 1))
    ax.grid(color='black')

    # Plot obstacles
    for (x, y) in obstacles:
        ax.add_patch(Rectangle((y, grid_size - x - 1), 1, 1, color='black'))
    
    # Plot goal
    ax.add_patch(Rectangle((goal_state[1], grid_size - goal_state[0] - 1), 1, 1, color='green', label="Goal"))
    
    # Plot agent's path
    for (x, y) in path:
        ax.add_patch(Rectangle((y, grid_size - x - 1), 1, 1, color='blue', alpha=0.3))

    # Plot agent's current position
    ax.add_patch(Rectangle((state[1], grid_size - state[0] - 1), 1, 1, color='red', label="Agent"))

    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.pause(0.5)  # Pause to visualize the steps
    plt.clf()  # Clear the figure for the next step

# Q-learning algorithm
for episode in range(num_episodes):
    state = (0, 0)  # Start position
    done = False

    while not done:
        state_idx = state_to_index(state)
        
        # Choose an action using epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            action_idx = random.choice(range(num_actions))  # Explore
        else:
            action_idx = np.argmax(q_table[state_idx])  # Exploit
        
        action = actions[action_idx]
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        next_state_idx = state_to_index(next_state)
        
        # Q-value update
        best_next_action = np.argmax(q_table[next_state_idx])
        q_table[state_idx, action_idx] += alpha * (
            reward + gamma * q_table[next_state_idx, best_next_action] - q_table[state_idx, action_idx]
        )
        
        state = next_state
        
        if state == goal_state or state in obstacles:
            done = True

# Test the learned policy
state = (0, 0)
path = [state]
print("Agent's Path to Goal:\n")

while state != goal_state:
    # Draw the current grid with agent's position and path
    draw_grid(state, path)

    # Get action based on the trained Q-table
    state_idx = state_to_index(state)
    action_idx = np.argmax(q_table[state_idx])
    action = actions[action_idx]
    state = get_next_state(state, action)
    path.append(state)

# Final grid showing the agent at the goal
draw_grid(state, path)
print("Learned path to goal:", path)

plt.show()  # Keep the last plot open
