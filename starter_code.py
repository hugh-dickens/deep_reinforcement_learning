# Import some modules from other libraries
import numpy as np
import torch
import time
import random
import collections
import cv2
from matplotlib import pyplot as plt

# Import the environment module
from environment import Environment


# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()


    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self):
        # Choose the next action.
        discrete_action = self._choose_next_action()
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition
    
        
#     # The class initialisation function for draw instead
#     def __init__(self, draw):
#         # Set the agent's environment.
#         self.draw = draw
#         # Create the agent's current state
#         self.state = None
#         # Create the agent's total reward for the current episode.
#         self.total_reward = None
#         # Reset the agent.
#         self.reset()

#     # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
#     # for draw
#     def reset(self):
#         # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
#         self.state = self.draw.reset()
#         # Set the agent's total reward for this episode to zero.
#         self.total_reward = 0.0

#     # Function to make the agent take one step in the environment.
#     def step(self, policy_indexed):
#         # Choose the next action.
#         discrete_action = self._choose_next_action_greedy(policy_indexed)
# #         discrete_action = self._choose_next_action()
#         # Convert the discrete action into a continuous action.
#         continuous_action = self._discrete_action_to_continuous(discrete_action)
#         # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
#         next_state, distance_to_goal = self.draw.step(self.state, continuous_action)
#         # Compute the reward for this paction.
#         reward = self._compute_reward(distance_to_goal)
#         # Create a transition tuple for this step.
#         transition = (self.state, discrete_action, reward, next_state)
#         # Set the agent's state for the next step, as the next state from this step
#         self.state = next_state
#         # Update the agent's reward for this episode
#         self.total_reward += reward
#         # Return the transition
#         return transition
    
    # Function for the agent to choose its next action
    def _choose_next_action(self):
        # Return discrete action 0
        # change this to a random discrete action from 0-3
        return np.random.choice((0,1,2,3))
    
    # Function for the agent to choose its next action
    def _choose_next_action_greedy(self, policy_indexed):
        col = int((self.state[0] - 0.05)*10)
        row = int((self.state[1]-0.05)*10)
        action = policy_indexed[col,row]
        # Return discrete action 0
        # change this to a random discrete action from 0-3
        return action

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            # Move right
            continuous_action = np.array([0.1, 0], dtype=np.float32)
        elif discrete_action == 1: 
            # Move up
            continuous_action = np.array([0, 0.1], dtype=np.float32)
        elif discrete_action == 2: # left
            # Move left
            continuous_action = np.array([-0.1, 0], dtype=np.float32)
        elif discrete_action == 3:
            # move down
            continuous_action = np.array([0, -0.1], dtype=np.float32)
        return continuous_action

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        return reward
    
class Agent_greedy:

            # The class initialisation function for draw instead
    def __init__(self, draw):
        # Set the agent's environment.
        self.draw = draw
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    # for draw
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.draw.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self, policy_indexed):
        # Choose the next action.
        discrete_action = self._choose_next_action_greedy(policy_indexed)
#         discrete_action = self._choose_next_action()
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.draw.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition
    
    # Function for the agent to choose its next action
    def _choose_next_action(self):
        # Return discrete action 0
        # change this to a random discrete action from 0-3
        return np.random.choice((0,1,2,3))
    
    # Function for the agent to choose its next action
    def _choose_next_action_greedy(self, policy_indexed):
        col = int((self.state[0] - 0.05)*10)
        row = int((self.state[1]-0.05)*10)
        action = policy_indexed[col,row]
        # Return discrete action 0
        # change this to a random discrete action from 0-3
        return action

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            # Move right
            continuous_action = np.array([0.1, 0], dtype=np.float32)
        elif discrete_action == 1: 
            # Move up
            continuous_action = np.array([0, 0.1], dtype=np.float32)
        elif discrete_action == 2: # left
            # Move left
            continuous_action = np.array([-0.1, 0], dtype=np.float32)
        elif discrete_action == 3:
            # move down
            continuous_action = np.array([0, -0.1], dtype=np.float32)
        return continuous_action

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        return reward
    
    
    
class QValueVisualiser:

    def __init__(self, environment, magnification=500):
        self.environment = environment
        self.magnification = magnification
        self.half_cell_length = 0.05 * self.magnification
        # Create the initial q values image
        self.q_values_image = np.zeros([int(self.magnification), int(self.magnification), 3], dtype=np.uint8)

    def draw_q_values(self, q_values):
        # Create an empty image
        self.q_values_image.fill(0)
        # Loop over the grid cells and actions, and draw each q value
        for col in range(10):
            for row in range(10):
                # Find the q value ranges for this state
                max_q_value = np.max(q_values[col, row])
                min_q_value = np.min(q_values[col, row])
                q_value_range = max_q_value - min_q_value
                # Draw the q values for this state
                for action in range(4):
                    # Normalise the q value with respect to the minimum and maximum q values
                    q_value_norm = (q_values[col, row, action] - min_q_value) / q_value_range
                    # Draw this q value
                    x = (col / 10.0) + 0.05
                    y = (row / 10.0) + 0.05
                    self._draw_q_value(x, y, action, float(q_value_norm))
        # Draw the grid cells
        self._draw_grid_cells()
        # Show the image
        cv2.imwrite('q_values_image.png', self.q_values_image)
        cv2.imshow("Q Values", self.q_values_image)
        cv2.waitKey(0)

    def _draw_q_value(self, x, y, action, q_value_norm):
        # First, convert state space to image space for the "up-down" axis, because the world space origin is the bottom left, whereas the image space origin is the top left
        y = 1 - y
        # Compute the image coordinates of the centre of the triangle for this action
        centre_x = x * self.magnification
        centre_y = y * self.magnification
        # Compute the colour for this q value
        colour_r = int((1 - q_value_norm) * 255)
        colour_g = int(q_value_norm * 255)
        colour_b = 0
        colour = (colour_b, colour_g, colour_r)
        # Depending on the particular action, the triangle representing the action will be drawn in a different position on the image
        if action == 0:  # Move right
            point_1_x = centre_x + self.half_cell_length
            point_1_y = centre_y + self.half_cell_length
            point_2_x = point_1_x
            point_2_y = centre_y - self.half_cell_length
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        elif action == 1:  # Move up
            point_1_x = centre_x + self.half_cell_length
            point_1_y = centre_y - self.half_cell_length
            point_2_x = centre_x - self.half_cell_length
            point_2_y = point_1_y
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        elif action == 2:  # Move left
            point_1_x = centre_x - self.half_cell_length
            point_1_y = centre_y - self.half_cell_length
            point_2_x = point_1_x
            point_2_y = centre_y + self.half_cell_length
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        elif action == 3:  # Move down
            point_1_x = centre_x - self.half_cell_length
            point_1_y = centre_y + self.half_cell_length
            point_2_x = centre_x + self.half_cell_length
            point_2_y = point_1_y
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    def _draw_grid_cells(self):
        # Draw the state cell borders
        for col in range(11):
            point_1 = (int((col / 10.0) * self.magnification), 0)
            point_2 = (int((col / 10.0) * self.magnification), int(self.magnification))
            cv2.line(self.q_values_image, point_1, point_2, (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
        for row in range(11):
            point_1 = (0, int((row / 10.0) * self.magnification))
            point_2 = (int(self.magnification), int((row / 10.0) * self.magnification))
            cv2.line(self.q_values_image, point_1, point_2, (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.Q_values = np.zeros([10, 10, 4])
        self.greedy_policy_indexed = np.zeros([10,10])

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, minibatch):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(minibatch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, minibatch):
#         pass
        # TODO
        # NOTE: when just training on a single example on each iteration, the NumPy array (and Torch tensor) still needs to have two dimensions: the mini-batch dimension, and the data dimension. And in this case, the mini-batch dimension would be 1, instead of 5. This can be done by using the torch.unsqueeze() function.
#         state_tensor = torch.tensor(minibatch[0],dtype=torch.float32).unsqueeze(0)
#         print(state_tensor)
        ## index the first and 3 tuple from the whole minibatch i.e. state and reward
        minibatch_states =  [a_tuple[0] for a_tuple in minibatch]
        minibatch_actions =  [b_tuple[1] for b_tuple in minibatch]
        minibatch_rewards =  [c_tuple[2] for c_tuple in minibatch]
        minibatch_states_tensor = torch.tensor(minibatch_states,dtype=torch.float32)
        minibatch_actions_tensor = torch.tensor(minibatch_actions, dtype = torch.long)
        minibatch_rewards_tensor = torch.tensor(minibatch_rewards,dtype=torch.float32)
#         print(minibatch_states_tensor)
#         print(minibatch_actions_tensor.unsqueeze(-1))
#         print(minibatch_rewards_tensor)
        # always assumes youre passing a mini batch into a NN using torch so we need to 
        #package the data up as if it were a mini-batch, in this case it is size 1 so we
        # have to create that with [0,...]- index first element in mini-batch
#         predicted_q_value_tensor = self.q_network.forward(state_tensor)[0,transition[1]]
        predicted_q_value_tensor = self.q_network.forward(minibatch_states_tensor).gather(dim=1,index= minibatch_actions_tensor.unsqueeze(-1)).squeeze(-1)
        loss= torch.nn.MSELoss()(predicted_q_value_tensor,minibatch_rewards_tensor)
        return loss
    
    def q_val_calculator(self):
        ## would probably be easier if this was a dictionary, nightmare keeping track of
        ## columns and rows
        for col in range(10):
            for row in range(10):
                for action_idx in range(4):
                    state = np.array([(col/10)+0.05,(row/10)+0.05])
                    state_tensor = torch.tensor(state,dtype=torch.float32).unsqueeze(0)
                    predicted_q_value_tensor = self.q_network.forward(state_tensor)[0,action_idx]

                    self.Q_values[col,row][action_idx] = predicted_q_value_tensor

                self.greedy_policy_indexed[col,row] = np.argmax(self.Q_values[col,row])

        return self.Q_values, self.greedy_policy_indexed

class ReplayBuffer:
    
    # The class initialisation function.
    def __init__(self):
        self.buffer = collections.deque(maxlen=5000)
    
    def add_transition(self, transition):
        #add a transition to the replay buffer
        self.buffer.append(transition)
        
        return self.buffer
    
    def sample_minibatch(self, minibatch_size):
        #Samples a total of elements equal to minibatch_size from buffer
        #if buffer contains enough elements. Otherwise return all elements
        if len(self.buffer) > minibatch_size:
            minibatch = random.sample(self.buffer, minibatch_size)
        else:
            minibatch = None
        return minibatch
        

class Draw:
    
    # The class initialisation function.
    def __init__(self, display, magnification=500):
        # Set whether the environment should be displayed after every step
        self.display = display
        # Set the magnification factor of the display
        self.magnification = magnification
        # Set the initial state of the agent
        self.init_state = np.array([0.35, 0.15], dtype=np.float32)
        # Set the initial state of the goal
        self.goal_state = np.array([0.35, 0.85], dtype=np.float32)
        # Set the space which the obstacle occupies
        self.obstacle_space = np.array([[0.0, 0.7], [0.4, 0.5]], dtype=np.float32)
        # Create an image which will be used to display the environment
        self.image = np.zeros([int(self.magnification), int(self.magnification), 3], dtype=np.uint8)
        self.state_list = []
       
    # Function to reset the environment, which is done at the start of each episode
    def reset(self):
        return self.init_state

    # Function to execute an agent's step within this environment, returning the next state and the distance to the goal
    def step(self, state, action):
        # Determine what the new state would be if the agent could move there
        next_state = state + action
#         prev_state = state
        # If this state is outside the environment's perimeters, then the agent stays still
        if next_state[0] < 0.0 or next_state[0] > 1.0 or next_state[1] < 0.0 or next_state[1] > 1.0:
            next_state = state
        # If this state is inside the obstacle, then the agent stays still
        if self.obstacle_space[0, 0] <= next_state[0] < self.obstacle_space[0, 1] and self.obstacle_space[1, 0] <= next_state[1] < self.obstacle_space[1, 1]:
            next_state = state
        # Compute the distance to the goal
        distance_to_goal = np.linalg.norm(next_state - self.goal_state)
        self.state_list.append(next_state)
#         print(len(self.state_list))
        # Draw and show the environment, if required
        if len(self.state_list) == 20:
#             self.draw_trace(next_state, prev_state)
            self.draw_trace(self.state_list)
        # Return the next state and the distance to the goal
        return next_state, distance_to_goal

        
#     def draw_trace(self,agent_state, prev_agent_state):
    def draw_trace(self,state_list):
        agent.reset()
        # Create the background image
        window_top_left = (0, 0)
        window_bottom_right = (self.magnification * 1, self.magnification * 1)
        cv2.rectangle(self.image, window_top_left, window_bottom_right, (246, 238, 229), thickness=cv2.FILLED)
        # Draw the obstacle
        obstacle_left = int(self.magnification * self.obstacle_space[0, 0])
        obstacle_top = int(self.magnification * (1 - self.obstacle_space[1, 1]))
        obstacle_width = int(self.magnification * (self.obstacle_space[0, 1] - self.obstacle_space[0, 0]))
        obstacle_height = int(self.magnification * (self.obstacle_space[1, 1] - self.obstacle_space[1, 0]))
        obstacle_top_left = (obstacle_left, obstacle_top)
        obstacle_bottom_right = (obstacle_left + obstacle_width, obstacle_top + obstacle_height)
        cv2.rectangle(self.image, obstacle_top_left, obstacle_bottom_right, (0, 0, 150), thickness=cv2.FILLED)
        # Create the border
        border_top_left = (0, 0)
        border_bottom_right = (self.magnification * 1, self.magnification * 1)
        cv2.rectangle(self.image, border_top_left, border_bottom_right, (0, 0, 0), thickness=int(self.magnification * 0.02))
        # Draw the agent
        agent_centre = (int(self.init_state[0] * self.magnification), int((1 - self.init_state[1]) * self.magnification))
        agent_radius = int(0.02 * self.magnification)
        agent_colour = (100, 199, 246)
        cv2.circle(self.image, agent_centre, agent_radius, agent_colour, cv2.FILLED)
        # Draw the goal
        goal_centre = (int(self.goal_state[0] * self.magnification), int((1 - self.goal_state[1]) * self.magnification))
        goal_radius = int(0.02 * self.magnification)
        goal_colour = (227, 158, 71)
        cv2.circle(self.image, goal_centre, goal_radius, goal_colour, cv2.FILLED)
        # Draw the trace
        state_list.insert(0, self.init_state)
#         print(state_list[0])
        for i in range(20):
            cv2.line(self.image,(int(state_list[i][0]*self.magnification),int((1-state_list[i][1])*self.magnification)),(int(state_list[i+1][0]*self.magnification),int((1-state_list[i+1][1])*self.magnification)),(0,255,0),2)
        # Show the image
        cv2.imshow("Environment", self.image)
        # This line is necessary to give time for the image to be rendered on the screen
        cv2.waitKey(0)


    
# Main entry point
if __name__ == "__main__":

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    
#     ### put display false to speed up training
    environment = Environment(display=False, magnification=500)
    # Create an agent
    agent = Agent(environment)
    # Create a DQN (Deep Q-Network)
    dqn = DQN()
    # Create lists to store the losses and epochs
    losses = []
    episodes = []
    minibatch_size = 100
    replay_buffer = ReplayBuffer()
    # Create a graph which will show the loss as a function of the number of episodes
    fig, ax = plt.subplots()
    ax.set(xlabel='episodes', ylabel='Loss', title='Loss Curve for DQN network')

    # Loop over episodes
#     while True:
    for number_episodes in range(100):
        # Reset the environment for the start of the episode.
        agent.reset()
        loss_temp = []
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(20):
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step()
            replay_buffer.add_transition(transition)
#             print(transition)
            minibatch = replay_buffer.sample_minibatch(minibatch_size)
#             print(transition)
            if minibatch is not None:
#                 print(minibatch)
                loss = dqn.train_q_network(minibatch)
#             loss = dqn.train_q_network(transition)
                # add the loss for each step to a list
                loss_temp.append(loss)
            # Store this loss in the list
        loss_ave = sum(loss_temp) / 20
        ### add the average loss for each episode
        losses.append(loss_ave)
                    # Update the list of iterations
        episodes.append(number_episodes)
                    # Plot and save the loss vs iterations graph
        ax.plot(episodes, losses, color='blue')
       
            # Sleep, so that you can observe the agent moving. Note: this line should be removed when you want to speed up training
#         time.sleep(0.2)
    plt.show()
    fig.savefig("loss_vs_iterations.png")

    ### This prints the q values
#     dqn = DQN() important not to recreated the DQN network so that it is the same as the one above
    q_values, greedy_policy = dqn.q_val_calculator()
    print('Q_values are:', q_values)
#     # Create a visualiser
    visualiser = QValueVisualiser(environment=environment, magnification=500)
#     # Draw the image
    visualiser.draw_q_values(q_values)
    print('The greedy policy indexed is:',greedy_policy)
    # Create a DQN (Deep Q-Network)

#### It doesn't stor greedy_policy from above so I need to come up with a way of combining these two parts together!!!!!!
    ### Need to change above function for draw!!!!
    draw = Draw(display=True, magnification=500)
#     Create an agent
    agent = Agent_greedy(draw)
    agent.reset()
        # Loop over steps within this episode. The episode length here is 20.
    for step_num in range(20):
            # Step the agent once, and get the transition tuple for this step
        transition = agent.step(greedy_policy)
        

            