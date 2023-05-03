

## Other written documents 

[Course Reflection](https://github.com/Tomeitous/SnakeLearning/blob/main/Course_reflection.md)

[Reinforcement Learning Explanation](https://github.com/Tomeitous/SnakeLearning/blob/main/Reinforcement_Learning_explanation.md)

# Snake Game Reinforcement Learning

This project is an implementation of reinforcement learning algorithm to play the snake game. The snake game environment is implemented using Pygame library and the reinforcement learning algorithm is implemented using PyTorch. The agent learns to play the game by maximizing the reward signal.

![image](https://user-images.githubusercontent.com/22236987/235773176-1fef537e-c9c7-40d1-a22b-2fad63c60478.png) ![image](https://user-images.githubusercontent.com/22236987/235774906-6c640e06-605e-4a2f-81d4-6326fe4ffc33.png)


## Requirements
- Pygame
- PyTorch
- Matplotlib
- Numpy

## File Structure
- SnakeEnv : Implements the environment of the snake game
- DQNAgent : Implements the reinforcement learning algorithm

## How to run
1. Clone the repository
2. Install the required libraries
3. Run the main.py file

## Code Explanation
### SnakeEnv
- The SnakeEnv class implements the environment of the snake game.
- The state of the environment is represented by the positions of the snake and the food.
- The action space consists of four possible actions: move up, move down, move left, and move right.
- The step function takes an action as input and returns the next state, reward, and done signal.

### DQNAgent
- The DQNAgent class implements the reinforcement learning algorithm.
- The model is a feedforward neural network that takes the state as input and outputs the Q-values for each action.
- The memory is a deque that stores the transitions experienced by the agent.
- The act function selects an action to take based on the current state and the current value of epsilon.
- The learn function updates the model using the experiences stored in the memory.
- The save and load functions save and load the model's parameters, respectively.

### Main
- The main function trains the agent and runs the game.
- The train function trains the agent for a specified number of episodes and returns the scores.
- The scores are plotted using Matplotlib.

## Conclusion
This project demonstrates how reinforcement learning can be applied to play simple games. The same approach can be extended to more complex games and other applications.


# Snake Game Documentation

## Import statements
This code uses several libraries, including:
- `pygame` for creating the game window and rendering graphics
- `random` for generating random food positions
- `sys` for system-specific parameters and functions
- `time` for timing functions
- `numpy` for numerical computations
- `torch` for building and training neural networks
- `torch.nn` for defining neural network models
- `torch.optim` for optimizing neural network models
- `collections.deque` for efficiently appending and popping elements from both ends of a list
- `collections.namedtuple` for creating named tuples
- `matplotlib.pyplot` for plotting data

## Pygame Initialization
The Pygame library is initialized using `pygame.init()`.

The game window is created with the following parameters:
- `screen_width`: 400 pixels
- `screen_height`: 400 pixels

The window is created using `screen = pygame.display.set_mode((screen_width, screen_height))`.
The caption for the window is set to "Snake Game" using `pygame.display.set_caption("Snake Game")`.

A font is created for the game using `font = pygame.font.SysFont("comicsansms", 35)`.
The clock is created for the game using `clock = pygame.time.Clock()`.

## DQN Model
The DQN model is a neural network that takes in an `input_size` and produces an `output_size`.
It is defined using the `nn.Module` class from the `torch.nn` library.
The model consists of a fully connected neural network with 3 hidden layers, each with 256 neurons and ReLU activation.
The input is passed through the fully connected neural network and the output is returned.

## SnakeEnv Class
The `SnakeEnv` class sets up the environment for the snake game.
It has the following attributes:
- `snake_color`: the color of the snake, which is green (0, 255, 0)
- `snake_width`: the width of the snake, which is 20 pixels
- `snake_height`: the height of the snake, which is 20 pixels
- `snake_speed`: the speed of the snake, which is 20 pixels
- `food_color`: the color of the food, which is red (255, 0, 0)
- `food_width`: the width of the food, which is 20 pixels
- `food_height`: the height of the food, which is 20 pixels
- `action_space`: a list of 4 possible actions the snake can take, [0, 1, 2, 3], representing left, right, up, down
- `state_size`: the size of the state, which is 12

The `reset` method initializes the starting position of the snake and food, sets the score to 0, and sets the direction to right.
It returns the initial state of the environment.

The `get_state` method returns the state of the environment.
The state consists of the following information:
- the x and y position of the snake relative to the food
- the direction of the food relative to the snake
- the direction of the tail relative to the snake

The `new_food_position` method generates a new position for the food.
The `step` method updates the state of the environment based on the action taken by the agent.
The action taken by the agent is passed as an argument.
The new position of the snake is calculated based on the current position and the direction.
If the snake hits the edge of the screen or its own body, the game is over and the method returns `done` as True.
Otherwise, the method returns `done` as False.

The `render` method updates the display with the current state of the environment.
It draws the snake and food on the screen and displays the score in the top left corner.

The `close` method closes the Pygame window and quits Pygame.

## DQNAgent Class
The `DQNAgent` class sets up the DQN agent for the snake game.
It has the following attributes:
- `input_size`: the size of the state
- `output_size`: the number of actions the agent can take
- `model`: the DQN model
- `target_model`: the target DQN model
- `optimizer`: the optimizer for the DQN model
- `memory`: a list of transitions (state, action, next_state, reward, done)
- `batch_size`: the size of the batch for training
- `gamma`: the discount factor
- `epsilon`: the exploration rate
- `epsilon_min`: the minimum exploration rate
- `epsilon_decay`: the rate at which the exploration rate decays
- `target_update_frequency`: the frequency with which the target model is updated
- `loss_fn`: the loss function for training the DQN model
- `training_step`: the number of steps the agent has taken

The `__init__` method initializes the DQNAgent with the given `input_size` and `output_size`.
It creates the DQN model and the target DQN model, the optimizer, and the memory.
It sets the batch size, discount factor, exploration rate, minimum exploration rate, and exploration rate decay.
It sets the target update frequency and the loss function.
It sets the training step to 0.

The `memorize` method adds a transition (state, action, next_state, reward, done) to the memory.

The `act` method returns the action the agent should take based on the current state.
If a random number is less than the exploration rate, the method returns a random action.
Otherwise, the method returns the action with the highest estimated value for the state.

The `learn` method trains the DQN model on a random batch of transitions from the memory.
If the size of the memory is less than the batch size, the method returns without training.
Otherwise, the method samples a batch of transitions from the memory, calculates the expected values for the actions taken, and updates the DQN model to minimize the loss.
It also updates the exploration rate and the target model.

The `save` method saves the DQN model to a file.

The `load` method loads the DQN model from a file.

## Main Function
The `train` function trains the agent on the environment.
It takes in the agent, the environment, the number of episodes to run, and an optional argument `render` to determine if the environment should be rendered during
