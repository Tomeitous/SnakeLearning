# Snake Game Reinforcement Learning

This project is an implementation of reinforcement learning algorithm to play the snake game. The snake game environment is implemented using Pygame library and the reinforcement learning algorithm is implemented using PyTorch. The agent learns to play the game by maximizing the reward signal.

## Requirements
- Pygame
- PyTorch
- Matplotlib
- Numpy

## File Structure
- SnakeEnv.py : Implements the environment of the snake game
- DQNAgent.py : Implements the reinforcement learning algorithm
- main.py : The entry point of the program

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
