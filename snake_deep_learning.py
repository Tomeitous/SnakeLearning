import pygame
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()

# Set up the game window
screen_width = 200
screen_height = 200
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Snake Game")

font = pygame.font.SysFont("comicsansms", 35)
clock = pygame.time.Clock()


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.fc(x)


class SnakeEnv:
    def __init__(self):
        self.snake_color = (0, 255, 0)
        self.snake_width = 20
        self.snake_height = 20
        self.snake_speed = 20
        self.food_color = (255, 0, 0)
        self.food_width = 20
        self.food_height = 20
        self.action_space = [0, 1, 2, 3]  # left, right, up, down
        self.state_size = 12  # snake_x, snake_y, food_x, food_y, direction


    def reset(self):
        self.snake_x = screen_width // 2
        self.snake_y = screen_height // 2
        self.snake_list = []
        self.snake_length = 1
        self.food_x = random.randint(0, (screen_width - self.food_width) // self.snake_speed) * self.snake_speed
        self.food_y = random.randint(0, (screen_height - self.food_height) // self.snake_speed) * self.snake_speed
        self.score = 0
        self.dx = self.snake_speed
        self.dy = 0
        self.direction = 1  # right
        state = self.get_state()
        return state

    def get_state(self):
        snake_x_rel_food_x = (self.snake_x - self.food_x) / screen_width
        snake_y_rel_food_y = (self.snake_y - self.food_y) / screen_height

        food_left = int(self.snake_x > self.food_x)
        food_right = int(self.snake_x < self.food_x)
        food_up = int(self.snake_y > self.food_y)
        food_down = int(self.snake_y < self.food_y)

        direction_x = self.dx / self.snake_speed
        direction_y = self.dy / self.snake_speed

        tail_left = 0
        tail_right = 0
        tail_up = 0
        tail_down = 0
        if len(self.snake_list) > 1:
            tail_left = int(self.snake_x > self.snake_list[-2].x)
            tail_right = int(self.snake_x < self.snake_list[-2].x)
            tail_up = int(self.snake_y > self.snake_list[-2].y)
            tail_down = int(self.snake_y < self.snake_list[-2].y)

        state = np.array(
            [snake_x_rel_food_x, snake_y_rel_food_y, food_left, food_right, food_up, food_down, direction_x,
             direction_y, tail_left, tail_right, tail_up, tail_down])

        return state

    def new_food_position(self):
        while True:
            food_x = random.randint(0, (screen_width - self.food_width) // self.snake_speed) * self.snake_speed
            food_y = random.randint(0, (screen_height - self.food_height) // self.snake_speed) * self.snake_speed
            food_rect = pygame.Rect(food_x, food_y, self.food_width, self.food_height)

            if not any(food_rect.colliderect(segment) for segment in self.snake_list):
                return food_x, food_y

    def step(self, action):

        food_left = int(self.snake_x > self.food_x)
        food_right = int(self.snake_x < self.food_x)
        food_up = int(self.snake_y > self.food_y)
        food_down = int(self.snake_y < self.food_y)

        tail_left = 0
        tail_right = 0
        tail_up = 0
        tail_down = 0
        if len(self.snake_list) > 1:
            tail_left = int(self.snake_x > self.snake_list[-2].x)
            tail_right = int(self.snake_x < self.snake_list[-2].x)
            tail_up = int(self.snake_y > self.snake_list[-2].y)
            tail_down = int(self.snake_y < self.snake_list[-2].y)

        # Update direction
        if action == 0 and self.dx != self.snake_speed:
            self.dx = -self.snake_speed
            self.dy = 0
            self.direction = 0  # left
        if action == 1 and self.dx != -self.snake_speed:
            self.dx = self.snake_speed
            self.dy = 0
            self.direction = 1  # right
        if action == 2 and self.dy != self.snake_speed:
            self.dx = 0
            self.dy = -self.snake_speed
            self.direction = 2  # up
        if action == 3 and self.dy != -self.snake_speed:
            self.dx = 0
            self.dy = self.snake_speed
            self.direction = 3  # down

        # Update snake position
        self.snake_x += self.dx
        self.snake_y += self.dy

        # Check for collisions with the walls
        if self.snake_x < 0 or self.snake_x + self.snake_width > screen_width or \
                self.snake_y < 0 or self.snake_y + self.snake_height > screen_height:
            done = True
            reward = -1
        else:
            done = False
            reward = 0

        # Check for collisions with the food
            if self.snake_x < self.food_x + self.food_width and \
                    self.snake_x + self.snake_width > self.food_x and \
                    self.snake_y < self.food_y + self.food_height and \
                    self.snake_y + self.snake_height > self.food_y:

                self.score += 10
                self.snake_length += 1
                reward = 10

                self.food_x, self.food_y = self.new_food_position()

            else:
                reward = -1 if done else 0.5 * (int(food_left and self.dx < 0) + int(food_right and self.dx > 0) + int(
                    food_up and self.dy < 0) + int(food_down and self.dy > 0)) - 2 * (
                                                     int(tail_left and self.dx < 0) + int(
                                                 tail_right and self.dx > 0) + int(tail_up and self.dy < 0) + int(
                                                 tail_down and self.dy > 0))

        # Update the snake's position
        snake_head = pygame.Rect(self.snake_x, self.snake_y, self.snake_width, self.snake_height)
        self.snake_list.append(snake_head)
        if len(self.snake_list) > self.snake_length:
            del self.snake_list[0]

        # Check for collisions with the snake's body only if the snake is longer than one segment
        if self.snake_length > 1:
            front_wall_x = self.snake_x + self.dx
            front_wall_y = self.snake_y + self.dy

            for segment in self.snake_list[:-1]:
                # Check if the front wall of the snake head collides with the segment
                if front_wall_x == segment.x and front_wall_y == segment.y:
                    done = True
                    reward = -1

        state = self.get_state()
        return state, reward, done

    def render(self):
        screen.fill((0, 0, 0))

        for segment in self.snake_list:
            pygame.draw.rect(screen, self.snake_color, segment)
        pygame.draw.rect(screen, self.food_color, (self.food_x, self.food_y, self.food_width, self.food_height))
        pygame.draw.rect(screen, self.food_color, (self.food_x, self.food_y, self.food_width, self.food_height))
        pygame.draw.rect(screen, self.food_color, (self.food_x, self.food_y, self.food_width, self.food_height))

        score_text = font.render("Score: " + str(self.score), True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        pygame.display.update()

    def close(self):
        pygame.quit()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class DQNAgent:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = DQN(input_size, output_size)
        self.target_model = DQN(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters())
        self.memory = deque(maxlen=20000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_frequency = 1000
        self.loss_fn = nn.MSELoss()
        self.training_step = 0

    def memorize(self, state, action, next_state, reward, done):
        self.memory.append(Transition(state, action, next_state, reward, done))


    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.output_size))
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.model(state)
                return q_values.argmax().item()


    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.training_step += 1
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.int64).unsqueeze(-1)
        reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32)
        done_batch = torch.tensor(np.array(batch.done), dtype=torch.bool)

        q_values = self.model(state_batch).gather(1, action_batch)
        next_q_values = self.target_model(next_state_batch).max(1)[0].detach()
        next_q_values[done_batch] = 0.0
        expected_q_values = reward_batch + self.gamma * next_q_values

        loss = self.loss_fn(q_values, expected_q_values.unsqueeze(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.training_step % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())


    def save(self, path):
        torch.save(self.model.state_dict(), path)


    def load(self, path):
        self.model.load_state_dict(torch.load(path))


def train(agent, env, episodes, render=False):
    scores = []
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if render:
                env.render()
                pygame.time.delay(10)
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.memorize(state, action, next_state, reward, done)
            agent.learn()
            state = next_state
            score += reward
        scores.append(score)
        print(f'Episode: {episode}, Score: {score}, Epsilon: {agent.epsilon:.4f}')

    return scores


if __name__ == '__main__':
    env = SnakeEnv()
    agent = DQNAgent(env.state_size, len(env.action_space))
    scores = train(agent, env, 2000, render=True)
    env.close()

    # Plot the scores
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()
