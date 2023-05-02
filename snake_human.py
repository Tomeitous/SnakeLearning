import pygame
import random
import sys
import time

# Initialize Pygame
pygame.init()

# Set up the game window
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Snake Game")

font = pygame.font.SysFont("comicsansms", 35)
clock = pygame.time.Clock()

def create_obstacle(snake_x, snake_y):
    min_distance = 80
    obstacle_width = random.choice([20, 40, 60])
    obstacle_height = random.choice([20, 40, 60])
    obstacle_x = random.randint(0, (screen_width - obstacle_width) // snake_speed) * snake_speed
    obstacle_y = random.randint(0, (screen_height - obstacle_height) // snake_speed) * snake_speed
    while abs(snake_x - obstacle_x) < min_distance and abs(snake_y - obstacle_y) < min_distance:
        obstacle_x = random.randint(0, (screen_width - obstacle_width) // snake_speed) * snake_speed
        obstacle_y = random.randint(0, (screen_height - obstacle_height) // snake_speed) * snake_speed

    return pygame.Rect(obstacle_x, obstacle_y, obstacle_width, obstacle_height)


restart = True
while restart:
    # Set up the game variables
    snake_color = (0, 255, 0)
    snake_width = 20
    snake_height = 20
    snake_x = screen_width // 2
    snake_y = screen_height // 2
    snake_speed = 50
    snake_list = []
    snake_length = 1
    food_color = (255, 0, 0)
    food_width = 20
    food_height = 20
    food_x = random.randint(0, (screen_width - food_width) // snake_speed) * snake_speed
    food_y = random.randint(0, (screen_height - food_height) // snake_speed) * snake_speed
    score = 0
    food_eaten = 0
    obstacles = [create_obstacle(snake_x, snake_y)]
    obstacle_color = (0, 0, 255)

    # Initialize the snake's velocity
    dx = snake_speed
    dy = 0


    # Game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Handle player input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and dx != snake_speed:
            dx = -snake_speed
            dy = 0
        if keys[pygame.K_RIGHT] and dx != -snake_speed:
            dx = snake_speed
            dy = 0
        if keys[pygame.K_UP] and dy != snake_speed:
            dx = 0
            dy = -snake_speed
        if keys[pygame.K_DOWN] and dy != -snake_speed:
            dx = 0
            dy = snake_speed

        # Update the snake's position
        snake_x += dx
        snake_y += dy

        # Check for collisions with the walls
        if snake_x < 0 or snake_x + snake_width > screen_width or snake_y < 0 or snake_y + snake_height > screen_height:
            running = False

        # Check for collisions with the food
        if snake_x < food_x + food_width and snake_x + snake_width > food_x and snake_y < food_y + food_height and \
                snake_y + snake_height > food_y:
            # Increase the score and snake length
            score += 10
            snake_length += 1

            # Move the food to a random location
            food_x = random.randint(0, (screen_width - food_width) // snake_speed) * snake_speed
            food_y = random.randint(0, (screen_height - food_height) // snake_speed) * snake_speed

            # If the score is a multiple of 100, add a new obstacle
            if score % 100 == 0:
                obstacles.append(create_obstacle(snake_x, snake_y))
                obstacles.append(create_obstacle(snake_x, snake_y))
                obstacles.append(create_obstacle(snake_x, snake_y))
                obstacles.append(create_obstacle(snake_x, snake_y))
                obstacles.append(create_obstacle(snake_x, snake_y))

            # If an obstacle doesn't exist, create one
            if not obstacle:
                obstacle = create_obstacle(snake_x, snake_y)

            # Update the snake's position
        snake_head = pygame.Rect(snake_x, snake_y, snake_width, snake_height)
        snake_list.append(snake_head)
        if len(snake_list) > snake_length:
            del snake_list[0]

        # Check for collisions with the snake's body only if the snake is longer than one segment
        if snake_length > 1:
            front_wall_x = snake_x + dx
            front_wall_y = snake_y + dy

            for segment in snake_list[:-1]:
                # Check if the front wall of the snake head collides with the segment
                if front_wall_x == segment.x and front_wall_y == segment.y:
                    running = False

            # Check for collisions with the obstacle(s)
            for obstacle in obstacles:
                if snake_head.colliderect(obstacle):
                    running = False

        # Clear the screen
        screen.fill((0, 0, 0))

        # Draw the snake and food
        for segment in snake_list:
            pygame.draw.rect(screen, snake_color, segment)
        pygame.draw.rect(screen, food_color, (food_x, food_y, food_width, food_height))


        # Draw the obstacle
        for obstacle in obstacles:
            pygame.draw.rect(screen, obstacle_color, obstacle)

        # Draw the score
        score_text = font.render("Score: " + str(score), True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        # Update the display
        pygame.display.update()

        # Control the game speed
        clock.tick(10)

    # Game over
    game_over_text = font.render("Game Over", True, (255, 255, 255))
    screen.blit(game_over_text, (screen_width // 2 - 75, screen_height // 2 - 20))
    pygame.display.update()
