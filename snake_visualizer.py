import pygame
import numpy as np

class SnakeVisualizer:
    def __init__(self, width, height, cell_size=20):
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Snake Game DQN")
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
    def draw_board(self, board):
        self.screen.fill(self.BLACK)
        
        # Draw the board
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                 self.cell_size, self.cell_size)
                
                # Get the color from the board state
                color = board[y, x]
                if np.any(color > 0):  # If there's a snake or apple
                    if color[0] > 0:  # Snake body
                        pygame.draw.rect(self.screen, self.GREEN, rect)
                    elif color[1] > 0:  # Apple
                        pygame.draw.rect(self.screen, self.RED, rect)
                    elif color[2] > 0:  # Snake head
                        pygame.draw.rect(self.screen, self.BLUE, rect)
                else:  # Empty cell
                    pygame.draw.rect(self.screen, self.WHITE, rect, 1)
        
        pygame.display.flip()
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        return True 