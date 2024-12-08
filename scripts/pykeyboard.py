import pygame
import numpy as np

class KeyboardInput:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Car Control Input")
        self.clock = pygame.time.Clock()
        self.running = True

        # Action states
        self.actions = np.zeros(4)  # Assuming 4 motor actions
        # self.action_step = 5.0  # Increment for actions

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        # Reset actions
        self.actions = np.zeros(4)

        # Handle key presses
        keys = pygame.key.get_pressed()

        # Forward (increase all motors)
        if keys[pygame.K_UP]:
            self.actions[0] = 1

        # Backward (decrease all motors)
        if keys[pygame.K_DOWN]:
            self.actions[1] = 1

        # Turn left (increase right motors, decrease left motors)
        if keys[pygame.K_LEFT]:
            self.actions[2] = 1

        if keys[pygame.K_RIGHT]:
            self.actions[3] = 1

        


    def get_actions(self):
        return self.actions

    def run(self):
        while self.running:
            self.handle_events()
            self.screen.fill((0, 0, 0))  # Black background

            # Display controls
            font = pygame.font.Font(None, 36)
            text = font.render(f"Actions: {self.actions}", True, (255, 255, 255))
            self.screen.blit(text, (10, 10))

            pygame.display.flip()
            self.clock.tick(30)  # Limit to 30 FPS

        self.quit()

    def quit(self):
        pygame.quit()

# Usage example
if __name__ == "__main__":
    keyboard_input = KeyboardInput()
    keyboard_input.run()
