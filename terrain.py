# Terrain generator
from random import choice
from time import sleep


# Class containg world attr.
class Attr(object):
    def __init__(self):
        self.TILES = '#*~o'
        self.WORLD_SIZE = 1000
        self.Y_LENGTH = 1000
        self.X_LENGTH = 1000


# Generator class
class Generator(object):
    def __init__(self):
        self.world = []
        self.attr = Attr()

    # Create an empty row
    def create_row(self):
        for _ in range(self.attr.X_LENGTH):
            self.world.append(choice(self.attr.TILES))


# Main generator class
class MainGen(object):
    def __init__(self):
        self.gen = Generator()
        self.attr = Attr()

    # Create the world
    def create_world(self):
        for _ in range(self.attr.WORLD_SIZE):
            self.gen.create_row()

    # Render the world
    def render_world(self):
        for tile in self.gen.world:
            print tile
            sleep(0.05)



# Main game class
class Game(object):
    def __init__(self):
        self.main_gen = MainGen()

    # Run the functions
    def run(self):
        self.main_gen.create_world()
        self.main_gen.render_world()


# Start the program
if __name__ == "__main__":
    game = Game()
    game.run()
