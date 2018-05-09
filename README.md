# Rover

Download pathfinder folder

dem_generator.py

''' Generates a DEM, converts to spherical and then to cartesian coordinates.
    Also takes in LIDAR input, reads into spherical coordinates and get a DEM from it.
    Outputs a cropped DEM, where distances further than 5 meters are eliminated.'''
    
elifs_dstar.py

''' Takes in a DEM and generates the best path depending on a max slope.
    Generates instructions for motors to move. '''
    
visualize.py

''' Plots DEM with path points, so we can see if the path is reasonable. '''
Execute ./jobs.sh

