#This file to keep the variables
#Probably this is what we will use to radio transmit
test_file_name = "test1_filtered.txt"
''' D* '''
#Factor of traveling one unit
traveling_factor = 10
#Factor for ascending
ascending_slope_factor = 400
#Factor for descending
descending_slope_factor = 10
#size of cell goes inside path function, which returns [(90.0, 'R'), 10.0]
size_of_cell = 14
#point we are starting from, in the matrix
start_cell = (0,0) #row 1 col 0
#point we are trying to get to
goal_cell = (13,13) #row 1 col 3
#Where we are facing...   Currently +y axis direction
initial_angle = 90
#max slope we want to bear
max_slope = 0.0005
#instead of saying "turn by 190 degrees", say "turn by 170 degrees"
max_rotation_angle = 180
#instructions file __name__
instructions_file = "instructions.txt"

'''DEM GENERATOR'''
nx = 18 #number of rows
ny = 18 #number of columns
sizex = 1 #factor for row compression
sizey = 1 #factor for col compression
show_dem = True
test = "test2_filtered.txt"

rover_size = 0.05 #0.254 #0.3556

dem_x_dim_size = nx * rover_size
dem_y_dim_size = ny * rover_size

'''LIDAR'''
#make this TRUE when we are reading in from LIDAR
lidar = True
#crop the area
crop_out = 5 #197 inches == 5 meters
lidar_input_is_text_file = True
lidar_filename = "test1_filtered.txt"

num_cells_in_dem = int(crop_out/rover_size) + 1

''' how much we are compressing by'''
#if we have 30 rows, 34 columns
#then to bind by 2 in row and col
#we divide

number_of_rows = nx
number_of_cols = ny
x_bind_factor = 1
y_bind_factor = 1
#number of rows with bind
m = number_of_rows/x_bind_factor
#number of cols with bind
n = number_of_cols/y_bind_factor
max_angle = 10 #maximum rotation angle
