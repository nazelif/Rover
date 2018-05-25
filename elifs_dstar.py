import sys
import csv
import math
from pandas import DataFrame #to pretty print matrices
import numpy as np
import variables as v

# A point class to keep track of coordinates
class Point(object):
	def __init__(self, x, y, z):
		self.X = x
		self.Y = y
		self.Z = z
	def getX():
		return self.X
	def getY():
		return self.Y
	def getZ():
		return self.Z
	def distance(self, other):
		dx = self.X - other.X
		dy = self.Y - other.Y
		dz = self.Z - other.Z
		return math.sqrt(dx**2 + dy**2 + dz**2)
	def __repr__(self):
		return "".join(["Point(", str(self.X), ",", str(self.Y), ",", str(self.Z), ")"])

def testPoint(x=0,y=0,z=0):
    p1 = Point(3, 4,5)
    p2 = Point(3,0,5)
    return Point.distance(p1,p2)

def read_csv_into_matrix(filename): #Ex. elevation_matrix.csv
	#we read in a csv
	matrix = list(csv.reader(open(filename)))
	#we want to make it a point matrix???

def slope(p1,p2): #takes a point class
	try:
		slope =  ((p2.Z - p1.Z) / math.sqrt( (p2.X-p1.X)**2 + (p2.Y-p1.Y)**2))
		if slope < 0: #we are ascending, factor = 20
			slope *= v.ascending_slope_factor
		else: #we are descending, descend factor = 10
			slope *= v.descending_slope_factor
		return math.fabs(slope)
	except ZeroDivisionError:
		print "Same point, Different elevation, check points"
		print(p1)
		print(p2)
		return -1

def cost(p1,p2):
	units = Point.distance(p1,p2)
	cost = v.traveling_factor * units
	if (p1.getX != p2.getX) ^ (p1.getY != p2.getY): #diagonally moving when x and y are both  different values
		cost *= math.sqrt(2)
	slope_factor = v.ascending_slope_factor
	if slope(p1,p2) < 0:
		slope_factor = v.descending_slope_factor
	return cost + math.fabs(slope(p1,p2)) * slope_factor

def test_functions():
	#print "distance = %s"%(testPoint())
	'''
	p1 = Point(3,1,4)
	p2 = Point(3,0,5)
	p3 = Point(2,1,2)
	print "cost = %s"%(cost(p1,p2))
	print "cost = %s"%(cost(p1,p3))
	print "cost = %s"%(cost(p2,p3))
	print "slope = %s"%(slope(p1,p2))
	print "slope = %s"%(slope(p1,p3))
	print "slope = %s"%(slope(p2,p3))
	'''

def fill_cost1(dem,cost1, start_cell, goal_cell):
	#now fill 10s for h/v moves, 14 for diagonal moves
	cost1[goal_cell] = 0
	g_row = goal_cell[0]
	g_col = goal_cell[1]

	#filing cols to left of goal
	for x in range(cost1.shape[1]): #shape[1] is num columns
		if g_col-x >= 1:
			cost1[g_row, g_col-x-1] = cost1[g_row, g_col-x] + 10
		if g_col-x<1: #until we hit 0 with g_col-x-1 = 0 i.e. until g_col-x = 1
			break


	#filling cols to right of goal
	for x in range(cost1.shape[1]): #shape[1] is num columns
		#until we hit 0 with g_col-x+1 = shape[1]-1 i.e. g_col-x = s[1]-2
		if g_col+x+1 <= cost1.shape[1]-1:
			cost1[g_row, g_col+x+1] = cost1[g_row, g_col+x] + 10
		if g_col+x+1 > cost1.shape[1]-1:
			break

	#filling rows up
	for x in range(cost1.shape[0]): #shape[0] is num rows
		if g_row-x >= 1:
			cost1[g_row-x-1, g_col] = cost1[g_row-x, g_col] + 10
		if g_row-x<1: #until we hit 0 with g_col-x-1 = 0 i.e. until g_col-x = 1
			break


	#filling rows down
	for x in range(cost1.shape[0]): #shape[0] is num rows
		if g_row+x+1 <= cost1.shape[0]-1:
			cost1[g_row+x+1, g_col] = cost1[g_row+x, g_col] + 10
		if g_row+x+1 > cost1.shape[0]-1:
			break

	#fill diagonals
	if g_row != 0:
		if g_col != 0:
			cost1[g_row-1, g_col-1] = 14
		if g_col != cost1.shape[1]-1:
			cost1[g_row-1, g_col+1] = 14

	if g_row != cost1.shape[0]-1:
		if g_col != 0 :
			cost1[g_row+1, g_col-1] = 14
		if g_col != cost1.shape[1]-1:
			cost1[g_row+1, g_col+1] = 14

	#1ST QUADRANT
	for row in range(g_row-1,-1,-1): #every row
		for col in range(g_col+1,cost1.shape[1],1): #every cell in that row

			if cost1[row][col] == -1:
				cost1[row,col] = min(cost1[row+1,col-1]+14,min(cost1[row,col-1]+10, cost1[row+1,col]+10))

	#2ND QUADRANT
	for row in range(g_row-1,-1,-1): #every row
		for col in range(g_col-1,-1,-1): #every cell in that row
			if cost1[row][col] == -1:
				cost1[row,col] = min(cost1[row+1,col+1]+14,min(cost1[row,col+1]+10, cost1[row+1,col]+10))

	#3RD QUADRANT
	for row in range(g_row+1,cost1.shape[0],1): #every row
		for col in range(g_col-1,-1,-1): #every cell in that row
			if cost1[row][col] == -1:
				cost1[row,col] = min(cost1[row-1,col+1]+14,min(cost1[row,col+1]+10, cost1[row-1,col]+10))

	#4TH QUADRANT
	for row in range(g_row+1,cost1.shape[0],1): #every row
		for col in range(g_col+1,cost1.shape[1],1): #every cell in that row
			if cost1[row][col] == -1:
				cost1[row,col] = min(cost1[row-1,col-1]+14,min(cost1[row,col-1]+10, cost1[row-1,col]+10))

def fill_cost2(dem,cost1, cost2, start_cell, goal_cell, max_slope):

	explored = [] #keep track of all visited nodes
	queue = [goal_cell] #keep track of nodes to be checked

	#Loop until q is empty
	while queue:
		node = queue.pop(0)
		print node
		if node not in explored:
			explored.append(node) #add it to the list of checked nodes
			if (node[0] == goal_cell[0] and node[1] == goal_cell[1]):
				cost2[goal_cell] = 0
			else:
				fill_cell(dem, cost2, goal_cell, node[0], node[1], max_slope)
			neighbors = get_nbrs(cost2, node[0], node[1]) #via the function defined below
			for neighbor in neighbors:
				queue.append(neighbor)

def get_nbrs(cost2, x, y):
	result = []
	for i in range(8):
		if i == 0:
			a = x
			b = y+1
		elif i == 1:
			a = x-1
			b = y+1
		elif i == 2:
			a = x-1
			b = y
		elif i == 3:
			a = x-1
			b = y-1
		elif i == 4:
			a = x
			b = y-1
		elif i == 5:
			a = x+1
			b = y-1
		elif i == 6:
			a = x+1
			b = y
		else:
			a = x+1
			b = y+1
		#check is a,b OOBs
		if (a >= 0 and b >= 0 and a < cost2.shape[0] and b < cost2.shape[1]):
			result.append([a,b])
	return result

def fill_cell(dem, cost2, goal_cell, x, y, max_slope):
	#the value at x will be the min of 8 values we will compute with the function

	temp = sys.maxsize #set this  to be the max val possible, then update as we see smaller ones
	for i in range(8):
		if i == 0:
			a = x
			b = y+1
		elif i == 1:
			a = x-1
			b = y+1
		elif i == 2:
			a = x-1
			b = y
		elif i == 3:
			a = x-1
			b = y-1
		elif i == 4:
			a = x
			b = y-1
		elif i == 5:
			a = x+1
			b = y-1
		elif i == 6:
			a = x+1
			b = y
		else:
			a = x+1
			b = y+1

		#check if [a,b] is NOT Out Of Boundries
		if (a >= 0 and b >= 0 and a < cost2.shape[0] and b < cost2.shape[1] and cost2[a,b] != -1):
			#print "Computations to fill " + str((x,y)) + " using the cell " + str((a,b)) + "\n"
			temp_val = cost2[a,b] #assuming that this value is the minimum it can be, and it won't change
			distance = 10 if math.fabs(a+b-x-y) == 1 else 14
			temp_val += distance
			slope = (dem[a,b] - dem[x,y])/ distance
			if (math.fabs(slope) > max_slope):
				slope = slope * 1000
			#print slope
			temp_val += math.fabs(slope) * v.ascending_slope_factor if slope > 0 else math.fabs(slope) * v.descending_slope_factor
			'''print i
			print cost2[a,b]
			print distance
			print slope
			print temp_val'''
			if temp_val < temp:
				temp = temp_val

	cost2[(x,y)] = temp # or min(temp, cost2[(x,y)])

def fill_cost3(dem,cost1, cost2, cost3, start_cell, goal_cell):

	explored = [] #keep track of all visited nodes
	queue = [goal_cell] #keep track of nodes to be checked

	while queue:
		node = queue.pop(0)
		if node not in explored:
			explored.append(node) #add it to the list of checked nodes
			print "\nCurrent node is " + str(node)

			if (node[0] == goal_cell[0] and node[1] == goal_cell[1]):
				cost3[goal_cell] = 0

			else:
				fill_cell_with_average(dem, cost2, cost3, goal_cell, node[0], node[1])

			''' The BFS traversal below works properly'''
			neighbors = get_nbrs(cost3, node[0], node[1]) #via the function defined below
			for neighbor in neighbors:
				queue.append(neighbor)

def fill_cell_with_average(dem, cost2, cost3, goal_cell, x, y):
	num = 1
	temp = cost2[x,y]
	print "Starting value for filling " + str([x,y]) + " is " + str(temp)
	for i in range(8):
		if i == 0:
			a = x
			b = y+1
		elif i == 1:
			a = x-1
			b = y+1
		elif i == 2:
			a = x-1
			b = y
		elif i == 3:
			a = x-1
			b = y-1
		elif i == 4:
			a = x
			b = y-1
		elif i == 5:
			a = x+1
			b = y-1
		elif i == 6:
			a = x+1
			b = y
		else:
			a = x+1
			b = y+1

		if (a >= 0 and b >= 0 and a < cost3.shape[0] and b < cost3.shape[1] and cost3[a,b] != -1):
			print "\nComputations to fill " + str((x,y)) + " using the cell " + str((a,b))
			print "cost2 " + str(cost2[a,b])
			print "cost3 " + str(cost3[a,b])
			temp += cost2[a,b]
			#temp += cost3[a,b]
			num += 1

	cost3[(x,y)] = temp/num # or min(temp, cost2[(x,y)])
	print "Result for " + str((x,y)) + " is " + str(cost3[(x,y)])

def output_path(cost1, start_cell, goal_cell, next_move, max_rotation_angle):
	#from the start point, trace a path towards the goal cell
	#follow the min pt
	cur_val = cost1[start_cell]
	cur_xy = start_cell
	cost_list = []

	next_move.append(cur_xy) #we start at the start cell
	cost_list.append(cur_val)

	temp_val = cur_val
	temp_xy = cur_xy

	slope_so_far = 0

	while cur_val != 0:
		if (cur_xy[1] != cost1.shape[1]-1) and (cur_xy[0] != cost1.shape[0]-1): #we are checking lower right diagonal
			if (cost1[cur_xy[0]+1, cur_xy[1]+1] <= temp_val):
			#	print "7"
				temp_val = cost1[cur_xy[0]+1 , cur_xy[1]+1]
				temp_xy = (cur_xy[0]+1,cur_xy[1]+1)

		if (cur_xy[1] != 0) and (cur_xy[0] != 0): #we are checking upper left diagonal
			if (cost1[cur_xy[0]-1, cur_xy[1]-1] <= temp_val):
			#	print "3"
				temp_val = cost1[cur_xy[0]-1 , cur_xy[1]-1]
				temp_xy = (cur_xy[0]-1, cur_xy[1]-1)

		if (cur_xy[1] != cost1.shape[1]-1) and (cur_xy[0] != 0): #we are checking upper right diagonal
			if (cost1[cur_xy[0]-1, cur_xy[1]+1] <= temp_val):
			#	print "1"
				temp_val = cost1[cur_xy[0]-1 , cur_xy[1]+1]
				temp_xy = (cur_xy[0]-1, cur_xy[1]+1)

		if (cur_xy[1] != 0) and (cur_xy[0] != cost1.shape[0]-1): #we are checking lower left diagonal
			if (cost1[cur_xy[0]+1, cur_xy[1]-1] <= temp_val):
			#	print "5"
				temp_val = cost1[cur_xy[0]+1 , cur_xy[1]-1]
				temp_xy = (cur_xy[0]+1, cur_xy[1]-1)

		if (cur_xy[0] != cost1.shape[0]-1): #we are checking down
			if (cost1[cur_xy[0]+1, cur_xy[1]] <= temp_val):
			#	print "6"
				temp_val = cost1[cur_xy[0]+1 , cur_xy[1]]
				temp_xy = (cur_xy[0]+1, cur_xy[1])

		if (cur_xy[0] != 0): #we are checking up
			if (cost1[cur_xy[0]-1, cur_xy[1]] <= temp_val):
			#	print "2"
				temp_val = cost1[cur_xy[0]-1 , cur_xy[1]]
				temp_xy = (cur_xy[0]-1, cur_xy[1])

		if (cur_xy[1] != cost1.shape[1]-1): #we are checking rhs
			if (cost1[cur_xy[0], cur_xy[1]+1] <= temp_val):
			#	print "4"
				temp_val = cost1[cur_xy[0] , cur_xy[1]+1]
				temp_xy = (cur_xy[0],cur_xy[1]+1)

		if (cur_xy[1] != 0): #we are checking lhs
			if (cost1[cur_xy[0], cur_xy[1]-1] <= temp_val):
			#	print "0"
				temp_val = cost1[cur_xy[0] , cur_xy[1]-1]
				temp_xy = (cur_xy[0],cur_xy[1]-1)

		cur_val = temp_val
		cur_xy = temp_xy
		next_move.append(cur_xy) #we start at the start cell
		cost_list.append(cur_val) #get the path


def output_path2(cost, start_cell, goal_cell, next_move, max_rotation_angle):
	tc = [[0 for c in range(cost.shape[1])] for r in range(cost.shape[0])]
	tc[goal_cell[0]][goal_cell[1]] = cost[goal_cell[0]][goal_cell[1]]
	m = start_cell[0]
	n = start_cell[1]
	for i in range(1, m+1):
		tc[i][goal_cell[1]] = tc[i-1][goal_cell[1]] + cost[i][goal_cell[1]]
	for j in range(1,n+1):
		tc[goal_cell[0]][j] = tc[goal_cell[0]][j-1] + cost[goal_cell[0]][j]
	for i in range(1, m+1):
		for j in range(1,n+1):
			tc[i][j] = min(tc[i-1][j-1], min(tc[i-1][j], tc[i][j-1] + cost[i][j]))
	#print tc
	return tc[m][n]

def polar(p): # a point w cartesian coordinates
	return (np.sqrt(p[0]**2 + p[1]**2) , np.arctan2(p[1],p[0]))

def angle(p1,p2): #(cur,next)
	#return np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / math.pi
	if (p2[0]<p1[0]):
		if (p2[1] < p1[1]):
			return 135
		elif (p2[1] == p1[1]):
			return 90
		else:
			return 45
	elif(p2[0] == p1[0]):
		if (p2[1] < p1[1]):
			return 180
		elif (p2[1] > p1[1]):
			return 0
	else:
		if (p2[1] < p1[1]):
			return 225
		elif (p2[1] == p1[1]):
			return 270
		else:
			return 315

def dist(p1,p2):
	return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def path(angl, size_of_cell, next_move):
	#next move is a cost_list
	path = []
	prev_angle = angl #0 radians
	for i in range(len(next_move)-1):
		cur = next_move[i]
		nex = next_move[i+1]
		cur_angle = angle(cur, nex)
		if prev_angle != cur_angle:
			# print cur_angle	print prev_angle
			turn_angle = prev_angle - cur_angle
			#print turn_angle
			orientation = "R"
			if turn_angle < 0:
				orientation  = "L"

			if math.fabs(turn_angle) > 180:
				turn_angle = 360 - math.fabs(turn_angle)

			turn_angle = math.fabs(turn_angle)
			prev_angle = cur_angle # this is what we
			path.append((turn_angle,orientation))
			d = dist(cur, nex) #This is how much we will move by
			path.append(d * size_of_cell)
		else:
			d = dist(cur,nex) #This is how much we will move by
			if not path:
				path.append(d)
			else:
				path[-1] = path[-1] + d
	return path #instructions

def smoothen_path(moving_instructions, max_angle):
	smooth_path = []
	for i, command in enumerate(moving_instructions):
		#if we are starting by moving forward, then just add this to list
		if (i==0 and type(command) is not tuple):
			smooth_path.append(command)
		if type(command) is tuple: #rotation case
			angle = command[0]
			rotation = command[1]
			if (angle <= max_angle): #we will NOT do slicing
				smooth_path.append(command)
			else:
				#Do slicing
				#Access the number to move forward by
				forward = moving_instructions[i+1] #this i+1 shouldn't be a problem since the last thing is always a float to move by
				num_times_to_loop = angle/max_angle
				for i in range(int(num_times_to_loop)):
					smooth_path.append((max_angle, rotation))
					smooth_path.append(forward * float(max_angle)/angle)
				#do the remaining angle and movement
				remaining_angle = angle - (int(num_times_to_loop) * max_angle)
				remaining_distance = forward - (forward * float(max_angle)/angle) * int(num_times_to_loop)
				if (remaining_angle > 0):
					smooth_path.append((remaining_angle, rotation))
					smooth_path.append(remaining_distance)

	#write
	f = open(v.instructions_file,"w+")
	for i, instruction in enumerate(smooth_path):
		f.write(str(instruction) + " ")
		# if (i != len(smooth_path)-1):
			# f.write("\n")
	f.close()
	return smooth_path

def fill_path_file(n, next_move):
	if (n==1):
		f = open("path1.txt","w+")
	else:
		f = open("path2.txt","w+")
	x_coordinates = [move[0] for move in next_move]
	y_coordinates = [move[1] for move in next_move]
	f.write(str(x_coordinates))
	f.write("\n")
	f.write(str(y_coordinates))
	f.close()
	print "works"

def main(argv):
	filename = argv[1] #csv file

	#elevation matrix
	dem = np.loadtxt(open(filename, "rb"), delimiter=" ")
	print dem.shape


	#cost matrix has the same size as the dem
	cost1 = np.empty(dem.shape)
	cost1.fill(-1)

	#fill_cost1(dem,cost1, v.start_cell, v.goal_cell)
	#fill_cost1(dem,cost1, (0,0), (dem.shape[0]-1, dem.shape[1]-1))
	fill_cost1(dem,cost1, (0,350), (350,0))

	next_move = [] #initially empty list
	#output_path(cost1, v.start_cell, v.goal_cell, next_move, v.max_rotation_angle)
	#output_path(cost1, (0,0), (dem.shape[0]-1, dem.shape[1]-1), next_move, v.max_rotation_angle)
	output_path(cost1, (0,350), (350,0), next_move, v.max_rotation_angle)


	list_of_commands = path(v.initial_angle, v.size_of_cell, next_move) #returns instructions

	print_path_1 = False

	if print_path_1:
		print "\nELEVATION MATRIX"
		print DataFrame(dem)
		print "\nCOST1 MATRIX"
		print DataFrame(cost1)
		print "\nFrom " + str(v.start_cell) + " to " + str(v.goal_cell) + " the path is: "
		print next_move
		print "Moving Instructions are "
		print str(list_of_commands)

	fill_path_file(1, next_move)

	#cost matrix has the same size as the dem
	#cost2 = np.array(cost1) #just copy the first matrix
	cost2 = np.empty(dem.shape)
	cost2.fill(-1)
	# fill_cost2(dem,cost1,cost2, (0,0), (dem.shape[0]-1, dem.shape[1]-1), v.max_slope)
	fill_cost2(dem,cost1,cost2, (0,350), (350,0), v.max_slope)

	next_move = [] #initially empty list
	# output_path(cost2, (0,0), (dem.shape[0]-1, dem.shape[1]-1), next_move, v.max_rotation_angle)
	print "all good"
	output_path(cost2, (0,350), (350,0), next_move, v.max_rotation_angle)
	list_of_commands = path(v.initial_angle, v.size_of_cell, next_move) #returns instructions
	smooth_path = smoothen_path(list_of_commands, v.max_angle)

	print_path_2 = False

	if print_path_2:
		print "\nCOST2 MATRIX"
		print DataFrame(cost2)
		print "\nFrom " + str(v.start_cell) + " to " + str(v.goal_cell) + " the path is: "
		print next_move
		print "Moving Instructions are "
		print str(list_of_commands)
		print "\nFrom " + str(v.start_cell) + " to " + str(v.goal_cell) + " the SMOOTHENED path is: "
		print smooth_path

	fill_path_file(2, next_move)

	#motors(list_of_commands) #DEFINED ON MEBA"s NOTEBOOK

if __name__ == "__main__":
	#print "RUN as: python elifs_dstar.py em2.csv"
	#print "Having trouble? See Elif for packages to install to run it"
	#test_functions()
	print "in d_star"
	main(sys.argv)
