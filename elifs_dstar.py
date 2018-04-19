import sys
import csv
import math
from pandas import DataFrame #to pretty print matrices
import numpy as np

traveling_factor = 10
ascending_slope_factor = 20
descending_slope_factor = 10
r = 10 #radius of wheel
size_of_cell = 1 #for now

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
			slope *= ascending_slope_factor
		else: #we are descending, descend factor = 10
			slope *= descending_slope_factor
		return math.fabs(slope)
	except ZeroDivisionError:
		print "Same point, Different elevation, check points"
		print(p1)
		print(p2)
		return -1

def cost(p1,p2):
	units = Point.distance(p1,p2)
	cost = traveling_factor * units
	if (p1.getX != p2.getX) ^ (p1.getY != p2.getY): #diagonally moving when x and y are both  different values
		cost *= math.sqrt(2)
	slope_factor = ascending_slope_factor
	if slope(p1,p2) < 0:
		slope_factor = descending_slope_factor
	return cost + math.fabs(slope(p1,p2)) * slope_factor

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

		'''print "\nCOST MATRIX"
		print DataFrame(cost1)'''

	#filling rows up
	for x in range(cost1.shape[0]): #shape[0] is num rows
		if g_row-x >= 1:
			cost1[g_row-x-1, g_col] = cost1[g_row-x, g_col] + 10
		if g_row-x<1: #until we hit 0 with g_col-x-1 = 0 i.e. until g_col-x = 1
			break

		'''print "\nCOST MATRIX"
		print DataFrame(cost1)'''

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

	#print "\nCOST MATRIX"
	#print DataFrame(cost1)

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

def fill_cost2(dem,cost1, cost2, bt, start_cell, goal_cell):

	explored = [] #keep track of all visited nodes
	queue = [goal_cell] #keep track of nodes to be checked

	#Loop until q is empty
	while queue:
		node = queue.pop(0)
		if node not in explored:
			explored.append(node) #add it to the list of checked nodes
			if (node[0] == goal_cell[0] and node[1] == goal_cell[1]):
				cost2[goal_cell] = 0
			else:
				fill_cell(dem, cost2, bt, goal_cell, node[0], node[1])
			neighbors = get_nbrs(cost2, node[0], node[1]) #via the function defined below
			for neighbor in neighbors:
				queue.append(neighbor)
			'''print "**"
			print node
			print "**"
			print queue
			print "**"
			print explored'''


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

def fill_cell(dem, cost2, bt, goal_cell, x, y):
	#the value at x will be the min of 8 values we will compute with the function
	''' components to calculate from cost2[a,b] where a,b are max 1 unit away from x,y
		val_of_prev_cell = cost2[a,b] #this is assumed to be filled before, if not what to do?
		distance = fabs(cost1[a,b] - cost1[x,y]) #might need to change it with 10 or 14
		slope = (dem[a,b] - dem[x,y])/ distance
		if slope < 0 (ascending), use ascending_slope_factor
		else use descending_slope_factor
	'''

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

		#check if [a,b] is Out Of Boundries
		if (a >= 0 and b >= 0 and a < cost2.shape[0] and b < cost2.shape[1] and cost2[a,b] != -1):
			#print "Computations to fill " + str((x,y)) + " using the cell " + str((a,b)) + "\n"
			temp_val = cost2[a,b] #assuming that this value is the minimum it can be, and it won't change
			distance = 10 if math.fabs(a+b-x-y) == 1 else 14
			temp_val += distance
			slope = (dem[a,b] - dem[x,y])/ distance
			#print slope
			temp_val += math.fabs(slope) * ascending_slope_factor if slope < 0 else math.fabs(slope) * descending_slope_factor
			'''print i
			print cost2[a,b]
			print distance
			print slope
			print temp_val'''
			if temp_val < temp:
				temp = temp_val
				bt[x,y] = i

	'''print temp, cost2[x,y]'''
	cost2[(x,y)] = temp

def output_path(cost1, start_cell, goal_cell, next_move):
	#from the start point, trace a path towards the goal cell
	#follow the min pt
	cur_val = cost1[start_cell]
	cur_xy = start_cell
	cost_list = []

	next_move.append(cur_xy) #we start at the start cell
	cost_list.append(cur_val)

	temp_val = cur_val
	temp_xy = cur_xy

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

def path(size_of_cell, next_move):
	#next move is a cost_list
	path = []
	prev_angle = 0 #0 radians
	#print next_move
	for i in range(len(next_move)-1):
		cur = next_move[i]
		nex = next_move[i+1]
		cur_angle =  angle(cur, nex)
#		print cur, nex
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
			path.append((turn_angle, orientation))
			d = dist(cur, nex) #This is how much we will move by
			path.append(d * size_of_cell)
		else:
			d = dist(cur, nex) #This is how much we will move by
			path[-1] = path[-1] + d
	return path #instructions

def main(argv):

	start_cell = (1,0) #row 1 col 0
	goal_cell = (1,3) #row 1 col 3

	filename = argv[1] #csv file

	#elevation matrix
	dem = np.loadtxt(open(filename, "rb"), delimiter=",")
	print "\nELEVATION MATRIX"
	print DataFrame(dem)

	#cost matrix has the same size as the dem
	cost1 = np.empty(dem.shape)
	cost1.fill(-1)

	bt = np.empty(dem.shape)
	bt.fill(-1)

	fill_cost1(dem,cost1, start_cell, goal_cell)
	print "\nCOST1 MATRIX"
	print DataFrame(cost1)

	#cost matrix has the same size as the dem
	#cost2 = np.array(cost1) #just copy the first matrix
	cost2 = np.empty(dem.shape)
	cost2.fill(-1)

	fill_cost2(dem,cost1,cost2, bt, start_cell, goal_cell)
	print "\nCOST2 MATRIX"
	print DataFrame(cost2)
	print "\nBT MATRIX"
	print DataFrame(bt)

	#update_cost1(cost1, start_cell, goal_cell)	#doesn't do anything yet

	next_move = [] #initially empty list
	output_path(cost1, start_cell, goal_cell, next_move)
	print "\nFrom " + str(start_cell) + " to " + str(goal_cell) + " the path is: "
	print next_move


	next_move = [] #initially empty list
	output_path(cost2, start_cell, goal_cell, next_move)
	print "\nFrom " + str(start_cell) + " to " + str(goal_cell) + " the path is: "
	print next_move


	list_of_commands = path(size_of_cell, next_move) #returns instructions
	#motors(list_of_commands) #DEFINED ON MEBA"s NOTEBOOK

	pstart = Point(start_cell[0], start_cell[1], cost1[start_cell])
	pgoal = Point(goal_cell[0], goal_cell[1], cost1[goal_cell])
	#print slope(pstart,pgoal)


if __name__ == "__main__":
	#print "RUN as: python elifs_dstar.py em2.csv"
	#print "Having trouble? See Elif for packages to install to run it"
	#test_functions()
	main(sys.argv)
