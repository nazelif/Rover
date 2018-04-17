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

def ask_meba():
	#for one rotation of the wheel the distance traversed by rover
	dist = 2 * math.pi * r #r is the radius of the wheel
	#encoder measurement (p1) for moving the rover by a distance of dist unites
	dist1 = 10
	p1 = traveling_factor * dist1 / dist

def fill_cost1(dem,cost1, cost2, start_cell, goal_cell):
	#now fill 10s for h/v moves, 14 for diagonal moves
	cost1[goal_cell] = 0
	cost2[goal_cell] = 0
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

def fill_cost2(dem,cost1, cost2, start_cell, goal_cell):
	#now fill 10s for h/v moves, 14 for diagonal moves
	cost1[goal_cell] = 0
	cost2[goal_cell] = 0
	g_row = goal_cell[0]
	g_col = goal_cell[1]



	#filling rows up
	for x in range(cost1.shape[0]): #shape[0] is num rows
		if g_row-x >= 1:
			s = slope(Point(g_row-x, g_col, dem[g_row-x, g_col]), Point(g_row-x-1, g_col, dem[g_row-x-1, g_col]))
			cost2[g_row-x-1, g_col] = cost2[g_row-x, g_col] + s+ 10
		if g_row-x<1: #until we hit 0 with g_col-x-1 = 0 i.e. until g_col-x = 1
			break

		'''print "\nCOST MATRIX"
		print DataFrame(cost1)'''

	#filling rows down
	for x in range(cost1.shape[0]): #shape[0] is num rows
		if g_row+x+1 <= cost1.shape[0]-1:
			s = slope(Point(g_row+x+1, g_col, dem[g_row+x+1, g_col]), Point(g_row+x, g_col, dem[g_row+x, g_col]))
			cost2[g_row+x+1, g_col] = cost2[g_row+x, g_col] + s + 10
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

def output_path(cost1, cost2, start_cell, goal_cell, next_move):
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
				temp_val = cost1[cur_xy[0]+1 , cur_xy[1]+1]
				temp_xy = (cur_xy[0]+1,cur_xy[1]+1)

		if (cur_xy[1] != 0) and (cur_xy[0] != 0): #we are checking upper left diagonal
			if (cost1[cur_xy[0]-1, cur_xy[1]-1] <= temp_val):
				temp_val = cost1[cur_xy[0]-1 , cur_xy[1]-1]
				temp_xy = (cur_xy[0]-1, cur_xy[1]-1)

		if (cur_xy[1] != cost1.shape[1]-1) and (cur_xy[0] != 0): #we are checking upper right diagonal
			if (cost1[cur_xy[0]-1, cur_xy[1]+1] <= temp_val):
				temp_val = cost1[cur_xy[0]-1 , cur_xy[1]+1]
				temp_xy = (cur_xy[0]-1, cur_xy[1]+1)

		if (cur_xy[1] != 0) and (cur_xy[0] != cost1.shape[0]-1): #we are checking lower left diagonal
			if (cost1[cur_xy[0]+1, cur_xy[1]-1] <= temp_val):
				temp_val = cost1[cur_xy[0]+1 , cur_xy[1]-1]
				temp_xy = (cur_xy[0]+1, cur_xy[1]-1)

		if (cur_xy[0] != cost1.shape[0]-1): #we are checking down
			if (cost1[cur_xy[0]+1, cur_xy[1]] <= temp_val):
				temp_val = cost1[cur_xy[0]+1 , cur_xy[1]]
				temp_xy = (cur_xy[0]+1, cur_xy[1])

		if (cur_xy[0] != 0): #we are checking up
			if (cost1[cur_xy[0]-1, cur_xy[1]] <= temp_val):
				temp_val = cost1[cur_xy[0]-1 , cur_xy[1]]
				temp_xy = (cur_xy[0]-1, cur_xy[1])

		if (cur_xy[1] != cost1.shape[1]-1): #we are checking rhs
			if (cost1[cur_xy[0], cur_xy[1]+1] <= temp_val):
				temp_val = cost1[cur_xy[0] , cur_xy[1]+1]
				temp_xy = (cur_xy[0],cur_xy[1]+1)

		if (cur_xy[1] != 0): #we are checking lhs
			if (cost1[cur_xy[0], cur_xy[1]-1] <= temp_val):
				temp_val = cost1[cur_xy[0] , cur_xy[1]-1]
				temp_xy = (cur_xy[0],cur_xy[1]-1)

		'''print temp_val
		print temp_xy'''

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

	start_cell = (0,0) #row 1 col 0
	goal_cell = (3,2) #row 1 col 3

	filename = argv[1] #csv file

	#elevation matrix
	dem = np.loadtxt(open(filename, "rb"), delimiter=",")
	print "\nELEVATION MATRIX"
	print DataFrame(dem)

	#cost matrix has the same size as the dem
	cost1 = np.empty(dem.shape)
	cost1.fill(-1)

	#cost matrix has the same size as the dem
	cost2 = np.empty(dem.shape)
	cost2.fill(-1)

	fill_cost1(dem,cost1, cost2, start_cell, goal_cell)
	print "\nCOST MATRIX"
	print DataFrame(cost1)

	'''
	fill_cost2(dem,cost1, cost2, start_cell, goal_cell)
	print "\nCOST2 MATRIX"
	print DataFrame(cost2)
	'''

	#update_cost1(cost1, start_cell, goal_cell)	#doesn't do anything yet

	next_move = [] #initially empty list
	output_path(cost1, cost2, start_cell, goal_cell, next_move)
	print "\nFrom " + str(start_cell) + " to " + str(goal_cell) + " the path is: "
	print next_move
	print path(size_of_cell, next_move) #returns instructions

	pstart = Point(start_cell[0], start_cell[1], cost1[start_cell])
	pgoal = Point(goal_cell[0], goal_cell[1], cost1[goal_cell])
	#print slope(pstart,pgoal)


if __name__ == "__main__":
	#print "RUN as: python elifs_dstar.py em2.csv"
	#print "Having trouble? See Elif for packages to install to run it"
	#test_functions()
	main(sys.argv)
