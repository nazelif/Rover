import sys
import csv
import math
from pandas import DataFrame #to pretty print matrices
import numpy as np

traveling_factor = 10
ascending_slope_factor = 50
descending_slope_factor = 20
r = 10 #radius of wheel


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


def slope(p1,p2):
	try: 
		return ((p2.Z - p1.Z) / math.sqrt( (p2.X-p1.X)**2 + (p2.Y-p1.Y)**2))
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



def fill_cost_matrix(cost_matrix, start_cell, goal_cell):
	#now fill 10s for h/v moves, 14 for diagonal moves
	cost_matrix[goal_cell] = 0
	goal_row = goal_cell[0]
	goal_col = goal_cell[1]

	#filing cols to left of goal
	for x in range(cost_matrix.shape[1]): #shape[1] is num columns
		if goal_col-x >= 1: 
			cost_matrix[goal_row, goal_col-x-1] = cost_matrix[goal_row, goal_col-x] + 10
		if goal_col-x<1: #until we hit 0 with goal_col-x-1 = 0 i.e. until goal_col-x = 1
			break
		'''print "\nCOST MATRIX"
		print DataFrame(cost_matrix)'''

	#filling cols to right of goal
	for x in range(cost_matrix.shape[1]): #shape[1] is num columns
		#until we hit 0 with goal_col-x+1 = shape[1]-1 i.e. goal_col-x = s[1]-2
		if goal_col+x+1 <= cost_matrix.shape[1]-1:
			cost_matrix[goal_row, goal_col+x+1] = cost_matrix[goal_row, goal_col+x] + 10
		if goal_col+x+1 > cost_matrix.shape[1]-1:
			break

		'''print "\nCOST MATRIX"
		print DataFrame(cost_matrix)'''

	#filling rows up
	for x in range(cost_matrix.shape[0]): #shape[0] is num rows
		if goal_row-x >= 1: 
			cost_matrix[goal_row-x-1, goal_col] = cost_matrix[goal_row-x, goal_col] + 10
		if goal_row-x<1: #until we hit 0 with goal_col-x-1 = 0 i.e. until goal_col-x = 1
			break
		'''print "\nCOST MATRIX"
		print DataFrame(cost_matrix)'''

	#filling rows down
	for x in range(cost_matrix.shape[0]): #shape[0] is num rows
		if goal_row+x+1 <= cost_matrix.shape[0]-1:
			cost_matrix[goal_row+x+1, goal_col] = cost_matrix[goal_row+x, goal_col] + 10
		if goal_row+x+1 > cost_matrix.shape[0]-1:
			break

	#fill diagonals
	if goal_row != 0:
		if goal_col != 0:
			cost_matrix[goal_row-1, goal_col-1] = 14
		if goal_col != cost_matrix.shape[1]-1:
			cost_matrix[goal_row-1, goal_col+1] = 14
	
	if goal_row != cost_matrix.shape[1]-1:
		if goal_col != 0 :
			cost_matrix[goal_row+1, goal_col-1] = 14
		if goal_col != cost_matrix.shape[0]-1:
			cost_matrix[goal_row+1, goal_col+1] = 14
	
	print "\nCOST MATRIX"
	print DataFrame(cost_matrix)

	#1ST QUADRANT
	for row in range(goal_row-1,-1,-1): #every row
		for col in range(goal_col+1,cost_matrix.shape[1],1): #every cell in that row
			
			if cost_matrix[row][col] == -1:
				cost_matrix[row,col] = min(cost_matrix[row+1,col-1]+14,min(cost_matrix[row,col-1]+10, cost_matrix[row+1,col]+10)) 

	#2ND QUADRANT
	for row in range(goal_row-1,-1,-1): #every row
		for col in range(goal_col-1,-1,-1): #every cell in that row
			if cost_matrix[row][col] == -1:
				cost_matrix[row,col] = min(cost_matrix[row+1,col+1]+14,min(cost_matrix[row,col+1]+10, cost_matrix[row+1,col]+10)) 

	#3RD QUADRANT
	for row in range(goal_row+1,cost_matrix.shape[0],1): #every row
		for col in range(goal_col-1,-1,-1): #every cell in that row
			if cost_matrix[row][col] == -1:
				cost_matrix[row,col] = min(cost_matrix[row-1,col+1]+14,min(cost_matrix[row,col+1]+10, cost_matrix[row-1,col]+10)) 

	#4TH QUADRANT
	for row in range(goal_row+1,cost_matrix.shape[0],1): #every row
		for col in range(goal_col+1,cost_matrix.shape[1],1): #every cell in that row
			if cost_matrix[row][col] == -1:
				cost_matrix[row,col] = min(cost_matrix[row-1,col-1]+14,min(cost_matrix[row,col-1]+10, cost_matrix[row-1,col]+10)) 

	print "\nCOST MATRIX"
	print DataFrame(cost_matrix)


def update_cost_matrix(cost_matrix, start_cell, goal_cell):
	goal_row = goal_cell[0]
	goal_col = goal_cell[1]



	#start with the goal cell
	#update the cost with cost + slope*factor
	#the cost function does this but it takes points as input
	#making them point objects seems unnecessary
	#IDK what to do uggh

	''' To keep myself sane
	I will start from bottom right corner and go to top left corner

	for row in range(cost_matrix.shape[0]): #every row
		for col in range(cost_matrix.shape[1]): #every cell in that row
		'''


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

def main(argv):
	start_cell = (0,0) #row 1 col 0
	goal_cell = (0,0) #row 1 col 3
	filename = argv[1] #csv file

	#elevation matrix
	em_matrix = np.loadtxt(open(filename, "rb"), delimiter=",")
	print "\nELEVATION MATRIX"
	print DataFrame(em_matrix)

	#cost matrix has the same size as the em_matrix
	cost_matrix = np.empty(em_matrix.shape)
	cost_matrix.fill(-1)

	fill_cost_matrix(cost_matrix, start_cell, goal_cell)

	update_cost_matrix(cost_matrix, start_cell, goal_cell)


if __name__ == "__main__": 
	print "RUN as: python elifs_dstar.py em2.csv"
	#test_functions()
	main(sys.argv)
