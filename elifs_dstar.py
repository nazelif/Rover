import csv
import math


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
	

def testPoint(x=0,y=0,z=0):
    p1 = Point(3, 4,5)
    p2 = Point(3,0,5)
    return Point.distance(p1,p2)


#em is elevation matrix
em_matrix = list(csv.reader(open("em.csv")))


def slope(p1,p2):
	return ((p2.Z - p1.Z) / math.sqrt( (p2.X-p1.X)**2 + (p2.Y-p1.Y)**2 ))

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

def test_functions():
	#print em_matrix
	#print "distance = %s"%(testPoint()) 
	p1 = Point(3, 4,5)
	p2 = Point(3,0,5)
	p3 = Point(2,1,2)
	print "cost = %s"%(cost(p1,p2))
	print "cost = %s"%(cost(p1,p3))
	print "cost = %s"%(cost(p2,p3))
	print "slope = %s"%(slope(p1,p2))
	print "slope = %s"%(slope(p1,p3))
	print "slope = %s"%(slope(p2,p3))

def main():
	start_cell = 10
	goal_cell = 13

if __name__ == "__main__":
	test_functions()
