import numpy as np
import matplotlib.pylab as plt
import ast

#open dem.txt
#read into an z_array

f = open("grid_result.txt", "r")
if f.mode == "r":
    dem = np.loadtxt("grid_result.txt")
f.close()


f = open("path1.txt","r")
if f.mode == "r":
    lines = f.readlines()
f.close()

x = ast.literal_eval(lines[0])
y = ast.literal_eval(lines[1])
plt.scatter(x,y,color='yellow')


f = open("path2.txt","r")
if f.mode == "r":
    lines = f.readlines()
f.close()

x = ast.literal_eval(lines[0])
y = ast.literal_eval(lines[1])
plt.scatter(x,y,color='green')

plt.imshow(dem, origin = "lower")
plt.show()
