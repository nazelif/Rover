import numpy as np
from pandas import DataFrame #to pretty print matrices
from scipy import signal
import matplotlib.pylab as plt
import variables as v
import math
from scipy.interpolate import griddata

#now turn this into Spherical coordinates and save in sph.txt
def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    el = np.arctan2(z,np.sqrt(XsqPlusYsq))     # theta
    az = np.arctan2(y,x)                         # phi
    return (az,el,r)

#Main function that calls the function above
def cart2sphA(pts, sph):
    for x in range(pts.shape[0]):
        for y in range(pts.shape[1]):
            z = pts[x][y]
            #sph[x][y] = cart2sph(x,y,z)
            sph.append(cart2sph(x,y,z))

#Spherical to cartesian
def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return (x, y, z)

def sph2cartA(sph, cart, x_matrix, y_matrix, z_matrix, lidar):
    if lidar:
        for line in sph:
            az = float(line.strip().split()[0])
            el = float(line.strip().split()[1])
            r = float(line.strip().split()[2])
            res = sph2cart(az, el, r)
            x_matrix.append(int(res[0]))
            y_matrix.append(int(res[1]))
            z_matrix.append(res[2])
            cart.append(res)
    else:
        for line in sph:
            az = line[0]
            el = line[1]
            r = line[2]
            res = sph2cart(az, el, r)
            x_matrix.append(int(res[0]))
            y_matrix.append(int(res[1]))
            z_matrix.append(res[2])
            cart.append(res)


#Generates a DEM, provided size nx,ny in variables.py
def generate_dem():
    dem1 = np.random.rand(v.nx,v.ny)

    x, y = np.mgrid[-v.sizex:v.sizex+1, -v.sizey:v.sizey+1]
    g = np.exp(-0.333*(x**2/float(v.sizex)+y**2/float(v.sizey)))
    filter = g/g.sum()

    demSmooth = signal.convolve(dem1,filter,mode='valid')
    # rescale so it lies between 0 and 1
    demSmooth = (demSmooth - demSmooth.min())/(demSmooth.max() - demSmooth.min())

    '''print "DEM1 SHAPE is " + str(dem1.shape)
    print "DEMSMOOTH SHAPE is " + str(demSmooth.shape)'''
    return demSmooth

def display_dem(dem):
    plt.imshow(dem, origin = "lower")
    plt.show()

#I used this when I was bothering with turning indices to integers
def int_index_cart_matrix(x_matrix, y_matrix, z_matrix):
    z_array = np.nan * np.empty((len(x_matrix),len(y_matrix)))
    z_array[x_matrix, y_matrix] = z_matrix
    return z_array

#I didnt need to use it, but this is code to delete nan vals
def delete_nan_values(elevation_matrix):
    ''' ATTEMPT TO DELETE nan VALUES
        print "z_array SHAPE is " + str(z_array.shape)
        print "x_matrix SHAPE is " + str(len(x_matrix))
        print "y_matrix SHAPE is " + str(len(y_matrix))
        for i in range(len(x_matrix)) :
            for j in range(len(y_matrix)):
                print i
                print j
                print z_array[i,j]
    '''
    df = DataFrame(elevation_matrix)
    for c in range(elevation_matrix.shape[1]):
        df.val = df.val.fillna((df.val.shift() + df.val.shift(-1))/2)
    return df.values

#Save np arrays into respective catergories
def savefiles(sph, cart_list, dem, dem_original_result, dem_cropped_result):
    f = open("sph.txt", "w+")
    np.savetxt("sph.txt", sph)
    f.close()

    f = open("cart.txt", "w+")
    np.savetxt("cart.txt", cart_list)
    f.close()

    f = open("dem.txt", "w+")
    np.savetxt("dem.txt", dem)
    f.close()

    f = open("dem_original_result.txt", "w+")
    np.savetxt("dem_original_result.txt", dem_original_result)
    f.close()

    f = open("dem_cropped_result.txt", "w+")
    np.savetxt("dem_cropped_result.txt", dem_cropped_result)
    f.close()

#Creates DEM from given cartesian coordinates
def cart_index(m,n,x_bind_factor,y_bind_factor,x_matrix, y_matrix, z_matrix):
    #first create an m by n MATRIX
    arr = np.zeros([m,n])
    #now, for each row, we map to a certain place...
    for i in range(len(x_matrix)):
        x = x_matrix[i]/x_bind_factor
        y = y_matrix[i]/y_bind_factor
        z = z_matrix[i]
        if arr[x][y] == 0:
            arr[x][y] = z
        else:
            arr[x][y] = (arr[x][y] + z)/2
    return arr

def main():

    ''' all i need to do is create a dem out of these list of points
    1. read in line by line
    2. x maps to x, y maps to y...
    3. fill in nan values with averages '''

    # I can just call cart_index
    # m, n will be max VALUES
    #bind factors are 1
    #fill in each x,y,z metricies from the file


    x_matrix = []
    y_matrix = []
    z_matrix = []
    x_max = 0
    y_max = 0
    x_min = 0
    y_min = 0

    # dont add anything that is bigger than 200 in z
    for line in open(v.test):
        row = line.split(',') #returns a list ["1","50","60"]
        if (int(row[2]) <= 200):
            x_matrix.append(int(row[0]))
            y_matrix.append(int(row[1]))
            z_matrix.append(int(row[2]))
            if (int(row[0]) > x_max):
                x_max = int(row[0])
            if (int(row[0]) < x_min):
                x_min = int(row[0])
            if (int(row[1]) > y_max):
                y_max = int(row[1])
            if (int(row[1]) < y_min):
                y_min = int(row[1])

    x_range = x_max - x_min
    y_range = y_max - y_min


    # plt.scatter(x_matrix, y_matrix, color='green')
    #plt.show()

    #lidar_result = cart_index(v.m,v.n,v.x_bind_factor,v.y_bind_factor,x_matrix, y_matrix, z_matrix)

    arr = np.zeros([x_range+1,y_range+1])
    #do shifting of points here

    x_matrix2 = x_matrix
    y_matrix2 = y_matrix

    for i in range(len(x_matrix)):
        x_matrix2[i] = x_matrix[i] - x_min
        x = x_matrix2[i]
        y_matrix2[i] = y_matrix[i] - y_min
        y = y_matrix2[i]
        z = z_matrix[i]
        if arr[x][y] == 0:
             arr[x][y] = z
        else:
             arr[x][y] =  (arr[x][y] + z)/2

    lidar_result = arr #[0:100, 0:100]

    f = open("lidar_result.txt", "w+")
    np.savetxt("lidar_result.txt", lidar_result)
    f.close()
    # display_dem(lidar_result)


    print x_min
    print x_max
    print y_min
    print y_max

    grid_x, grid_y = np.mgrid[x_min:x_max:1, y_min:y_max:1]#[-1000:700:1, 0:500:1]
    points = np.asarray([x_matrix,y_matrix]).T
    values = np.asarray(z_matrix)
    '''print points.shape
    print values.shape
    print type((grid_x, grid_y))'''
    grid_z0 = griddata(points, values, (grid_y, grid_x), method='nearest')
    plt.imshow(grid_z0.T, extent=(x_min, x_max, y_min, y_max), origin='lower')

    f = open("grid_result.txt", "w+")
    np.savetxt("grid_result.txt", grid_z0)
    f.close()
    plt.title("grid result")
    plt.show()

    #
    #
    # if not v.show_dem:
    #     display_dem(grid_z0)
    #     display_dem(lidar_result)

if __name__ == "__main__":
    main()
