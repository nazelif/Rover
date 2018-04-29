import numpy as np
from pandas import DataFrame #to pretty print matrices
from scipy import signal
import matplotlib.pylab as plt

#now turn this into Spherical coordinates and save in sph.txt
def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    el = np.arctan2(z,np.sqrt(XsqPlusYsq))     # theta
    az = np.arctan2(y,x)                         # phi
    return (az,el,r)

def cart2sphA(pts, sph):
    for x in range(pts.shape[0]):
        for y in range(pts.shape[1]):
            z = pts[x][y]
            #sph[x][y] = cart2sph(x,y,z)
            sph.append(cart2sph(x,y,z))

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return (x, y, z)

def sph2cartA(sph, cart, x_matrix, y_matrix, z_matrix):
    for line in sph:
        az = line[0]
        el = line[1]
        r = line[2]
        res = sph2cart(az, el, r)
        x_matrix.append(int(res[0]))
        y_matrix.append(int(res[1]))
        z_matrix.append(res[2])
        cart.append(res)

def generate_dem():
    nx = 36 #360
    ny = 36 #360

    dem1 = np.random.rand(nx,ny)

    sizex = 3 #30
    sizey = 1 #10

    x, y = np.mgrid[-sizex:sizex+1, -sizey:sizey+1]
    g = np.exp(-0.333*(x**2/float(sizex)+y**2/float(sizey)))
    filter = g/g.sum()

    demSmooth = signal.convolve(dem1,filter,mode='valid')
    # rescale so it lies between 0 and 1
    demSmooth = (demSmooth - demSmooth.min())/(demSmooth.max() - demSmooth.min())

    '''print "DEM1 SHAPE is " + str(dem1.shape)
    print "DEMSMOOTH SHAPE is " + str(demSmooth.shape)'''
    return demSmooth

def display_dem(dem):
    plt.imshow(dem)
    plt.show()

def int_index_cart_matrix(x_matrix, y_matrix, z_matrix):
    z_array = np.nan * np.empty((len(x_matrix),len(y_matrix)))
    z_array[x_matrix, y_matrix] = z_matrix
    return z_array

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

def savefiles(sph, cart_list, dem, elevation_int_index, dem_result):
    np.savetxt("sph.txt", sph)
    np.savetxt("cart.txt", cart_list)
    np.savetxt("dem.txt", dem)
    np.savetxt("dem_int_index.txt", elevation_int_index)
    np.savetxt("dem_result.txt", dem_result)

def cart_index(m,n,x_size,y_size,x_matrix, y_matrix, z_matrix, cart_list):
    #first create an m by n MATRIX
    arr = np.zeros([m,n])
    #now, for each row, we map to a certain place...
    for line in cart_list:
        if arr[line[0]][line[1]] == 0:
            arr[line[0]][line[1]] = line[2]
        else:
            arr[line[0]][line[1]] = (arr[line[0]][line[1]] + line[2])/2
    return arr


def main():

    ''' FIRST, create a DEM'''
    dem = generate_dem()

    ''' CONVERT DEM TO SPH'''
    sph = [] #np.empty(demSmooth.shape)
    cart2sphA(dem, sph)

    ''' CONVERT SPH TO CART'''
    cart_list = []
    x_matrix = []
    y_matrix = []
    z_matrix = []
    sph2cartA(sph, cart_list, x_matrix, y_matrix, z_matrix)

    elevation_int_index = int_index_cart_matrix(x_matrix, y_matrix, z_matrix)
    #delete_nan_values(elevation_int_index)
    m = 30
    n = 34
    x_size = 1
    y_size = 1


    dem_result = cart_index(m,n,x_size,y_size,x_matrix, y_matrix, z_matrix,cart_list)

    savefiles(sph, cart_list, dem, elevation_int_index,dem_result)
    display_dem(dem)

if __name__ == "__main__":
    main()
