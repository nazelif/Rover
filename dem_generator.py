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

def sph2cartA(sph, cart):
    for line in sph:
        az = line[0]
        el = line[1]
        r = line[2]
        cart.append(sph2cart(az, el, r))

def main():
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

    sph = [] #np.empty(demSmooth.shape)
    cart = []

    cart2sphA(demSmooth, sph)
    sph2cartA(sph, cart)

    np.savetxt("sph.txt", sph)
    np.savetxt("cart.txt", cart)

    #print DataFrame(demSmooth)
    np.savetxt("dem.txt", demSmooth)

    plt.imshow(dem1)
    plt.imshow(demSmooth)

    plt.show()

if __name__ == "__main__":
    main()
