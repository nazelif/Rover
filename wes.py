# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Python Tutorial
# Wesley Andres Watters
# ASTR 203, Wellesley College, Spring 2013
# version 2013.04.01+21.33

# Pylab has several libraries for scientific computing, called NumPy, SciPy, and
# Matplotlib.  Parts of these are bundled into a single library called Pylab.

# Here is a great resource for all the functions in matplotlib, most of which
# can be found in the library "pylab": http://matplotlib.org/

# While Python, NumPy, SciPy, and Matplotlib are free (in *both* senses of that
# word), a company called Enthought packages them for convenient download and
# installation on Mac OS and Windows.  If you use GNU/Linux, installing these
# libraries is trivial and you won't need any help from Enthought.  To download
# a copy of pylab from Enthought, go to this address:

# http://www.enthought.com/products/epd_free.php

# Note that you can download all of these libraries in other ways.  This is one
# tool you don't have to worry about losing after you graduate! (i.e., when
# student licenses typically expire).

# To begin, it is important to be clear about the difference between the python
# terminal prompt and the unix terminal prompt ("python prompt" and "unix
# prompt" for short).  When you open a "terminal window" on the Mac (see
# Applications -> Utilities -> Terminal) or on Linux, you are shown a unix
# prompt where you can type only *unix* commands.  The unix prompt usually ends
# in a dollar sign ($).  Here is where you type "python" to enter a "python
# terminal", where the prompt looks like ">>>".  At the python prompt, you can
# type only python commands.

# Download all of the files for the homework assignment to the same directory.
# If this directory is called "Downloads", then you will need to change to that
# directory at the unix prompt before you can run the script.  You can usually
# do this by typing "cd Downloads" at the unix prompt.  To confirm that this
# script is in the current directlry, type "ls" to print a directory listing and
# make sure all of the necessary files are in the current directory.  To run the
# current program (the one you're reading), at the unix prompt type "python -i
# python_tutorial.py".  After the script runs, you will be returned to the
# *python* prompt, where you can query variables.

# A python "script" is a text file like this one that contains a bunch of python
# commands.  These can be executed in the way I just mentioned, by typing
# "python -i script_name.py" at the unix prompt.  Python then executes each line
# of code separately.  Alternatively (and this is *not* recommended), you could
# open python and copy-and-paste each line of code into the python prompt to
# have the same effect.

# When writing a solution to the homework assignment in python, you should write
# print statements that indicate the values that are being printed, as well as
# print statements that actually print those values.  Here is an example:

A = 5.0
print('The value of A is: ')
print(A)

# When you run this script (or if you copy this text into the python prompt),
# the output will be:

# The value of A is:
# 5.0

# Do not copy and paste the result that is shown in the python terminal back
# into this script.  Executing the script will print all of the relevant
# results, and you can just copy those into your write-up.

# Arithmetic operations in python are very simple.  An asterisk indicates
# multiplication, a minus sign indicates subtraction, a plus sign addition, and
# a forward-slash (/) means division.  A double asterisk (**) indicates "to the
# power of".  Always express your numbers as decimals, to make sure that the
# answers are not rounded to integers.

# If you are using Linux, uncomment the "ion()" command below.  This puts the
# script in an "interactive mode" so that you don't have to close the figure
# windows to recover the python prompt.  Unfortunately, this doesn't seem to
# work on Macs.

# Finally, note that you can find help at the command prompt about any function
# by typing:

# help(function_name)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# This line is an example of a comment: it's not read by the Python interpreter
# because it is prefaced by the hash symbol (#).  It's important to add comments
# to that other people (including a future you) can easily make sense of code
# that you wrote.

# First, let's load pylab, an essential "library" of useful tools.

from pylab import *   # Load everything from the pylab library
from numpy import *   # Load everything from the numpy library

x = arange(0,30,0.5)  # Make array of numbers from 0 to 30 spaced by 0.5

y = x**2 * sin(x)     # Compute the function x**2 * sin(x)

# An array is a collection of numbers.  It can be 1-D (a string of numbers) or
# it can be multi-dimensional.  You can think of a 2-D array as a matrix,
# although what you've learned about matrix multiplication does not apply!  An
# array is a kind of matrix of numbers.  To multiply two arrays, they must have
# the same size, in which case every element in the first array is multiplied by
# a corresponding element in the second array (note that this is unlike matrix
# multiplication: for *that*, python has a special kind of array called a
# matrix).

# If you would like to print out the values of a specific element in the array
# (say, the 5th one), use the print command and specify the relevant element by
# its index in square brackets. (Note that python starts counting at "0": that
# is, the first element in an array is indexed by 0, the second by 1, the third
# by 2, and so on...)

print('Printing the first and fifth value of x:')
print(x[0], x[4])

print('Printing the first and fifth value of y: ')
print(y[0], y[4])

# Note that x**n means x to the nth power!

#ion()                # This puts python in interactive mode, so that your plots
                      # show up immediately and do not block input at the python
                      # prompt.  Unfortunately, it doesn't work on Macs, so
                      # leave this line commented if you're using a Mac.

# Now let's plot our function:

figure(1)             # Create a figure

plot(x,y,'-o')        # This plots the function we defined above with large
                      # dots connected by straight lines

xlabel('x [m]')       # This adds the x axis label
ylabel('y [m]')       # This adds the y axis label

title('ASTR 203 Python tutorial demo plot')

# >> This shows you how to plot a few points that are specified explicitly

xp = np.array([  5.0,  10.0,  15.0,  20.0,   22.0])
yp = np.array([100.0,   0.0,-100.0, 450.0, -100.0])

plot(xp,yp, 's')

# >> Note that you can save any plot to an image file, for later processing in
# >> another program like GIMP and Inkscape (or Photoshop and Illustrator).
# >> This comes in handy if you need to annotate one of your plots.

savefig('plot01.png', dpi=150)  # save with resolution of 150 dots per inch

# Note that you can add numbers to arrays, or multiply them by any factor:

y2 = 0.5 * y + 200.0 # Multiply the y values by one half, add 200
plot(x,y2,'-')       # Plot the result as a continuous line (no dots)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Now let's do this in more dimensions.  First, generate a "mesh grid": this is
# a pair of arrays, the first of which contains the x coordinate at each point
# in space, and the second of which contains the y coordinate at each point in
# space.  We take the 1-dimensional arrays that we generated above to create
# these two 2-dimensional arrays. (You won't need this step to do hwk 4: this
# just shows you how to create 2-D arrays of x and y values in order to
# construct a surface mathematically.  Our goal here is to generate some
# artificial topography.)

[x,y] = meshgrid(x,x)

# For example, the x and y coordinates of the 5th point in the x direction and
# the 10th point in the y direction are:

print('Printing 5th (x,y) coordinates of 5th point in x direction, 10th in y:')
print(x[4,9],y[4,9])  # ... remember that python starts counting at zero!

# Now we can define a 3-D surface as functions of x and y; this is called
# "elev", short for "elevation": our goal is to generate an artificial elevation
# map.  In this case, the elevations will resemble the surface of an egg carton:

elev = sin(x) + cos(y)

# The rest of this part is concerned with plotting the result.  First, we set
# the limits of each axis as the minimum and maximum values of the arrays x and
# y.  For example, the minimum value of an array "A" is "A.min()".  The
# following variable stores the 2-D axis limits used by the plotting function
# (again, this is not something that you need for the assignment, but it comes
# in handy):

axis_limits = [x.min(), x.max(), y.min(), y.max()]

# Now create a new figure and plot the results using imshow.  In the reference
# to "imshow" (which means "image show"), "extent" is set equal to the x and y
# axis limits, and "cmap" sets the color map for representing the 'elevation" of
# the surface at each point.

figure(2)
imshow(elev, extent = axis_limits, cmap = cm.jet)
colorbar()

# Once again, let's add the axis labels:

xlabel('x [m]')
ylabel('y [m]')
title('Artificial topography')

# The variable elev now holds a 2-D array, like the maps that we'll use in the
# assignment.  You can think of it as an image: it has a set of pixels in a
# two-dimentional array, each of which have values representing an artificial
# elevation.  Note that we can also multiply arrays (maps) by each other,
# multiply them by numbers, as well as perform arithmetic with arrays.  Some
# examples are shown below.  Note that "**" Means "to the power of".

figure(3)

elev2 =  x**2 * 2*elev - 5*y**2
imshow(elev2, extent = axis_limits, cmap = cm.jet)

xlabel('x [m]')
ylabel('y [m]')
title('Artificial topography arithmetic')

# Now suppose we have a map like the one that's stored in elev.  And suppose
# that we would like to extract a profile along the 30th column of the array and
# plot it. We can do this as follows:

figure(4)
Z =  elev[:,29]  # Remember that python starts counting at 0, so the 30th
X =     y[:,29]  # column corresponds to index 29.

plot(X,Z)

xlabel('x [m]');
ylabel('y [m]');
title('Transect of artificial topography')

# Note that we can extract some simple statistics from elev as follows:

print('minimum elevation :')
print(elev.min())
print('maximum elevation :')
print(elev.max())
print('average elevation :')
print(elev.mean())

# We can even print the histogram of elevations in elev as follows (the function
# .ravel() "flattens" the 2-D array into a 1-D array).

figure(5)
hist(elev.ravel(), 50)            # Plot the histogram with 50 bins
title('Elevation histogram')

# Scientific notation and final notes: Please note that the number 3.0 * 10**5
# is represented as 3.0e5.  For the assignment, always work with decimal
# numbers: don't use integers.  That is, to set a variable equal to two, use a =
# 2.0, and not a = 2.

show()
