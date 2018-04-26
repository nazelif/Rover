# function to generate a terain to use as input for testing read_csv_into_matrix
from random import *

def generate_terrain(file, num_rows):
    for r in range(num_rows):
        file.write(str(randint(-90,90))) #alt btw 90,-90
        file.write(" ")
        file.write(str(randint(0,360))) #az btw 0,360
        file.write(" ")
        file.write(str(randint(0,1000))) #range in cms
        file.write(" ")
        file.write("\n")

def main():
    file = open("alt_az_range.txt", "w")
    generate_terrain(file, 1000)

if __name__ == "__main__":
	main()
