# -*- coding: utf-8 -*-
import RPi.GPIO as GPIO # import GPIO library
import time
from time import sleep
import math

#some variables
GPIO.setmode(GPIO.BOARD)
C = 2*math.pi*0.1016

def setpins(Motor1A, Motor1B, Motor1E, Motor2A, Motor2B, Motor2E):
    # set pins as outputs and inputs
    #GPIO.setwarnings(False)
    GPIO.setup(Motor1A,GPIO.OUT)
    GPIO.setup(Motor1B,GPIO.OUT)
    GPIO.setup(Motor1E,GPIO.OUT)

    GPIO.setup(Motor2A,GPIO.OUT)
    GPIO.setup(Motor2B,GPIO.OUT)
    GPIO.setup(Motor2E,GPIO.OUT)

def turn_left(Motor1A, Motor1B, Motor1E, Motor2A, Motor2B, Motor2E, time, pwm1, pwm2):
    pwm2.start(80)
    GPIO.output(Motor2A,GPIO.LOW) # set terminal A of motor to HIGH
    GPIO.output(Motor2B,GPIO.HIGH) # set terminal B of motor to LOW
    GPIO.output(Motor2E,GPIO.HIGH)

    pwm1.start(100)
    GPIO.output(Motor1A,GPIO.LOW) # set terminal A of motor to HIGH
    GPIO.output(Motor1B,GPIO.HIGH) # set terminal B of motor to LOW
    GPIO.output(Motor1E,GPIO.HIGH)

    sleep(time)
    GPIO.output(Motor1E,GPIO.LOW)
    GPIO.output(Motor2E,GPIO.LOW)

    print"pause"
    pwm1.start(0)
    pwm2.start(0)
    sleep(2)

def turn_right(Motor1A, Motor1B, Motor1E, Motor2A, Motor2B, Motor2E, time, pwm1, pwm2):
    pwm1.start(80)
    GPIO.output(Motor1A,GPIO.LOW) # set terminal A of motor to HIGH
    GPIO.output(Motor1B,GPIO.HIGH) # set terminal B of motor to LOW
    GPIO.output(Motor1E,GPIO.HIGH)

    pwm2.start(100)
    GPIO.output(Motor2A,GPIO.LOW) # set terminal A of motor to HIGH
    GPIO.output(Motor2B,GPIO.HIGH) # set terminal B of motor to LOW
    GPIO.output(Motor2E,GPIO.HIGH)

    sleep(time)
    GPIO.output(Motor1E,GPIO.LOW)
    GPIO.output(Motor2E,GPIO.LOW)

def move_straight(Motor1A, Motor1B, Motor1E, Motor2A, Motor2B, Motor2E, time, distance, pwm1, pwm2):
    pwm1.start(100)
    GPIO.output(Motor1A,GPIO.HIGH) # set terminal A of left motor to HIGH
    GPIO.output(Motor1B,GPIO.LOW) # set terminal B of left motor to LOW
    GPIO.output(Motor1E,GPIO.HIGH) # enable left motor to move

    pwm2.start(100)
    GPIO.output(Motor2A,GPIO.HIGH) # set terminal A of right motor to HIGH
    GPIO.output(Motor2B,GPIO.LOW) # set terminal B of right motor to LOW
    GPIO.output(Motor2E,GPIO.HIGH) # enable right motor to move

    time = distance/(C*5) # time it will take to reach distance
    sleep(time) # keep moving for this much time


### Note: relative High/low status of terminals A and B determines direction of rotation
def move(list_of_commands, Motor1A, Motor1B, Motor1E, Motor2A, Motor2B, Motor2E, pwm1, pwm2):
    for command in list_of_commands:
        time = 0
        if type(command) is tuple:
            rotation_angle = command[0]
            angle_rad = rotation_angle * math.pi/180 # converting degrees to rad
            time = angle_rad * 0.25/(0.25*2*math.pi*0.319)
            direction = command[1] #Left or right
            if (direction == 'L'):
                print "turn left"
                turn_left(Motor1A, Motor1B, Motor1E, Motor2A, Motor2B, Motor2E, time, pwm1, pwm2)
            else:
                print "turn right"
                turn_right(Motor1A, Motor1B, Motor1E, Motor2A, Motor2B, Motor2E, time, pwm1, pwm2)
        else:#MOVING STRAIGHT
            distance = command
            move_straight(Motor1A, Motor1B, Motor1E, Motor2A, Motor2B, Motor2E, time, distance, pwm1, pwm2)


def main(argv):
    # Declare pin names
    Motor1A = 3
    Motor1B = 5
    Motor1E = 7
    Motor2A = 16
    Motor2B = 18
    Motor2E = 22
    setpins(Motor1A, Motor1B, Motor1E, Motor2A, Motor2B, Motor2E)

    #sets pins as pwms
    pwm1 = GPIO.PWM(7,100)
    pwm2 = GPIO.PWM(22,100)

    #set duty cycle t0 number in parenthesis%
    pwm1.start(0)
    pwm2.start(0) # start at rest and wait the sleep time in seconds before starting

    print "wait"
    sleep(3)

    # read in the intructions file
    if (len(argv) == 2):
        instructions_file = argv[1]
        f = open(instructions_file, "r")
        command_list = f.readlines()

    else:
        command_list = [(5.0, 'R'), 1.5]

    print command_list
    move(command_list, Motor1A, Motor1B, Motor1E, Motor2A, Motor2B, Motor2E, pwm1, pwm2)

    GPIO.cleanup() # cleans commands from pins

if __name__ == "__main__":
    main(sys.argv)
