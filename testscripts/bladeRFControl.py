#subprocess library to open and pass new commands to the bladeRF-cli programs we need to run the SDR
from subprocess import Popen, PIPE, STDOUT
import time
import sys
from os import path, remove
import os

class bladeRFControl:

    def __init__(self):

        #the device with 35d must be the master becuause the clkout on "35d" is connected to clkin on "cbd"
        self.master_process = Popen(['bladeRF-cli', '-d', '*:serial=cbd','-s','/home/jta/bladerf_testing/testscripts/mastercommands', '-i'], stdin=PIPE, stdout=PIPE)

        print("Master Opened")
        #give it some time to open the device and load the commands
        #we need to wait because we can't set the external clock on the slave until the master is already outputting a clock signal
        time.sleep(4)

        self.slave_process = Popen(['bladeRF-cli', '-d', '*:serial=35d', '-s', '/home/jta/bladerf_testing/testscripts/slavecommands', '-i'], stdin=PIPE, stdout=PIPE)

        print("Slave Opened")
        # same thing, give it some time to open and load commands
        time.sleep(4)

        print("BladeRFs opened")

    def PrepareReceive(self):

        self.sendBladeRFCommand("rx start;", "master",waitTime=1)

        self.sendBladeRFCommand("rx start;", "slave",waitTime=1)

        print("Rx Started")

        time.sleep(3)

    def recieve(self):

        print("Sending Trigger")

        self.sendBladeRFCommand("trigger j51-1 tx fire", "master", waitTime=1)

        print("Done Receiving")

    def sendBladeRFCommand(self, command, device, waitTime=0):

        commandBytes = bytes(command + "\n", 'utf-8')

        if device == "master":
            self.master_process.stdin.write(commandBytes)

        else:
            self.slave_process.stdin.write(commandBytes)

        if waitTime != 0:
            time.sleep(waitTime)

    def printBladeRFCommand(self, command, device):


        commandBytes = bytes(command + "\n", 'utf-8')

        if device == "master":
            self.master_process.stdin.write(commandBytes)
            while True:
                line = self.master_process.stdout.readline()
                if line != 'bladeRF>':
                    print(line.strip())
                else:
                    break
            
            


    def closeDevices(self):
        
        print("Closing BladeRFs")

        time.sleep(5)

        self.sendBladeRFCommand("q", "master")
        self.sendBladeRFCommand("q", "slave")

        print("Closed Devices Successfully")

b = bladeRFControl()

# b.PrepareReceive()

# b.recieve()

b.closeDevices()