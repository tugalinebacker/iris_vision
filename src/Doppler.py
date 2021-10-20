#!/usr/bin/env python3
#coding: utf-8

import os
import rospy
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64, String, Header
from cola2_msgs.msg import DVL

class Doppler:
    def __init__(self):
        self.dvl_data = 0
        self.velocity = 0
        self.altitude = 0
        self.x_velocity = 0
        self.pub_velocity = rospy.Publisher("/dvl/velocity", Float64, queue_size=1000)
        self.pub_altitude = rospy.Publisher("/dvl/altitude", Float64, queue_size=1000)
        self.dvl_subscriber = rospy.Subscriber("/iris/navigator/dvl", DVL, self.callback)
        
    def callback(self, data):
        self.dvl_data = data
        self.velocity = data.velocity
        self.altitude = data.altitude
        self.x_velocity = data.velocity.x
        self.pub_velocity.publish(self.x_velocity)
        self.pub_altitude.publish(self.altitude)


if __name__ == '__main__':
    rospy.init_node('dvl_control')
    Doppler()
    rospy.spin()

    
