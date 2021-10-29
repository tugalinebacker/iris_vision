#!/usr/bin/env python3
#coding: utf-8

import os
import rospy
import tf
import math
from geometry_msgs.msg import Vector3, Quaternion
from std_msgs.msg import Float64, String, Header
from cola2_msgs.msg import DVL
from sensor_msgs.msg import Imu

class Doppler:
    def __init__(self):
        self.dvl_data = 0
        self.velocity = 0
        self.altitude = 0
        self.x_velocity = 0
        # self.pub_data = rospy.Publisher("/dvl/data", DVL, queue_size=10)
        self.pub_velocity = rospy.Publisher("/dvl/velocity", Float64, queue_size=10)
        self.pub_altitude = rospy.Publisher("/dvl/altitude", Float64, queue_size=10)
        self.dvl_subscriber = rospy.Subscriber("/iris/navigator/dvl", DVL, self.callback)
        
    def callback(self, data):
        self.dvl_data = data
        self.velocity = data.velocity
        self.altitude = data.altitude
        self.x_velocity = data.velocity.x
        self.pub_velocity.publish(self.x_velocity)
        self.pub_altitude.publish(self.altitude)
        # self.pub_data.publish(self.data)

class IMU:
    def __init__(self):
        # TESTS
        # self.pub_orientation = rospy.Publisher("/IMU/orientation", Quaternion, queue_size=10)
        # self.pub_orientation_x = rospy.Publisher("/IMU/orientation_x", Float64, queue_size=10)
        # self.ang_vel = rospy.Publisher("/IMU/av", Vector3, queue_size=10)
        # self.ang_vel_x = rospy.Publisher("/IMU/av_x", Float64, queue_size=10)
        self.imu_subscriber = rospy.Subscriber("/iris/navigator/imu", Imu, self.callback_IMU)
        self.flag_yaw = rospy.Publisher("/iris/yaw_control", String, queue_size=10)


    def callback_IMU(self, data):
        self.iris_orientation = data.orientation
        self.iris_orientation_x = data.orientation.x
        self.iris_av = data.angular_velocity
        self.iris_av_x = data.angular_velocity.x

        #TUPLE TO SAVE ORIENT DATA IN QUATERNION
        self.orientation = (
            data.orientation.x,
            data.orientation.y,
            data.orientation.z,
            data.orientation.w
        )
        
        #GET ORIENTATION IN EULER AND CONVERT TO DEGREE
        self.euler = tf.transformations.euler_from_quaternion(self.orientation)
        self.roll = (self.euler[0]*180)/math.pi
        self.pitch = (self.euler[1]*180)/math.pi
        self.yaw = (self.euler[2]*180)/math.pi

        rospy.loginfo('Roll: %sº', self.roll)
        rospy.loginfo('Pitch: %sº', self.pitch)
        rospy.loginfo('Yaw: %sº', self.yaw)

        if self.yaw >= 160:
            # self.flag_yaw.publish("WARNING")
            rospy.loginfo('WARNING')

        # self.angular = (
        #     data.angular_velocity.x,
        #     data.angular_velocity.y,
        #     data.angular_velocity.z,
        # )

        # rospy.loginfo('AV_x: %s', self.angular[0])
        # rospy.loginfo('AV_y: %s', self.angular[1])
        # rospy.loginfo('AV_z: %s', self.angular[2])


        # TESTS
        # self.pub_orientation.publish(self.iris_orientation)
        # self.pub_orientation_x.publish(self.iris_orientation_x)
        # self.ang_vel.publish(self.iris_av)
        # self.ang_vel_x.publish(self.iris_av_x)




if __name__ == '__main__':
    try:
        rospy.init_node('sensor_nav')
        Doppler()
        IMU()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

    
