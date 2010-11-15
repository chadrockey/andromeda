#! /usr/bin/env python
# -*- coding: utf-8 -*-

import roslib
roslib.load_manifest("sparkfun_9dof_razor")
import rospy

from sparkfun_9dof_razor.msg import State
from sparkfun_9dof_razor.msg import RawFilter
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion
import tf.transformations as tf_math

from numpy import *
import numpy.linalg

class SF9DOF_UKF:
    def __init__(self):
        self.is_initialized = False
        self.pub = rospy.Publisher("imu", Imu)
        self.raw_pub = rospy.Publisher("raw_filter", RawFilter)
        self.beta = rospy.get_param("~beta", 2.)
        self.alpha = rospy.get_param("~alpha", 0.001)
        self.kappa = rospy.get_param("~kappa", 0.)
        self.n = 4
        self.kf_lambda = pow(self.alpha,2.) * (self.n + self.kappa) - self.n
        self.weight_covariance = ones(self.n * 2 + 1)
        self.weight_mean = ones(self.n * 2 + 1)
        self.weight_mean = self.weight_mean * (1. / (2 * (self.n +
            self.kf_lambda)))
        self.weight_covariance = self.weight_covariance * (1. / (2 * (self.n +
            self.kf_lambda)))
        self.weight_mean[0] = self.kf_lambda / (self.n + self.kf_lambda)
        self.weight_covariance[0] = self.kf_lambda / (self.n + self.kf_lambda)\
                + (1- pow(self.alpha, 2) + self.beta)

    def initialize_filter(self, time):
        self.time = time
        self.kalman_state = zeros((self.n,1))
        # initalize quaternions
        self.kalman_state[3,0] = 1.0
        self.kalman_covariance = diag(ones(self.kalman_state.shape[0]))
        self.is_initialized = True

    @staticmethod
    def prediction(current_state, dt, controls = None):
	predicted_state = current_state.copy()
	return predicted_state

    @staticmethod
    def process_noise(current_state, dt, controls = None):
        noise = ones(current_state.shape[0]) * 0.01
        noise[0:4] = .01 # quaternion uncertanty
        return diag(noise)

    @staticmethod
    def measurement_noise(measurement, dt):
        noise = ones(measurement.shape[0]) * 0.01
        noise[0:3] = .01 # accelerometer noise
        return diag(noise)

    @staticmethod
    def measurement_update(current_state, dt, measurement):
        predicted_measurement = zeros(measurement.shape)
        a = current_state[0,0]
        b = current_state[1,0]
        c = current_state[2,0]
        d = current_state[3,0]
        # Rotation matrix components for quaternions
        #R11 = math.pow(d,2)+math.pow(a,2)-math.pow(b,2)-math.pow(c,2)
        #R12 = 2*(a*b-c*d)
        #R13 = 2*(a*c+b*d)
        #R21 = 2*(a*b+c*d)
        #R22 = math.pow(d,2)+math.pow(b,2)-math.pow(a,2)-math.pow(c,2)
        #R23 = 2*(b*c-a*d)
        R31 = 2*(a*c-b*d)
        R32 = 2*(b*c+a*d)
        R33 = math.pow(d,2)+math.pow(c,2)-math.pow(b,2)-math.pow(a,2)
        denom = math.pow(a,2)+math.pow(b,2)+math.pow(c,2)+math.pow(d,2)
        g = math.sqrt(math.pow(measurement[0,0],2)+math.pow(measurement[1,0],2)+math.pow(measurement[2,0],2))
        predicted_measurement[0,0] = R31*g/denom
        predicted_measurement[1,0] = R32*g/denom
        predicted_measurement[2,0] = R33*g/denom
        print current_state[0:4]
        print "----"
        return predicted_measurement
        
    def normalizeQuaternionAndCovariance(self):
	magnitude = linalg.norm(self.kalman_state[0:4])
	self.kalman_state[0:4] = self.kalman_state[0:4] / magnitude
	self.kalman_covariance[0:4,:] = self.kalman_covariance[0:4,:] / magnitude
	self.kalman_covariance[:,0:4] = self.kalman_covariance[:,0:4] / magnitude

    def estimate_mean(self, transformed_sigmas):
        est_mean = zeros(self.kalman_state.shape)
        #Compute estimated mean for non-quaternion components
        for i in range(0,self.n*2+1):
            est_mean += self.weight_mean[i] * transformed_sigmas[i]
        return est_mean

    def estimate_covariance(self, est_mean, transformed_sigmas):
        est_covariance = zeros(self.kalman_covariance.shape)
        diff = zeros(est_mean.shape)
        for i in range(0,self.n*2+1):
            diff = transformed_sigmas[i] - est_mean
            prod = dot(diff, diff.T)
            term = self.weight_covariance[i] * prod
            est_covariance += term
        return est_covariance

    def estimate_measurement_mean(self, measurement_sigmas):
        est_measurement = zeros(measurement_sigmas[0].shape)
        for i in range(0, self.n*2+1):
            term = self.weight_mean[i] * measurement_sigmas[i]
            est_measurement += term
        return est_measurement

    def estimate_measurement_covariance(self, measurement_mean, \
            measurement_sigmas):
        est_measurement_covariance = eye(measurement_mean.shape[0])
        for i in range(0,self.n*2+1):
            diff = measurement_sigmas[i] - measurement_mean
            prod = dot(diff, diff.T)
            term = self.weight_covariance[i] * prod
            est_measurement_covariance += term
        return est_measurement_covariance

    def cross_correlation_mat(self, est_mean, est_sigmas, meas_mean, meas_sigmas):
        cross_correlation_mat = zeros((est_mean.shape[0], meas_mean.shape[0]))
        est_diff = zeros(est_mean.shape)
        for i in range(0,self.n*2+1):
            est_diff = est_sigmas[i] - est_mean
            meas_diff = meas_sigmas[i] - meas_mean
            prod = dot(est_diff, meas_diff.T)
            term = self.weight_covariance[i] * prod
            cross_correlation_mat += term
        return cross_correlation_mat

    @staticmethod
    def stateMsgToMat(measurement_msg):
        measurement = zeros((3,1))
        measurement[0,0] = measurement_msg.linear_acceleration.x
        measurement[1,0] = measurement_msg.linear_acceleration.y
        measurement[2,0] = measurement_msg.linear_acceleration.z
        return measurement

    def handle_measurement(self, measurement_msg):
        if not self.is_initialized:
            rospy.logwarn("Filter is unintialized. Discarding measurement")
        else:
            t0 = rospy.Time.now()
            dt = (measurement_msg.header.stamp - self.time).to_sec()
            measurement = SF9DOF_UKF.stateMsgToMat(measurement_msg)
            self.measurement = measurement
            p_noise = SF9DOF_UKF.process_noise(self.kalman_state, dt)
            sigmas = self.generate_sigma_points(self.kalman_state,
                    self.kalman_covariance + p_noise)
            #Run each sigma through the prediction function
            transformed_sigmas = [SF9DOF_UKF.prediction(sigma, dt) for sigma in sigmas]
            #Estimate the mean
            est_mean = self.estimate_mean(transformed_sigmas)
            est_covariance = self.estimate_covariance(est_mean, \
                    transformed_sigmas)
            est_sigmas = self.generate_sigma_points(est_mean, est_covariance)
            #Run each of the new sigmas through the measurement update function
            measurement_sigmas = [SF9DOF_UKF.measurement_update(sigma, dt, \
                    measurement) for sigma in est_sigmas]
            measurement_mean = \
                    self.estimate_measurement_mean(measurement_sigmas)
            measurement_covariance = \
                    self.estimate_measurement_covariance(measurement_mean, \
                    measurement_sigmas)
            measurement_covariance += SF9DOF_UKF.measurement_noise(measurement_mean, dt)
            cross_correlation_mat = self.cross_correlation_mat(est_mean, \
                    est_sigmas, measurement_mean, measurement_sigmas)
            s_inv = numpy.linalg.pinv(measurement_covariance)
            kalman_gain = dot(cross_correlation_mat, s_inv)
            innovation =  measurement - measurement_mean
            correction = dot(kalman_gain, innovation)
            self.kalman_state = est_mean + correction
            temp = dot(kalman_gain, measurement_covariance)
            temp = dot(temp, kalman_gain.T)
            self.kalman_covariance = est_covariance - temp
	    # Normalize quaternion and adjust covariance matrix
	    self.normalizeQuaternionAndCovariance()
            self.time = measurement_msg.header.stamp
            self.publish_imu()
            self.publish_raw_filter()
            rospy.logdebug("Took %s sec to run filter",(rospy.Time.now() - \
                    t0).to_sec())
        
    def publish_imu(self):
        imu_msg = Imu()
        imu_msg.header.stamp = self.time
        imu_msg.header.frame_id = 'imu_odom'
        a = self.kalman_state[0,0]
        b = self.kalman_state[1,0]
        c = self.kalman_state[2,0]
        d = self.kalman_state[3,0]
        imu_msg.orientation.x = a
        imu_msg.orientation.y = b
        imu_msg.orientation.z = c
        imu_msg.orientation.w = d
        imu_msg.orientation_covariance = list(self.kalman_covariance[0:3,0:3].flatten())
        imu_msg.angular_velocity.x = 0
        imu_msg.angular_velocity.y = 0
        imu_msg.angular_velocity.z = 0
        imu_msg.angular_velocity_covariance = list(zeros((3,3)).flatten())
        imu_msg.linear_acceleration.x = self.measurement[0,0]
        imu_msg.linear_acceleration.y = self.measurement[1,0]
        imu_msg.linear_acceleration.z = self.measurement[2,0]
        acc_cov = SF9DOF_UKF.measurement_noise(self.measurement, 1.0)[0:4][0:4]
        imu_msg.linear_acceleration_covariance = list(acc_cov.flatten())
        self.pub.publish(imu_msg)

    def publish_raw_filter(self):
        filter_msg = RawFilter()
        filter_msg.header.stamp = self.time
        filter_msg.state = list(self.kalman_state.flatten())
        filter_msg.covariance = list(self.kalman_covariance.flatten())
        self.raw_pub.publish(filter_msg)

    def generate_sigma_points(self, mean, covariance):
        sigmas = []
        sigmas.append(mean)
        temp = numpy.linalg.cholesky(covariance)
        temp = temp * sqrt(self.n + self.kf_lambda)
        for i in range(0,self.n):
            #Must use columns in order to get the write thing out of the
            #Cholesky decomposition
            #(http://en.wikipedia.org/wiki/Cholesky_decomposition#Kalman_filters)
            column = temp[:,i].reshape((self.n,1))
            #Do the additive sample
            new_mean = mean + column
            sigmas.append(new_mean)
            #Do the subtractive sample
            new_mean = mean - column
            sigmas.append(new_mean)
        return sigmas

if __name__ == "__main__":
    rospy.init_node("sf9dof_ukf_quat", log_level=rospy.DEBUG)
    ukf = SF9DOF_UKF()
    ukf.initialize_filter(rospy.Time.now())
    rospy.Subscriber("state", State, ukf.handle_measurement)
    rospy.spin()
