""" Miscellaneous Utility functions """
from pydrake.all import (
    LeafSystem,
    RigidTransform,
    RotationMatrix,
)
from manipulation.meshcat_utils import AddMeshcatTriad

import numpy as np

from src.utils import *


"""
Quadrotor state is represented as the (18,) vector:
[x, y, z, x_dot, y_dot, z_dot, R1, R2, R3, R4, R5, R6, R7, R8, R9, W1, W2, W3].T

Note: R is rotation from body-fixed frame to world frame ({}^W R^B)
Note: W is w.r.t. body-fixed frame
"""

kp = 50.0
kv = 50.0
kR = 3.0
kW = 2.55

class Controller(LeafSystem):
    def __init__(self, meshcat, SHOW_TRIADS):
        LeafSystem.__init__(self)
        
        # Define input port for the current state of the drone
        self.input_port_drone_state = self.DeclareVectorInputPort("drone_state", 18)  # [x, y, z, x_dot, y_dot, z_dot, R1, R2, R3, R4, R5, R6, R7, R8, R9, W1, W2, W3].T

        # Define input port for the desired state from the DDP trajectory
        self.input_port_desired_state = self.DeclareVectorInputPort("desired_state", 18)

        # Define input port for the desired acceleration from the DDP trajectory
        self.input_port_desired_acceleration = self.DeclareVectorInputPort("desired_acceleration", 3)

        # Define input port for the desired angular acceleration from the DDP trajectory
        self.input_port_desired_angular_acceleration = self.DeclareVectorInputPort("desired_angular_acceleration", 3)
        
        # Define output port for the controller output
        self.output_port_controller_output = self.DeclareVectorOutputPort("controller_output",
                                                                           4,
                                                                           self.CalcOutput)
        
        self.meshcat = meshcat
        self.prev_desired_state = np.zeros(18)
        self.SHOW_TRIADS = SHOW_TRIADS
        
        
    def CalcOutput(self, context, output):
        """
        Computes the output of the SE3 controller.

        Args:
            context: The Context object containing the input data.
            output: The output port to which the computed controller output is set.
        """
        drone_state = self.input_port_drone_state.Eval(context)
        # print(drone_state)
        desired_state = self.input_port_desired_state.Eval(context)
        # print(desired_state)
        desired_accel = self.input_port_desired_acceleration.Eval(context)
        # print(desired_accel)
        desired_angular_accel = self.input_port_desired_angular_acceleration.Eval(context)

        if self.SHOW_TRIADS and not np.all(np.isclose(desired_state, self.prev_desired_state)):
            AddMeshcatTriad(self.meshcat, f"Triads/Desired_State_t={context.get_time()}", X_PT=RigidTransform(RotationMatrix(desired_state[6:15].reshape((3,3))), desired_state[:3]), opacity=0.5)
        self.prev_desired_state = desired_state

        # TEMPORARY
        # desired_state = np.array([0, 0, 4, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0.1])
        # desired_accel = np.array([0, 0, 0])

        # Current State
        p = drone_state[:3]                     # position in inertial frame
        v = drone_state[3:6]                    # velocity in inertial frame
        R = drone_state[6:15].reshape(3, 3)     # rotation (^W R^B)
        W = R.T @ drone_state[15:]              # angular velocity in body frame (convert from inertial to body frame)

        # Desired position/velocity/acceleration/yaw
        pd = desired_state[:3]                  # desired position in inertial frame
        vd = desired_state[3:6]                 # desired velocity in inertial frame
        ad = desired_accel                      # desired accel in inertial frame
        yawd = np.arctan2(desired_state[9], desired_state[6])  # desired yaw = arctan(R_21 / R_11)
        Wd = R.T @ desired_state[15:]           # desired angular velocity in body frame (convert from inertial to body frame)
        Ad = R.T @ desired_angular_accel        # desired angular acceleration in body frame (convert from inertial to body frame)

        # Error Values
        ep = p - pd     # position error in inertial frame
        ev = v - vd     # velocity error in inertial frame

        b3d = -kp*ep- kv*ev + m*g*e3 + m*ad
        b3d /= np.linalg.norm(b3d)
        b1d = np.array([np.cos(yawd), np.sin(yawd), 0.])
        b2d = np.cross(b3d, b1d)
        b2d /= np.linalg.norm(b2d)

        Rd = np.hstack((np.cross(b2d, b3d).reshape(3, 1), b2d.reshape(3, 1), b3d.reshape(3, 1)))
        
        eR = 0.5 * vee_map(Rd.T @ R - R.T @ Rd)
        eW = W - Wd  # Wd is already expressed in body frame, so there is no need to transform it

        # Desired force and moment
        f = -kp*ep - kv*ev + m*g*e3 + m*ad  # Ideal force vector in world frame
        f_z = (f).dot(R * e3)[2]  # Project ideal force into body-frame Z-axis and take z-component
        M = -kR*eR - kW*eW + np.cross(W, J @ W) - J @ (hat_map(W) @ R.T @ Rd @ Wd - R.T @ Rd @ Ad)

        # Backsolve rotor velocities from desired force and moment
        u = np.linalg.inv(F2W) @ np.concatenate(([f_z], M))
        # print(f"{u=}")

        output.SetFromVector(u)