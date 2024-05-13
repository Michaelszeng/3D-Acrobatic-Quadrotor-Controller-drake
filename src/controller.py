""" Miscellaneous Utility functions """
from pydrake.all import (
    LeafSystem,
    BasicVector,
    AbstractValue,
)
import numpy as np

from src.utils import *


"""
Quadrotor state is represented as the (18,) vector:
[x, y, z, x_dot, y_dot, z_dot, R1, R2, R3, R4, R5, R6, R7, R8, R9, W1, W2, W3].T
"""
class DirtyDerivative:
    def __init__(self, order, tau=0.05):
        self.tau = tau
        self.order = order  # order of derivative
        self.dot = None  # store computed derivative
        self.x_d1 = None  # store previous data sample
        self.it = 1  # iteration counter

    def calculate(self, x, Ts):
        """
        x is the data sample.

        Ts is the time from the last sample.
        """
        self.a1 = (2 * self.tau - Ts) / (2 * self.tau + Ts)
        self.a2 = 2 / (2 * self.tau + Ts)

        x = np.array(x).reshape(-1, 1)  # Ensure x is a column vector
        if self.it == 1:
            self.dot = np.zeros_like(x)
            self.x_d1 = np.zeros_like(x)

        if self.it > self.order:
            self.dot = self.a1 * self.dot + self.a2 * (x - self.x_d1)

        self.it += 1
        self.x_d1 = x
        return self.dot


class SE3Controller(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        
        # Define input port for the current state of the drone
        self.input_port_drone_state = self.DeclareVectorInputPort("drone_state", 18)  # [x, y, z, x_dot, y_dot, z_dot, R1, R2, R3, R4, R5, R6, R7, R8, R9, W1, W2, W3].T

        # Define input port for the trajectory from the DDP solver
        self.input_port_x_trajectory = self.DeclareVectorInputPort("x_trajectory", 18)
        
        # Define output port for the controller output
        self.output_port_controller_output = self.DeclareVectorOutputPort("controller_output",
                                                                           4,
                                                                           self.CalcOutput)
        
        # Control gains (taken from Lee2011, arXiv:1003.2005v4)
        self.kx = 16 * m
        self.kv = 5.6 * m
        self.kR = 8.81
        self.kW = 2.54

        self.prev_desired_state = np.empty((18,))
        self.prev_desired_state.fill(np.nan)
        self.prev_desired_state_t = 0

        self.xd_ddot = np.zeros(3)
        self.Wd_dot = np.zeros(3)

        
    def CalcOutput(self, context, output):
        """
        Computes the output of the SE3 controller.

        Args:
            context: The Context object containing the input data.
            output: The output port to which the computed controller output is set.
        """
        # Retrieve input data from input ports
        drone_state = self.get_input_port(0).Eval(context)
        print(drone_state)
        desired_state = self.get_input_port(1).Eval(context)
        print(desired_state)

        if not np.any(np.isnan(self.prev_desired_state)) and not np.all(np.isclose(desired_state, self.prev_desired_state)):
            # New desired_state has been received
            t = context.get_time()

            self.xd_ddot = (desired_state[:3] - self.prev_desired_state[:3]) / (t - self.prev_desired_state_t + eps)  # Difference Quotient
            self.Wd_dot = (desired_state[15:] - self.prev_desired_state[15:]) / (t - self.prev_desired_state_t + eps)  # Difference Quotient

            self.prev_desired_state = desired_state
            self.prev_desired_state_t = t

            print(f"{self.xd_ddot=}")
            print(f"{self.Wd_dot=}")

        # Drone current state
        x = drone_state[:3]
        v = drone_state[3:6]
        R = drone_state[6:15].reshape(3, 3)
        W = drone_state[15:]

        # Position/velocity/accelertion desired and error
        xd = desired_state[:3]
        vd = desired_state[3:6]
        e_x = x - xd
        e_v = v - vd
        xd_ddot = self.xd_ddot  # for convenience so I don't have to repeat `self.`

        # Rotation/Rotational velocity/Angular acceleration desired and error
        A = -self.kx*e_x - self.kv*e_v + m*g*np.array([0,0,1]) + m*xd_ddot
        # print(f"{A=}")
        b3d = A / np.linalg.norm(A)                                          # b3d is determined by necesary heading to reach position setpoint
        b1d = desired_state[6:15].reshape(3, 3) @ np.array([1, 0, 0])        # b1d is set by the DDP trajectory
        # print(f"{b1d=}")
        b2d = np.cross(b3d, b1d) / np.linalg.norm(np.cross(b3d, b1d))        # b2d is computed as cross product of b3d and Proj(b1d) onto the normal plane to b3d
        Rd_traj = desired_state[6:15].reshape(3, 3)
        Rd = np.concatenate((np.cross(b2d, b3d), b2d, b3d)).reshape(3, 3)
        # print(f"{Rd=}")
        # print(f"{Rd_traj=}")
        Wd = desired_state[15:]
        e_R = 0.5 * vee_map(Rd.T @ R - R.T @ Rd)
        e_W = W - R.T @ Rd @ Wd
        Wd_dot = self.Wd_dot  # for convenience so I don't have to repeat `self.`

        f = (A * R @ np.array([0, 0, 1]))[2]  # z-component holds force magnitude
        M = -self.kR*e_R - self.kW*e_W + np.cross(W, I @ W) - I @ (hat_map(W) @ R.T @ Rd @ Wd - R.T @ Rd @ Wd_dot)

        net_force_moments_matrix = np.array([[1, 1, 1, 1],
                                             [0, L, 0, -L],
                                             [-L, 0, L, 0],
                                             [kM, -kM, kM, -kM]])

        # print(f"{f=}")
        # print(f"{M=}")

        u = np.linalg.inv(net_force_moments_matrix) @ np.concatenate(([f], M))
        # print(f"{u=}")

        # Set output
        output.SetFromVector(u)