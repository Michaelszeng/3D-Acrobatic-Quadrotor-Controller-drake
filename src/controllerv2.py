""" Miscellaneous Utility functions """
from pydrake.all import (
    LeafSystem,
    BasicVector,
    AbstractValue,
)
import numpy as np

from src.utils import *


def attitude_errors(R, Rd, W, Wd):
    eR = 0.5*vee_map(Rd.T.dot(R) - R.T.dot(Rd))
    eW = W - R.T.dot(Rd.dot(Wd))
    return (eR, eW)

def position_errors(x, xd, v, vd):
    ex = x - xd
    ev = v - vd
    return (ex, ev)

def get_Rc(A, A_dot, A_2dot, b1d, b1d_dot, b1d_ddot):
    # move this as a function
    norm_A = np.linalg.norm(A)
    b3c = - A/norm_A
    b3c_dot = - A_dot/norm_A + ( np.dot(A, A_dot)*A )/norm_A**3
    b3c_2dot = - A_2dot/norm_A + ( 2*np.dot(A*A_dot,A_dot) )/norm_A**3 + np.dot( A_dot* A_dot + A*A_2dot ,A)/norm_A**3 - 3*np.dot((A*A_dot)**2,A)/norm_A**5

    b_ = np.cross(b3c, b1d)
    b_norm = np.linalg.norm(b_)
    b_dot = np.cross(b3c_dot, b1d) + np.cross(b3c, b1d_dot)
    b_2dot = np.cross(b3c_2dot, b1d) + 2*np.cross(b3c_dot, b1d_dot) + np.cross(b3c, b1d_ddot)

    b1c = -np.cross( b3c, b_ )/b_norm
    b1c_dot = -( np.cross(b3c_dot, b_) + np.cross(b3c, b_dot) )/b_norm + np.cross(b3c, b_)*(b_dot* b_)/b_norm**3

    # intermediate steps to calculate b1c_2dot
    m_1 = ( np.cross(b3c_2dot, b_) + 2*np.cross(b3c_dot, b_dot) + np.cross(b3c, b_2dot) )/b_norm
    m_2 = ( np.cross(b3c_dot, b_) + np.cross(b3c, b_dot) )*np.dot(b_dot, b_)/b_norm**3
    m_dot = m_1 - m_2
    n_1 = np.cross(b3c, b_)*np.dot(b_dot, b_)
    n_1dot = ( np.cross(b3c_dot, b_) + np.cross(b3c, b_dot) )*np.dot(b_dot, b_) + np.cross(b3c, b_)*( np.dot(b_2dot, b_)+np.dot(b_dot, b_dot) )
    n_dot = n_1dot/b_norm**3 - 3*n_1*np.dot(b_dot, b_)/b_norm**5
    b1c_2dot = -m_dot + n_dot

    Rc = np.reshape([b1c, np.cross(b3c, b1c), b3c],(3,3)).T
    Rc_dot = np.reshape([b1c_dot, ( np.cross(b3c_dot, b1c) + np.cross(b3c, b1c_dot) ), b3c_dot],(3,3)).T
    Rc_2dot = np.reshape( [b1c_2dot, ( np.cross(b3c_2dot, b1c) + np.cross(b3c_dot, b1c_dot) + np.cross(b3c_dot, b1c_dot) + np.cross(b3c, b1c_2dot) ), b3c_2dot],(3,3)).T
    Wc = vee_map(Rc.T.dot(Rc_dot))
    Wc_dot= vee_map( Rc_dot.T.dot(Rc_dot) + Rc.T.dot(Rc_2dot))
    return (Rc, Wc, Wc_dot)


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
        # self.kx = 16 * m
        # self.kv = 5.6 * m
        # self.kR = 8.81
        # self.kW = 2.54
        self.kx = 16*m
        self.kv = 5.6*m
        self.kR = 0.001
        self.kW = 0.0

        self.prev_desired_state = np.empty((18,))
        self.prev_desired_state.fill(np.nan)
        self.prev_desired_state_t = 0

        self.xd_ddot = np.zeros(3)
        self.Wd_dot = np.zeros(3)




        self.m = 0.775
        self.g = 9.81
        self.J = I
        self.e3 = np.array([0.,0.,1.])
        # self.kR = 8.81 # attitude gains
        # self.kW = 2.54 # attitude gains
        self.kR = 0 # attitude gains
        self.kW = 0 # attitude gains
        self.kx = 16.*self.m # position gains
        self.kv = 5.6*self.m # position gains
        self.xd = None
        self.xd_dot = None
        self.command = None

        
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

        # Drone current state
        x = drone_state[:3]
        v = drone_state[3:6]
        R = drone_state[6:15].reshape(3, 3) @ np.array([[1,0,0],[0,-1,0],[0,0,-1]])  # rotate by 180 deg in x-axis to account for difference in body frame defn.
        W = np.array([drone_state[15], -drone_state[16], -drone_state[17]])  # negate b2 and b3 angular velocity to account for difference in body frame defn.

        xd = desired_state[:3]
        xd_dot = desired_state[3:6]
        xd_2dot = np.array([0, 0, 0])  # TEMPORARY
        xd_3dot = np.array([0, 0, 0])  # TEMPORARY?
        xd_4dot = np.array([0, 0, 0])  # TEMPORARY?
        
        b1d = desired_state[6:15].reshape(3, 3) @ np.array([1, 0, 0])        # b1d is set by the DDP trajectory
        b1d_dot = np.array([0, 0, 0])  # TEMPORARY
        b1d_ddot = np.array([0, 0, 0])  # TEMPORARY?

        (ex, ev) = position_errors(x, xd, v, xd_dot)

        f = np.dot(self.kx*ex + self.kv*ev + self.m*self.g*self.e3 - self.m*xd_2dot, R.dot(self.e3) )
        W_hat = hat_map(W)
        R_dot = R.dot(W_hat)
        x_2dot = self.g*self.e3 - f*R.dot(self.e3)/self.m
        ex_2dot = x_2dot - xd_2dot

        f_dot = ( self.kx*ev + self.kv*ex_2dot - self.m*xd_3dot).dot(R.dot(self.e3)) + ( self.kx*ex + self.kv*ev + self.m*self.g*self.e3 - self.m*xd_3dot).dot(np.dot(R_dot,self.e3))

        x_3dot = -1/self.m*( f_dot*R + f*R_dot ).dot(self.e3)
        ex_3dot = x_3dot - xd_3dot

        A = -self.kx*ex - self.kv*ev - self.m*self.g*self.e3 + self.m*xd_2dot
        A_dot = -self.kx*ev - self.kv*ex_2dot + self.m*xd_3dot
        A_2dot = -self.kx*ex_2dot - self.kv*ex_3dot + self.m*xd_4dot

        (Rd, Wd, Wd_dot) = get_Rc(A, A_dot, A_2dot , b1d, b1d_dot, b1d_ddot)

        (eR, eW) = attitude_errors( R, Rd, W, Wd )
        M= -self.kR*eR - self.kW*eW + np.cross(W, self.J.dot(W)) - self.J.dot(W_hat.dot(R.T.dot(Rd.dot(Wd))) - R.T.dot(Rd.dot(Wd_dot)))


        # These values match the body frame defn. in Lee et al.
        net_force_moments_matrix = np.array([[1, 1, 1, 1],
                                             [0, -L, 0, L],
                                             [L, 0, -L, 0],
                                             [-kM, kM, -kM, kM]])

        # print(f"{f=}")
        # print(f"{M=}")

        u = np.linalg.inv(net_force_moments_matrix) @ np.concatenate(([f], M))
        u = np.clip(u, -1e2, 1e2)
        u = np.array([u[0], u[3], u[1], u[2]])  # swap propellors 2 and 4 to account for difference in body frame defn.
        print(f"{u=}")

        # Set output
        output.SetFromVector(u)
        
        return (f, M)









        # if not np.any(np.isnan(self.prev_desired_state)) and not np.all(np.isclose(desired_state, self.prev_desired_state)):
        #     # New desired_state has been received
        #     t = context.get_time()

        #     self.xd_ddot = (desired_state[:3] - self.prev_desired_state[:3]) / (t - self.prev_desired_state_t + eps)  # Difference Quotient
        #     self.Wd_dot = (desired_state[15:] - self.prev_desired_state[15:]) / (t - self.prev_desired_state_t + eps)  # Difference Quotient

        #     self.prev_desired_state = desired_state
        #     self.prev_desired_state_t = t

        #     print(f"{self.xd_ddot=}")
        #     print(f"{self.Wd_dot=}")

        # Drone current state
        x = drone_state[:3]
        v = drone_state[3:6]
        R = drone_state[6:15].reshape(3, 3) @ np.array([[1,0,0],[0,-1,0],[0,0,-1]])  # rotate by 180 deg in x-axis to account for difference in body frame defn.
        W = np.array([drone_state[15], -drone_state[16], -drone_state[17]])  # negate b2 and b3 angular velocity to account for difference in body frame defn.

        # Position/velocity/accelertion desired and error
        xd = desired_state[:3]
        vd = desired_state[3:6]
        e_x = x - xd
        e_v = v - vd
        print(f"{e_x=}")
        print(f"{e_v=}")
        xd_ddot = self.xd_ddot  # for convenience so I don't have to repeat `self.`

        # Rotation/Rotational velocity/Angular acceleration desired and error
        A = -self.kx*e_x - self.kv*e_v + m*g*np.array([0,0,1]) + m*xd_ddot  # note that g is negative
        print(f"{A=}")
        b3d = -A / np.linalg.norm(A)                                         # b3d is determined by necessary heading to reach position setpoint
        b1d = desired_state[6:15].reshape(3, 3) @ np.array([1, 0, 0])        # b1d is set by the DDP trajectory
        b2d = np.cross(b3d, b1d) / np.linalg.norm(np.cross(b3d, b1d))        # b2d is computed as cross product of b3d and Proj(b1d) onto the normal plane to b3d
        print(f"{b1d=}, norm: {np.linalg.norm(b1d)}")
        print(f"{b2d=}, norm: {np.linalg.norm(b2d)}")
        print(f"{b3d=}, norm: {np.linalg.norm(b3d)}")
        # Computing Rd_traj just to compare to Rd; should be very close
        Rd_traj = desired_state[6:15].reshape(3, 3) @ np.array([[1,0,0],[0,-1,0],[0,0,-1]])  # rotate by 180 deg in x-axis to account for difference in body frame defn.
        Rd = np.hstack((np.cross(b2d, b3d).reshape((3, 1)), b2d.reshape((3, 1)), b3d.reshape((3, 1))))
        print(f"{Rd=}")
        print(f"{Rd_traj=}")
        Wd = np.array([desired_state[15], -desired_state[16], -desired_state[17]])  # negate b2 and b3 angular velocity to account for difference in body frame defn.
        e_R = 0.5 * vee_map(Rd.T @ R - R.T @ Rd)
        e_W = W - R.T @ Rd @ Wd
        Wd_dot = self.Wd_dot  # for convenience so I don't have to repeat `self.`

        f = np.dot(-A, R @ np.array([0, 0, 1]))
        M = -self.kR*e_R - self.kW*e_W + np.cross(W, I @ W) - I @ (hat_map(W) @ R.T @ Rd @ Wd - R.T @ Rd @ Wd_dot)

        # These values match the body frame defn. in Lee et al.
        net_force_moments_matrix = np.array([[1, 1, 1, 1],
                                             [0, -L, 0, L],
                                             [L, 0, -L, 0],
                                             [-kM, kM, -kM, kM]])

        # print(f"{f=}")
        # print(f"{M=}")

        u = np.linalg.inv(net_force_moments_matrix) @ np.concatenate(([f], M))
        u = np.clip(u, -1e2, 1e2)
        # u = np.array([u[0], u[3], u[1], u[2]])  # swap propellors 2 and 4 to account for difference in body frame defn.
        print(f"{u=}")

        # Set output
        output.SetFromVector(u)