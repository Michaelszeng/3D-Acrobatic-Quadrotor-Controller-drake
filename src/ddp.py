from pydrake.all import (
    RollPitchYaw,
    RigidTransform,
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pydrake.symbolic as sym

from src.utils import *

"""
Quadrotor state is represented as:

[x, y, z, x_dot, y_dot, z_dot, R1, R2, R3, R4, R5, R6, R7, R8, R9, W1, W2, W3].T
"""

def continuous_dynamics(x, u):
    """
    Dynamics equation based on https://arxiv.org/pdf/1003.2005.

    u is a (4,) np vector of propeller forces.

    x is the (12,) np vector containing the current state in the form:

    [x, y, z, R, P, Y, x_dot, y_dot, z_dot, R_dot, P_dot, Y_dot].T

    Notation:
     - p = position vector \in R3
     - W = angular velocity \in R3
    """
    def hat_map(v):
        """
        Convenience function to perform the hat map operation.The hat map of 
        $x$, $\hat{x}$, is simply a convenience linear operator that expresses 
        the 3D vector $x$ as a "skew-symmetric" 3x3 matrix. This 3x3 matrix can 
        be used to apply angular velocities to a rotation matrix, or to perform 
        cross products using just matrix multiplicaton (i.e. $\hat{x}y = x 
        \times y$)
        """
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
    
    # [f, M1, M2, M3].T
    net_force_moments_vector = np.array([[1, 1, 1, 1],
                                         [0, -L, 0, L],
                                         [L, 0, -L, 0],
                                         [-kM, kM, -kM, kM]]) @ u

    # scalar net force in -b3 direction 
    f = net_force_moments_vector[0]
    # (3,1) moment vector 
    M = net_force_moments_vector[1:]
    
    # Should be equivalent method of calculating Rotation matrix from body-fixed frame to world frame
    R = euler_to_rotation_matrix(np.array([x[3], x[4], x[5]]))

    v = x[6:9]  # Linear Velocity
    W = x[9:]   # Angular Velocity ((3,) Vector)

    # Calculate x_dot
    p_dot = v                                                                   # Linear Velocity
    v_dot = np.array([[0],[0],[g]]) - (f * R @ np.array([[0],[0],[1]]))/m       # Linear Acceleration (due to gravity & propellors)
    print(np.shape(v_dot.flatten()))
    R_dot = R @ hat_map(W)                                                      # Rotational Velocity
    W_dot = np.linalg.inv(I) @ (M - np.cross(W, I @ W))                         # Angular Acceleration

    return np.concatenate((p_dot, v_dot.flatten(), R_dot.flatten(), W_dot))


def discrete_dynamics(x, u):
    """
    Calculates next state based on current state and control input, using
    continuous dynamics.
    """
    dt = 0.1

    x_dot = continuous_dynamics(x, u)

    # Reformat x to be in the form [x, y, z, x_dot, y_dot, z_dot, R1, R2, R3, R4, R5, R6, R7, R8, R9, W1, W2, W3].T
    R = euler_to_rotation_matrix(np.array([x[3], x[4], x[5]]))
    x = np.concatenate((x[0:3], x[6:9], R.flatten(), x[9:]))

    return x_dot*dt + x


def trajectory_cost(pose_goal, x, u):
    """
    Goal of the cost function is to reach the goal pose while minimizing energy.
    
    pose_goal is a [x, y, z, R, P, Y] vector

    x is the (12,) np vector containing the current state

    u is the (4,) np vector containing control inputs
    """
    energy_cost = np.dot(u, u)

    translation_error_cost = np.dot(x[:3] - pose_goal[:3], x[:3] - pose_goal[:3])

    rotation_error = np.array([angular_distance(x[3] - pose_goal[3]),
                               angular_distance(x[4] - pose_goal[4]),
                               angular_distance(x[5] - pose_goal[5])])

    rotation_error_cost = np.dot(rotation_error, rotation_error)

    return energy_cost + translation_error_cost + rotation_error_cost


def terminal_cost(pose_goal, x):
    """
    Terminal cost is the distance to the goal.
    """
    return trajectory_cost(pose_goal, x, np.zeros(4))


def cost_trj(x_trj, u_trj):
    N = np.shape(x_trj)[0]

    total = 0
    
    # sum cost at each action step
    for i in range(N-1):
        total += trajectory_cost(x_trj[i], u_trj[i])

    total += terminal_cost(x_trj[N-1])  # add final cost
    return total


class derivatives:
    def __init__(self, pose_goal, discrete_dynamics, trajectory_cost, terminal_cost):
        n_x = 12
        n_u = 4
        self.x_sym = np.array([sym.Variable("x_{}".format(i)) for i in range(n_x)])
        self.u_sym = np.array([sym.Variable("u_{}".format(i)) for i in range(n_u)])
        x = self.x_sym
        u = self.u_sym

        l = trajectory_cost(pose_goal, x, u)
        self.l_x = sym.Jacobian([l], x).ravel()
        self.l_u = sym.Jacobian([l], u).ravel()
        self.l_xx = sym.Jacobian(self.l_x, x)
        self.l_ux = sym.Jacobian(self.l_u, x)
        self.l_uu = sym.Jacobian(self.l_u, u)

        l_final = terminal_cost(pose_goal, x)
        self.l_final_x = sym.Jacobian([l_final], x).ravel()
        self.l_final_xx = sym.Jacobian(self.l_final_x, x)

        f = discrete_dynamics(x, u)
        self.f_x = sym.Jacobian(f, x)
        self.f_u = sym.Jacobian(f, u)

    def stage(self, x, u):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})

        l_x = sym.Evaluate(self.l_x, env).ravel()
        l_u = sym.Evaluate(self.l_u, env).ravel()
        l_xx = sym.Evaluate(self.l_xx, env)
        l_ux = sym.Evaluate(self.l_ux, env)
        l_uu = sym.Evaluate(self.l_uu, env)

        f_x = sym.Evaluate(self.f_x, env)
        f_u = sym.Evaluate(self.f_u, env)

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u

    def final(self, x):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}

        l_final_x = sym.Evaluate(self.l_final_x, env).ravel()
        l_final_xx = sym.Evaluate(self.l_final_xx, env)

        return l_final_x, l_final_xx


def solve_trajectory(plant, plant_context, pose_goal):
    """
    pose_goal is a [x, y, z, R, P, Y] vector
    """
    derivs = derivatives(pose_goal, discrete_dynamics, trajectory_cost, terminal_cost)