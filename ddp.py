import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pydrake.symbolic as sym

from src.utils import *


def discrete_dynamics(plant, plant_context, x, u):
    """
    Calculates next state based on current state and control input.

    Dynamics equation based on https://arxiv.org/pdf/1003.2005.

    u is a (4,) np vector of propeller forces.

    x is the (12,) np vector containing the current state.
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
    f = net_force_moments_vector[0,0]
    # (3,1) moment vector 
    M = net_force_moments_vector[1:]
    
    # Rotation matrix from body-fixed frame to world frame
    W_R_B = plant.CalcRelativeRotationMatrix(plant_context, 
                                             plant.world_frame(), 
                                             plant.GetFrameByName("base_link"))

    # Get quadrotor's Inertia matrix in body-fixed frame
    I = plant.CalcSpatialInertia(plant_context, plant.GetFrameByName("base_link"), 0)  # body index 0
    g = plant.gravity_field().gravity_vector()[2]

    q_dot = x[6:9]
    q_ddot = np.array([[0],[0],[g]]) - f * W_R_B @ np.array([[0],[0],[1]])
    



def angular_distance(angle_diff):
    """Calculate the minimum distance between two angles."""
    return 180 - abs(abs(angle_diff) - 180)


def trajectory_cost(pose_goal, x, u):
    """
    Goal of the cost function is to reach the goal pose while minimizing energy.
    """
    energy_cost = u**2
    translation_error_cost = (x[:3] - pose_goal[:3])**2
    rotation_error_cost = np.apply_along_axis(angular_distance, 0, x[3:6] - pose_goal[3:])
    return energy_cost + translation_error_cost + rotation_error_cost


def terminal_cost(pose_goal, x):
    """
    Terminal cost is the distance to the goal.
    """
    translation_error_cost = (x[:3] - pose_goal[:3])**2
    rotation_error_cost = np.apply_along_axis(angular_distance, 0, x[3:6] - pose_goal[3:])
    return translation_error_cost + rotation_error_cost


def cost_trj(x_trj, u_trj):
    N = np.shape(x_trj)[0]

    total = 0
    
    # sum cost at each action step
    for i in range(N-1):
        total += trajectory_cost(x_trj[i], u_trj[i])

    total += terminal_cost(x_trj[N-1])  # add final cost
    return total


class derivatives:
    def __init__(self, plant, plant_context, discrete_dynamics, cost_stage, cost_final):
        n_x = 12
        n_u = 4
        self.x_sym = np.array([sym.Variable("x_{}".format(i)) for i in range(n_x)])
        self.u_sym = np.array([sym.Variable("u_{}".format(i)) for i in range(n_u)])
        x = self.x_sym
        u = self.u_sym

        l = cost_stage(x, u)
        self.l_x = sym.Jacobian([l], x).ravel()
        self.l_u = sym.Jacobian([l], u).ravel()
        self.l_xx = sym.Jacobian(self.l_x, x)
        self.l_ux = sym.Jacobian(self.l_u, x)
        self.l_uu = sym.Jacobian(self.l_u, u)

        l_final = cost_final(x)
        self.l_final_x = sym.Jacobian([l_final], x).ravel()
        self.l_final_xx = sym.Jacobian(self.l_final_x, x)

        f = discrete_dynamics(plant, plant_context, x, u)
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


derivs = derivatives(discrete_dynamics, trajectory_cost, terminal_cost)


def solve_trajectory(plant):
    pass