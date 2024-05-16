from pydrake.all import (
    RollPitchYaw,
    RigidTransform,
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pydrake.symbolic as sym
from scipy.linalg import expm  # matrix exponential in the sense of Lie groups

from src.utils import *

"""
Quadrotor state is represented as the (18,) vector:
[x, y, z, x_dot, y_dot, z_dot, R1, R2, R3, R4, R5, R6, R7, R8, R9, W1, W2, W3].T
"""

def continuous_dynamics(x, u):
    """
    Dynamics equation based on https://arxiv.org/pdf/1003.2005.

    u is a (4,) np vector of propeller forces.

    x is the (18,) np vector containing the current state.

    Notation:
     - p = position vector \in R3
     - W = angular velocity \in R3
    """    
    # [f, M1, M2, M3].T
    # net_force_moments_vector = np.array([[1, 1, 1, 1],
    #                                      [L/np.sqrt(2), L/np.sqrt(2), -L/np.sqrt(2), -L/np.sqrt(2)],
    #                                      [-L/np.sqrt(2), L/np.sqrt(2), L/np.sqrt(2), -L/np.sqrt(2)],
    #                                      [-kM, kM, -kM, kM]]) @ u
    net_force_moments_vector = np.array([[1, 1, 1, 1],
                                         [0, L, 0, -L],
                                         [-L, 0, L, 0],
                                         [kM, -kM, kM, -kM]]) @ u

    # scalar net force in -b3 direction 
    f = net_force_moments_vector[0]
    # (3,1) moment vector 
    M = net_force_moments_vector[1:]

    R = x[6:15].reshape(3, 3)  # Rotation
    v = x[3:6]  # Linear Velocity
    W = x[15:]  # Angular Velocity ((3,) Vector)

    # Calculate x_dot
    p_dot = v                                                                   # Linear Velocity
    v_dot = np.array([[0],[0],[g]]) + (f * R @ np.array([[0],[0],[1]]))/m       # Linear Acceleration (due to gravity & propellors)
    R_dot = R @ hat_map(W)                                                      # Rotational Velocity
    W_dot = np.linalg.inv(I) @ (M - np.cross(W, I @ W))                         # Angular Acceleration

    # Also return the current rotation and angular velocity to do exponential map method of integrating rotation
    return np.concatenate((p_dot, v_dot.flatten(), R_dot.flatten(), W_dot)), R, W


def compute_discrete_dynamics_time_step(n, dt_nominal):
    # https://www.desmos.com/calculator/vegofqyne7
    beta = 5
    dt = -1/(beta*(n+0.5) + (1/dt_nominal)) + dt_nominal
    return dt


def discrete_dynamics(x, u, n, dt_nominal):
    """
    Calculates next state based on current state and control input, using
    continuous dynamics.

    u is a (4,) np vector of propeller forces.

    x is the (18,) np vector containing the current state.
    """
    dt = compute_discrete_dynamics_time_step(n, dt_nominal)

    x_dot, R, W = continuous_dynamics(x, u)

    # Precise integration of rotation matrix to preserve orthogonality and det=1 using Exponential Map
    W_norm = np.sqrt(np.dot(W, W) + eps)
    theta = W_norm * dt  # change in rotation over time step
    k = W / W_norm  # axis of rotation
    K_hat = hat_map(k)  # convert axis of rotation to skew symmetric matrix
    # Rodrigues' Formula
    R_new = R @ (np.eye(3) + np.sin(theta) * K_hat + (1 - np.cos(theta)) * np.dot(K_hat, K_hat))

    # print(f"{R_new=}")

    # Euler Integration for other components of state
    return np.concatenate((x[:6]+x_dot[:6]*dt, R_new.flatten(), x[15:]+x_dot[15:]*dt))


def dynamics_rollout(x0, u_trj, dt):
    """
    Computes x_trj, the state trajectory, given a sequence of control inputs.

    x0 is the initial state vector.

    u_trj is a (N,4) array containing sequence of control inputs in the trajectory.
    """
    N = np.shape(u_trj)[0]

    x_trj = [x0]
    for i in range(N):
        x_trj.append(discrete_dynamics(x_trj[-1], u_trj[i], i, dt))

    return np.array(x_trj)


def trajectory_cost(pose_goal, x, u, n, N, beta=10):
    """
    Goal of the cost function is to reach the goal pose while minimizing energy.
    
    pose_goal is a [x, y, z, R, P, Y] vector

    x is the (18,) np vector containing the current state

    u is the (4,) np vector containing control inputs
    """
    scale = np.exp((beta * (n - N)) / N)  # Exponential scaling factor

    energy_cost = np.dot(u, u)
    # energy_cost = 0

    translation_error_cost = np.dot(x[:3] - pose_goal[:3], x[:3] - pose_goal[:3])

    R = x[6:15].reshape(3, 3)
    R_goal = euler_to_rotation_matrix(pose_goal[3:])

    # Error based on magnitude of angle difference between two rotations
    # R_relative = R @ R_goal.T
    # print(f"{R_relative}")
    # trace_R = np.trace(R_relative)
    # trace_R = soft_clamp(trace_R, -1, 3)  # Softly clamp trace between -1 and 3 so that input to arccos is within its domain [-1, 1]
    # if trace_R.GetVariables().empty():  # Convert from symbolic expression to float value
    #     trace_R = trace_R.Evaluate()
    # rotation_error = np.arccos((trace_R - 1) / 2)

    # Error from taking norm of e_R (equation (8)) from Lee et al.
    rotation_error_vec = 0.5 * vee_map(R_goal.T @ R - R.T @ R_goal)
    rotation_error = np.sqrt(np.dot(rotation_error_vec, rotation_error_vec))  # compute norm

    # rotation_error = 0
    rotation_error_cost = np.dot(rotation_error, rotation_error)

    # try:
    #     print(f"energy_cost: {energy_cost:>25}    translation_error_cost: {translation_error_cost:>25}    rotation_error_cost: {rotation_error_cost:>25}")
    # except:
    #     pass

    return scale * (0.001*energy_cost + 0.1*translation_error_cost + 0.1*rotation_error_cost)


def terminal_cost(pose_goal, x, N):
    """
    Terminal cost is the distance to the goal.
    """
    return 5*trajectory_cost(pose_goal, x, np.zeros(4), N, N)


def cost_trj(pose_goal, x_trj, u_trj):
    N = np.shape(x_trj)[0]

    total = 0
    
    # sum cost at each action step
    for i in range(N-1):
        total += trajectory_cost(pose_goal, x_trj[i], u_trj[i], i, N)

    total += terminal_cost(pose_goal, x_trj[N-1], N)  # add final cost
    return total


class derivatives:
    def __init__(self, pose_goal, discrete_dynamics, trajectory_cost, terminal_cost, N, dt):
        n_x = 18
        n_u = 4
        self.x_sym = np.array([sym.Variable("x_{}".format(i)) for i in range(n_x)])
        self.u_sym = np.array([sym.Variable("u_{}".format(i)) for i in range(n_u)])
        self.n_sym = np.array([sym.Variable("n")])
        x = self.x_sym
        u = self.u_sym
        n = self.n_sym
        self.N = N

        l = trajectory_cost(pose_goal, x, u, n, N)
        self.l_x = sym.Jacobian([l], x).ravel()
        self.l_u = sym.Jacobian([l], u).ravel()
        self.l_xx = sym.Jacobian(self.l_x, x)
        self.l_ux = sym.Jacobian(self.l_u, x)
        self.l_uu = sym.Jacobian(self.l_u, u)

        l_final = terminal_cost(pose_goal, x, N)
        self.l_final_x = sym.Jacobian([l_final], x).ravel()
        self.l_final_xx = sym.Jacobian(self.l_final_x, x)

        f = discrete_dynamics(x, u, n, dt)
        self.f_x = sym.Jacobian(f, x)
        self.f_u = sym.Jacobian(f, u)

    def stage(self, x, u, n):
        # Populate real values of x and u into symbolic variable
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})
        env.update({self.n_sym[0]: n})

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


def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
    """
    Compute partial derivatives of the Q-function.

    np.shape(l_x): (18,)
    np.shape(l_u): (4,)
    np.shape(l_xx): (18, 18)
    np.shape(l_ux): (4, 18)
    np.shape(l_uu): (4, 4)
    np.shape(f_x): (18, 18)
    np.shape(f_u): (18, 4)
    np.shape(V_x): (18,)
    np.shape(V_xx): (18, 18)
    """

    # print(f"np.shape(l_x): {np.shape(l_x)}")
    # print(f"np.shape(l_u): {np.shape(l_u)}")
    # print(f"np.shape(l_xx): {np.shape(l_xx)}")
    # print(f"np.shape(l_ux): {np.shape(l_ux)}")
    # print(f"np.shape(l_uu): {np.shape(l_uu)}")
    # print(f"np.shape(f_x): {np.shape(f_x)}")
    # print(f"np.shape(f_u): {np.shape(f_u)}")
    # print(f"np.shape(V_x): {np.shape(V_x)}")
    # print(f"np.shape(V_xx): {np.shape(V_xx)}")

    Q_x = l_x.T + V_x.T @ f_x
    Q_u = l_u.T + V_x.T @ f_u
    Q_xx = l_xx + f_x.T @ V_xx @ f_x
    Q_ux = l_ux + f_u.T @ V_xx @ f_x   # also works: (l_ux.T + f_x.T @ V_xx @ f_u).T
    Q_uu = l_uu + f_u.T @ V_xx @ f_u
    return Q_x, Q_u, Q_xx, Q_ux, Q_uu


def V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
    """
    Computes partial derivatives of Value function. These equations are derived
    from the fasct that V(x[n]) = Q(x[u], u[n]*). Then, by taking a 2nd order
    Taylor approximation of both sides of the equation, we can express V_x and
    V_xx in terms of the partials of Q by matching the dx[n] terms.

    https://github.com/Michaelszeng/6.8210-Underactuated-Robotics-Notes/blob/5de0106c80cd37a132fa1903d7812a3a59eecaa1/5)%20Trajectory%20Optimization.md?plain=1#L358

    np.shape(Q_x): (18,)
    np.shape(Q_u): (4,)
    np.shape(Q_xx): (18, 18)
    np.shape(Q_ux): (4, 18)
    np.shape(Q_uu): (4, 4)
    np.shape(K): (4, 18)
    np.shape(k): (4,)
    """

    # print(f"np.shape(Q_x): {np.shape(Q_x)}")
    # print(f"np.shape(Q_u): {np.shape(Q_u)}")
    # print(f"np.shape(Q_xx): {np.shape(Q_xx)}")
    # print(f"np.shape(Q_ux): {np.shape(Q_ux)}")
    # print(f"np.shape(Q_uu): {np.shape(Q_uu)}")
    # print(f"np.shape(K): {np.shape(K)}")
    # print(f"np.shape(k): {np.shape(k)}")

    #18x1  1x18    4x1  2x5  1x2   2x5    1x2   2x2   2x5
    V_x = (Q_x.T + Q_u.T @ K + k.T @ Q_ux + k.T @ Q_uu @ K).T

    #5x5   5x5    5x2     2x5  5x2   2x5    5x2   2x2   2x5
    V_xx = Q_xx + Q_ux.T @ K + K.T @ Q_ux + K.T @ Q_uu @ K

    return V_x, V_xx


def gains(Q_uu, Q_u, Q_ux):
    """
    Computes feedforward gains k and K 
    """
    Q_uu_inv = np.linalg.inv(Q_uu)
    k = -Q_uu_inv @ Q_u
    K = -Q_uu_inv @ Q_ux
    return k, K


def expected_cost_reduction(Q_u, Q_uu, k):
    """
    Expected cost reduction of an of back/forward passes.
    """
    return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))


def forward_pass(x_trj, u_trj, k_trj, K_trj, dt):
    """
    Apply optimal control actions computed in the back pass to find a new
    nominal x and u trajectory.
    """
    N = u_trj.shape[0]

    x_trj_new = np.zeros(x_trj.shape)
    u_trj_new = np.zeros(u_trj.shape)

    x_trj_new[0, :] = x_trj[0, :]  # Same initial state as nominal trajectory

    # Iterate through all time steps in trajectory
    for n in range(N):
        # Compute deviations from desired trajectory
        dx = x_trj_new[n] - x_trj[n]
        du = k_trj[n] + K_trj[n] @ dx  #d u[n]* = k + K @ dx[n]

        u_trj_new[n,:] = u_trj[n,:] + du
        x_trj_new[n+1,:] = discrete_dynamics(x_trj_new[n], u_trj_new[n], n, dt)

    return x_trj_new, u_trj_new


def backward_pass(derivs, x_trj, u_trj, regu):
    """
    Given nominal trajectory x_trj, compute the optimal control policy k_trj and
    K_trj at each time step in the trajectory.
    """
    # Initialize arrays to hold k and K
    k_trj = np.zeros([u_trj.shape[0], u_trj.shape[1]])
    K_trj = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])

    # Initialize expected cost reduction from this iteration of back/forward pass to zero
    expected_cost_redu = 0

    # Set terminal boundary condition (V_x, V_xx)
    V_x, V_xx = derivs.final(x_trj[np.shape(x_trj)[0]-1])

    # Reverse iterate from trajectory end to start
    for n in range(u_trj.shape[0] - 1, -1, -1):
        l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = derivs.stage(x_trj[n], u_trj[n], n)
        Q_x, Q_u, Q_xx, Q_ux, Q_uu = Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)

        # Add regularization to ensure that Q_uu is invertible and nicely conditioned
        Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0]) * regu
        
        # Calculate control input gains
        k, K = gains(Q_uu_regu, Q_u, Q_ux)
        k_trj[n, :] = k
        K_trj[n, :, :] = K

        # Calculate Value function derivatives to be used next iteration
        V_x, V_xx = V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)

        expected_cost_redu += expected_cost_reduction(Q_u, Q_uu, k)
    return k_trj, K_trj, expected_cost_redu


def solve_trajectory_fixed_timesteps_fixed_interval(x0, pose_goal, N, dt, max_iter=200, regu_init=100):
    """
    x0 is the Drake default [x, y, z, R, P, Y, x_dot, y_dot, z_dot, R_dot, P_dot, Y_dot] state vector.

    pose_goal is a [x, y, z, R, P, Y] vector.
    """
    # First, convert Drake initial state representation to SE(3) form [x, y, z, x_dot, y_dot, z_dot, R1, R2, R3, R4, R5, R6, R7, R8, R9, W1, W2, W3].T
    R0 = euler_to_rotation_matrix(x0[5:2:-1])  # NOTE: THE ORDER OF ROLL PITCH YAW IN THE STATE REPRESETATION IS rz,ry,rxs
    W0 = rpy_rates_to_angular_velocity(x0[11:8:-1], x0[5:2:-1])  # NOTE: THE ORDER OF ROLL PITCH YAW IN THE STATE REPRESETATION IS rz,ry,rxs
    x0 = np.concatenate((x0[:3], x0[6:9], R0.flatten(), W0))

    Rf = euler_to_rotation_matrix(pose_goal[3:6])
    xf = np.concatenate((pose_goal[:3], np.zeros(3), Rf.flatten(), np.zeros(3)))

    derivs = derivatives(pose_goal, discrete_dynamics, trajectory_cost, terminal_cost, N, dt)

    # u_trj = np.random.randn(N - 1, n_u) * 0.0001
    # x_trj = dynamics_rollout(x0, u_trj)
    # l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = derivs.stage(x_trj[0], u_trj[0])
    # V_x, V_xx = derivs.final(x_trj[np.shape(x_trj)[0]-1])
    # Q_x, Q_u, Q_xx, Q_ux, Q_uu = Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
    # print(Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx))
    # k, K = gains(Q_uu, Q_u, Q_ux)
    # print(gains(Q_uu, Q_u, Q_ux))
    # print(V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k))


    # Initial Guesses for trajectory
    u_trj = np.ones((N - 1, n_u)) * (-m * g / 4)
    # u_trj = np.random.randn(N - 1, n_u) * 0.0001
    x_trj = dynamics_rollout(x0, u_trj, dt)
    total_cost = cost_trj(pose_goal, x_trj, u_trj)

    regu = regu_init
    max_regu = 10000
    min_regu = 0.01

    # Setup traces
    cost_trace = [total_cost]
    expected_cost_redu_trace = []
    redu_ratio_trace = [1]
    redu_trace = []
    regu_trace = [regu]

    # Run main loop
    for i in range(max_iter):
        # Backward and forward pass
        k_trj, K_trj, expected_cost_redu = backward_pass(derivs, x_trj, u_trj, regu)
        x_trj_new, u_trj_new = forward_pass(x_trj, u_trj, k_trj, K_trj, dt)

        # Evaluate new trajectory
        total_cost = cost_trj(pose_goal, x_trj_new, u_trj_new)
        cost_redu = cost_trace[-1] - total_cost
        redu_ratio = cost_redu / abs(expected_cost_redu)

        # Accept or reject iteration
        if cost_redu > 0:
            # Improvement! Accept new trajectories and lower regularization
            redu_ratio_trace.append(redu_ratio)
            cost_trace.append(total_cost)
            x_trj = x_trj_new
            u_trj = u_trj_new
            regu *= 0.7
        else:
            # Reject new trajectories and increase regularization
            regu *= 2.0
            cost_trace.append(cost_trace[-1])
            redu_ratio_trace.append(0)

        regu = min(max(regu, min_regu), max_regu)
        regu_trace.append(regu)
        redu_trace.append(cost_redu)

        # Early termination if expected improvement is small
        if expected_cost_redu <= 1e-6:
            print("Expected cost reduction too low. Terminating early.")
            break

    return x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace


def solve_trajectory(x0, pose_goal, N, num_dt_to_search=3):
    """
    Run iLQR to generat an optimal trajectory, while performing Linear Search to
    find optimal time interval dt.
    """
    min_cost = np.inf
    min_cost_traj = []
    min_cost_time_interval = 0
    # for i in np.linspace(0.025, 0.1, num_dt_to_search):
    for i in [0.1]:
        x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace = solve_trajectory_fixed_timesteps_fixed_interval(x0, pose_goal, N, i)
        print(f"=====================================cost (dt={i}): {cost_trace[-1]}=====================================")
        if cost_trace[-1] <= min_cost:
            print(f"step {i}: {cost_trace[-1]} <= current minimum: {min_cost}")
            min_cost = cost_trace[-1]
            min_cost_traj = [x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace]
            min_cost_time_interval = i

    # Compute errors at the very end of the trajectory
    final_translation_error = np.linalg.norm(x_trj[-1][:3] - pose_goal[:3])
    R = x_trj[-1][6:15].reshape(3, 3)
    R_goal = euler_to_rotation_matrix(pose_goal[3:])
    # Error from taking norm of e_R (equation (8)) from Lee et al.
    final_rotation_error = np.linalg.norm(0.5 * vee_map(R_goal.T @ R - R.T @ R_goal))

    # Generate list of time steps corresponding to each x and u in the trajectory
    dt_array = []
    for n in range(N):
       dt_array.append(compute_discrete_dynamics_time_step(n, min_cost_time_interval))

    return *min_cost_traj, min_cost_time_interval, dt_array, final_translation_error, final_rotation_error


def make_basic_test_traj(x0, N, dt=0.1):
    u_trj = np.ones((N - 1, n_u)) * (-m * g / 4)
    u_trj[:, 1] += 0.1  # make prop 3 spin faster to propel quadrotor forward
    x_trj = dynamics_rollout(x0, u_trj, dt)

    # Generate list of time steps corresponding to each x and u in the trajectory
    dt_array = []
    for n in range(N):
       dt_array.append(compute_discrete_dynamics_time_step(n, dt))

    return x_trj, u_trj