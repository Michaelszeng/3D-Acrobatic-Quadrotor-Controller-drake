""" Miscellaneous Utility functions """
from pydrake.all import (
    Diagram,
    LeafSystem,
    BasicVector,
    AbstractValue,
)
import pydrake.symbolic as sym

from typing import BinaryIO, Optional, Union, Tuple
import pydot
import numpy as np


# Quadrotor Constants (derived from quadrotor MultibodyPlant)
m = 0.775       # quadrotor mass
L = 0.15        # distance from the center of mass to the center of each rotor in the b1, b2 plane
kM = 0.0245     # relates moment applied to quadrotor to the thurst generated
g = -9.81       # gravity
I = np.array([[1.50000000e-03, 0.00000000e+00, 2.02795951e-16],
              [0.00000000e+00, 2.50000000e-03, 0.00000000e+00],
              [2.02795951e-16, 0.00000000e+00, 3.50000000e-03]])  # Rotational Inertia

n_u = 4         # number of control inputs
n_x = 18        # number of state variables

eps = 1e-6      # help prevent divide by zero


def diagram_visualize_connections(diagram: Diagram, file: Union[BinaryIO, str]) -> None:
    """
    Create SVG file of system diagram.
    """
    if type(file) is str:
        file = open(file, "bw")
    graphviz_str = diagram.GetGraphvizString()
    svg_data = pydot.graph_from_dot_data(
        diagram.GetGraphvizString())[0].create_svg()
    file.write(svg_data)


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


def vee_map(S):
    """
    Convenience function to perform the vee map operation. The vee map of a
    skew-symmetric matrix S, denoted by vec(S), is an operation that converts the
    matrix into a 3D vector. This function assumes that S is a 3x3 skew-symmetric
    matrix and returns the corresponding 3D vector.
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def euler_to_rotation_matrix(angles):
    """
    Convert from Euler Angles to 3x3 Rotation Matrix using Drake Symbolic.
    
    `angles` is a (3,) np array containing [R, P, Y]
    """
    roll, pitch, yaw = angles

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Calculate the rotation matrix
    R_roll = np.array([[1, 0, 0],
                       [0, cos_roll, -sin_roll],
                       [0, sin_roll, cos_roll]])

    R_pitch = np.array([[cos_pitch, 0, sin_pitch],
                        [0, 1, 0],
                        [-sin_pitch, 0, cos_pitch]])

    R_yaw = np.array([[cos_yaw, -sin_yaw, 0],
                      [sin_yaw, cos_yaw, 0],
                      [0, 0, 1]])

    # Combine the rotation matrices
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))

    return R


def soft_clamp(x, mi, mx): 
    """
    Softly clamps the value `x` between `mi` and `mx` using a sigmoid function.
    """
    scaled = (x - mi) / (mx - mi)
    exponent = -scaled + 0.5
    base = 99999
    print(f"{exponent=}")
    return mi + (mx - mi) * sym.pow(1 + sym.pow(base, exponent), -1)


class StateConverter(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        
        # Define input port for the drone state from drake
        # [x, y, z, R, P, Y, x_dot, y_dot, z_dot, R_dot, P_dot, Y_dot].T
        self.input_port_drone_state = self.DeclareInputPort("drone_state", 
                                                             BasicVector(12))
        
        # Define output port for drone state in SE(3) form:
        # [x, y, z, x_dot, y_dot, z_dot, R1, R2, R3, R4, R5, R6, R7, R8, R9, W1, W2, W3].T
        self.output_port_drone_state_se3 = self.DeclareVectorOutputPort("drone_state_se3",
                                                                           BasicVector(18),
                                                                           self.CalcOutput)
        
    def CalcOutput(self, context, output):
        """
        Simply convert the state representation and set the output
        """
        # Retrieve input data from input ports
        drone_state = self.input_port_drone_state.Eval(context)

        R = euler_to_rotation_matrix(drone_state[3:6])
        drone_state_se3 = np.concatenate((drone_state[:3], drone_state[6:9], R.flatten(0), drone_state[9:]))

        # Set output
        output.SetFromVector(drone_state_se3)



class TrajectoryDesiredStateSource(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)

        # Input port for x_traj, computed by ddp
        traj = AbstractValue.Make(np.array([]))
        self.DeclareAbstractInputPort("trajectory", traj)
        
        # Define output port for desired drone state in SE(3) form:
        # [x, y, z, x_dot, y_dot, z_dot, R1, R2, R3, R4, R5, R6, R7, R8, R9, W1, W2, W3].T
        self.output_port_drone_state_se3 = self.DeclareVectorOutputPort("trajectory_desired_state",
                                                                           BasicVector(18),
                                                                           self.CalcOutput)
        
        self.dt_array = None
        self.n = 0
        self.traj_elapsed_time = 0


    def set_time_intervals(self, dt_array):
        self.dt_array = dt_array
        self.N = np.shape(dt_array)[0]
        

    def CalcOutput(self, context, output):
        """
        Simply convert the state representation and set the output
        """
        traj = self.get_input_port(0).Eval(context)

        t = context.get_time()

        self.n = min(self.n, self.N)

        if t > self.traj_elapsed_time + self.dt_array[self.n]:
            self.traj_elapsed_time += self.dt_array[self.n]
            self.n += 1

        desired_state = traj[self.n]
        output.SetFromVector(desired_state)