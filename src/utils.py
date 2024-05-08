""" Miscellaneous Utility functions """
from pydrake.all import (
    Diagram,
)
import pydrake.symbolic as sym

from typing import BinaryIO, Optional, Union, Tuple
import pydot
import numpy as np


m = 0.775       # quadrotor mass
L = 0.15        # distance from the center of mass to the center of each rotor in the b1, b2 plane
kM = 0.0245     # relates propellers' torque generated to thrust generated
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
    return mi + (mx - mi) * sym.pow(1 + sym.pow(base, exponent), -1)