""" Miscellaneous Utility functions """
from typing import BinaryIO, Optional, Union, Tuple
from pydrake.all import (
    Diagram,
)
import pydot

m = 0.775                    # quadrotor mass
L = 0.15                        # distance from the center of mass to the center of each rotor in the b1, b2 plane
kM = 0.0245                     # relates propellers' torque generated to thrust generated


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