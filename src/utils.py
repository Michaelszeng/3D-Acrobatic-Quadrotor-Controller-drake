""" Miscellaneous Utility functions """
from typing import BinaryIO, Optional, Union, Tuple
from pydrake.all import (
    Diagram,
)
import pydot

MASS = 0.775  # quadrotor mass


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