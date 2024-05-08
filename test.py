import pydrake.symbolic as sym
import numpy as np
from src.utils import *

def soft_clamp(x, mi, mx): 
    """
    Softly clamps the value `x` between `mi` and `mx` using a sigmoid function.
    """

    # return mi + (mx-mi)*(lambda t: (1+200**(-t+0.5))**(-1) )( (x-mi)/(mx-mi) )

    scaled = (x - mi) / (mx - mi)
    exponent = -scaled + 0.5
    base = 99999
    return mi + (mx - mi) * sym.pow(1 + sym.pow(base, exponent), -1)


# print(soft_clamp(-234, -1, 3))


R = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
# R_goal = np.array([[1, 0, 0],
#                    [0, 0, -1],
#                    [0, 1, 0]])
theta = 0.1  # Rotation angle in radians
R_goal = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])

R_relative = R @ R_goal.T
trace_R = np.trace(R_relative)
print(f"trace_R: {trace_R}")
trace_R = soft_clamp(trace_R, -1, 3)  # Softly clamp trace between -1 and 3 so that input to arccos is within its domain [-1, 1]
if trace_R.GetVariables().empty():  # Convert from symbolic expression to float value
    trace_R = trace_R.Evaluate()
print(f"trace_R: {trace_R}")
rotation_error = np.arccos((trace_R - 1) / 2)

print(rotation_error)

rotation_error_cost = np.dot(rotation_error, rotation_error)

print(rotation_error_cost)