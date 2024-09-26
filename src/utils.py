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
from scipy.spatial.transform import Rotation


# Quadrotor Constants (derived from quadrotor MultibodyPlant)
m = 0.775       # quadrotor mass
L = 0.15        # distance from the center of mass to the center of each rotor in the b1, b2 plane
kM = 1e-5       # relates moment applied to quadrotor to the thurst generated
kF = 1e-3       # Force input constant
g = 9.81        # gravity
J = np.array([[1.50000000e-03, 0.00000000e+00, 2.02795951e-16],
              [0.00000000e+00, 2.50000000e-03, 0.00000000e+00],
              [2.02795951e-16, 0.00000000e+00, 3.50000000e-03]])  # Rotational Inertia

a = kF * L / np.sqrt(2)
F2W = net_force_moments_matrix = np.array([[kF, kF, kF, kF],
                                           [a, a, -a, -a],
                                           [-a, a, a, -a],
                                           [kM, -kM, kM, -kM]])

n_u = 4         # number of control inputs
n_x = 18        # number of state variables

eps = 1e-6      # help prevent divide by zero

e3 = np.array([0.,0.,1.])


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
    Convert from Euler Angles to 3x3 Rotation Matrix.
    
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


def rpy_rates_to_angular_velocity(rpy_rates, rpy_angles):
    """
    Convert roll, pitch, yaw rates to angular velocities in body-fixed frame.
    
    rpy_rates: A numpy array of shape (3,) containing [roll_rate, pitch_rate, yaw_rate].
    rpy_angles: A numpy array of shape (3,) containing [roll, pitch, yaw].
    
    Returns a numpy array of shape (3,) containing [omega_x, omega_y, omega_z].
    """
    roll_rate, pitch_rate, yaw_rate = rpy_rates
    roll, pitch, yaw = rpy_angles
    
    transformation_matrix = np.array([
        [1, 0, -np.sin(pitch)],
        [0, np.cos(roll), np.cos(pitch) * np.sin(roll)],
        [0, -np.sin(roll), np.cos(pitch) * np.cos(roll)]
    ])
    
    angular_velocity = np.dot(transformation_matrix, np.array([roll_rate, pitch_rate, yaw_rate]))
    
    return angular_velocity


def soft_clamp(x, mi, mx): 
    """
    Softly clamps the value `x` between `mi` and `mx` using a sigmoid function.
    """
    scaled = (x - mi) / (mx - mi)
    exponent = -scaled + 0.5
    base = 99999
    print(f"{exponent=}")
    return mi + (mx - mi) * sym.pow(1 + sym.pow(base, exponent), -1)


def convert_state(drone_state):
    """
    From Drake's state format to SE(3) format.

    Drake's state format is:
    [qw, qx, qy, qz, x, y, z, w1, w2, w3, x_dot, y_dot, z_dot].T

    SE(3) format is:
    [x, y, z, x_dot, y_dot, z_dot, R1, R2, R3, R4, R5, R6, R7, R8, R9, W1, W2, W3].T
    """
    x = drone_state[4:7]
    v = drone_state[10:13]
    W = drone_state[7:10]

    qw = drone_state[0]
    qx = drone_state[1]
    qy = drone_state[2]
    qz = drone_state[3]
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

    drone_state_se3 = np.concatenate((x, v, R.flatten(), W))
    return drone_state_se3


class StateConverter(LeafSystem):
    """
    Converts Drake's state output to an SE(3) format for DDP and controller to 
    use.
    """
    def __init__(self):
        LeafSystem.__init__(self)
        
        # Define input port for the drone state from drake
        # [qw, qx, qy, qz, x, y, z, w1, w2, w3, x_dot, y_dot, z_dot].T
        self.input_port_drone_state = self.DeclareVectorInputPort("drone_state", 13)
        
        # Define output port for drone state in SE(3) form:
        # [x, y, z, x_dot, y_dot, z_dot, R1, R2, R3, R4, R5, R6, R7, R8, R9, W1, W2, W3].T
        self.output_port_drone_state_se3 = self.DeclareVectorOutputPort("drone_state_se3",
                                                                           BasicVector(18),
                                                                           self.CalcOutput)
        
    def CalcOutput(self, context, output):
        """
        Simply convert the state representation and set the output.
        """
        drone_state = self.get_input_port(0).Eval(context)
        output.SetFromVector(convert_state(drone_state))


class TrajectoryDesiredStateSource(LeafSystem):
    """
    Output the desired state and acceleration based on the target trajectory and
    current time.
    """
    def __init__(self):
        LeafSystem.__init__(self)

        # Input port for x_traj, computed by ddp
        traj = AbstractValue.Make(np.array([]))
        self.input_port_traj = self.DeclareAbstractInputPort("trajectory", traj)
        
        # Define output port for desired drone state in SE(3) form:
        # [x, y, z, x_dot, y_dot, z_dot, R1, R2, R3, R4, R5, R6, R7, R8, R9, W1, W2, W3].T
        self.output_port_drone_state_se3 = self.DeclareVectorOutputPort("trajectory_desired_state",
                                                                         BasicVector(18),
                                                                         self.CalcDesiredState)
        
        # Define output port for desired drone acceleration
        self.output_port_drone_acceleration = self.DeclareVectorOutputPort("trajectory_desired_acceleration",
                                                                            BasicVector(3),
                                                                            self.CalcDesiredAccel)
        
        # Define output port for desired drone angular acceleration
        self.output_port_drone_acceleration = self.DeclareVectorOutputPort("trajectory_desired_angular_acceleration",
                                                                            BasicVector(3),
                                                                            self.CalcDesiredAngularAccel)
        
        self.n = 0
        self.traj_elapsed_time = 0


    def set_time_intervals(self, dt_array):
        self.dt_array = dt_array
        self.N = np.shape(dt_array)[0]
        

    def CalcDesiredState(self, context, output):
        """
        Simply convert the state representation and set the output
        """
        traj = self.input_port_traj.Eval(context)

        t = context.get_time()

        self.n = min(self.n, self.N)

        if t > self.traj_elapsed_time + self.dt_array[self.n]:
            self.traj_elapsed_time += self.dt_array[self.n]
            self.n = min(self.n+1, self.N-1)  # prevent out of bounds error
            # print(f"==========OUTPUTTING NEW DESIRED STATE: {traj[self.n]}==========")

        desired_state = traj[self.n]
        np.set_printoptions(precision=3)
        output.SetFromVector(desired_state)


    def CalcDesiredAccel(self, context, output):
        desired_accel = np.array([0, 0, 0])
        output.SetFromVector(desired_accel)


    def CalcDesiredAngularAccel(self, context, output):
        desired_angular_accel = np.array([0, 0, 0])
        output.SetFromVector(desired_angular_accel)