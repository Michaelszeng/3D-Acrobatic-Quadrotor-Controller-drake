from pydrake.all import (
    DiagramBuilder,
    LinearQuadraticRegulator,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    Propeller,
    PropellerInfo,
    RollPitchYaw,
    RigidTransform,
    RobotDiagramBuilder,
    ConstantVectorSource,
    SceneGraph,
    AddMultibodyPlantSceneGraph,
    Simulator,
    StartMeshcat,
    namedview,
    JointSliders
)
from underactuated.scenarios import AddFloatingRpyJoint

import numpy as np
import os
import time
import argparse
import yaml

from src.utils import *
from src.ddp import solve_trajectory, solve_trajectory_fixed_timesteps_fixed_interval
# from src.se3_leaf import SE3Controller
from src.controller import SE3Controller

meshcat = StartMeshcat()


################################################################################
##### User-Defined Constants
################################################################################
# Set initial pose of quadrotor
x0 = -1.5
y0 = 0
z0 = 1
rx0 = 0.0
ry0 = 0.0
rz0 = 0.0

pose_goal = np.array([0, 0, 0, 0, 0, 1.57])


################################################################################
##### Diagram Setup
################################################################################
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
parser = Parser(plant)
(model_instance,) = parser.AddModelsFromUrl("package://drake/examples/quadrotor/quadrotor.urdf")

# Set up the floating base type for the quadrotor.
# NOTE: THE ORDER OF ROLL PITCH YAW IN THE STATE REPRESETATION IS rz,ry,rxs
AddFloatingRpyJoint(
    plant,
    plant.GetFrameByName("base_link"),
    model_instance,
    use_ball_rpy=False
)

# Add visual quadrotor to show the desired pose of the main quadrotor
(visual_quadrotor_model_instance,) = parser.AddModelsFromUrl(f"file://{os.path.abspath('visual_quadrotor.urdf')}")
visual_quadrotor_pose = RigidTransform(RollPitchYaw(pose_goal[3:]), pose_goal[:3])

plant.Finalize()

# Set up the propellers to generate spatial force on quadrotor
body_index = plant.GetBodyByName("base_link").index()
kF = 1.0  # Force input constant
# Propellors 1 and 3 rotate CW, 2 and 4 rotate CCW
# prop_info = [
#     PropellerInfo(body_index, RigidTransform([L/np.sqrt(2), L/np.sqrt(2), 0]), kF, -kM),
#     PropellerInfo(body_index, RigidTransform([-L/np.sqrt(2), L/np.sqrt(2), 0]), kF, kM),
#     PropellerInfo(body_index, RigidTransform([-L/np.sqrt(2), -L/np.sqrt(2), 0]), kF, -kM),
#     PropellerInfo(body_index, RigidTransform([L/np.sqrt(2), -L/np.sqrt(2), 0]), kF, kM),
# ]
prop_info = [
    PropellerInfo(body_index, RigidTransform([L, 0, 0]), kF, kM),
    PropellerInfo(body_index, RigidTransform([0, L, 0]), kF, -kM),
    PropellerInfo(body_index, RigidTransform([-L, 0, 0]), kF, kM),
    PropellerInfo(body_index, RigidTransform([0, -L, 0]), kF, -kM),
]
propellers = builder.AddSystem(Propeller(prop_info))
builder.Connect(
    propellers.get_output_port(),
    plant.get_applied_spatial_force_input_port()
)
builder.Connect(
    plant.get_body_poses_output_port(),
    propellers.get_body_poses_input_port()
)

se3_controller = builder.AddSystem(SE3Controller())
state_converter = builder.AddSystem(StateConverter())
desired_state_source = builder.AddSystem(TrajectoryDesiredStateSource())
builder.Connect(
    plant.GetOutputPort("quadrotor_state"),
    state_converter.GetInputPort("drone_state")
)
builder.Connect(
    state_converter.GetOutputPort("drone_state_se3"),
    se3_controller.GetInputPort("drone_state")
)
builder.Connect(
    desired_state_source.GetOutputPort("trajectory_desired_state"),
    se3_controller.GetInputPort("x_trajectory")
)
builder.Connect(
    se3_controller.GetOutputPort("controller_output"),
    propellers.get_command_input_port()
)


### TEMPORARY: CONSTANT CONTROL INPUT = mg ###
# g = plant.gravity_field().gravity_vector()[2]
# constant_thrust_command = [-m * g / 4] * 4
# constant_input_source = builder.AddSystem(ConstantVectorSource(constant_thrust_command))
# builder.Connect(constant_input_source.get_output_port(), propellers.get_command_input_port())

MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

# Build the diagram and create an svg
diagram = builder.Build()
diagram_visualize_connections(diagram, "diagram.svg")


################################################################################
##### Simulation Setup
################################################################################
simulator = Simulator(diagram)
context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(context)
propellers_context = propellers.GetMyMutableContextFromRoot(context)
desired_state_source_context = desired_state_source.GetMyMutableContextFromRoot(context)

# Set visual quadrotor position
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("visual_quadrotor_base_link"), visual_quadrotor_pose)
visual_quadrotor_joint_idx = plant.GetJointIndices(visual_quadrotor_model_instance)[0]
visual_quadrotor_joint = plant.get_joint(visual_quadrotor_joint_idx)  # Joint object
visual_quadrotor_joint.Lock(plant_context)

# Set initial state
# plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link"), RigidTransform([0, 0, 1]))
plant.GetJointByName("x").set_translation(plant_context, x0)
plant.GetJointByName("y").set_translation(plant_context, y0)
plant.GetJointByName("z").set_translation(plant_context, z0)
plant.GetJointByName("rx").set_angle(plant_context, rx0)  # Roll
plant.GetJointByName("ry").set_angle(plant_context, ry0)  # Pitch
plant.GetJointByName("rz").set_angle(plant_context, rz0)  # Yaw

################################################################################
##### Run Trajectory Optimization
################################################################################
# Solve for trajectory
N=20  # Number of time steps in trajectory
x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace, dt, dt_array, final_translation_error, final_rotation_error = solve_trajectory(plant.get_state_output_port().Eval(plant_context), pose_goal, N)
# x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace = solve_trajectory_fixed_timesteps(plant.get_state_output_port().Eval(plant_context), pose_goal, N)

print(f"{dt=}\n")
print(f"{dt_array=}\n")
print(f"{x_trj=}\n")
print(f"{u_trj=}\n")
print(f"{cost_trace=}\n")
print(f"{regu_trace=}\n")
print(f"{redu_ratio_trace=}\n")
print(f"{redu_trace=}\n")
print(f"final_translation_error: {final_translation_error:>25}    final_rotation_error: {final_rotation_error:>25}")

# Visualize Trajectory
pos_3d_matrix = x_trj[:,:3].T
# print(f"{pos_3d_matrix.T=}")
meshcat.SetLine("ddp traj", pos_3d_matrix)

desired_state_source.set_time_intervals(dt_array)
desired_state_source.GetInputPort("trajectory").FixValue(desired_state_source_context, x_trj)


# Run the simulation
t = 0
meshcat.StartRecording()
simulator.set_target_realtime_rate(1.0)

# Testing DDP with open-loop control
# for i in range(np.shape(u_trj)[0]):
#     propellers.get_command_input_port().FixValue(propellers_context, u_trj[i])
#     t += dt
#     simulator.AdvanceTo(t)

simulator.AdvanceTo(5)

meshcat.PublishRecording()