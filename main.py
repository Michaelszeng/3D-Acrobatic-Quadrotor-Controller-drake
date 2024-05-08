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
from pydrake.examples import QuadrotorGeometry, QuadrotorPlant, StabilizingLQRController
from underactuated.scenarios import AddFloatingRpyJoint

import numpy as np
import os
import time
import argparse
import yaml

from src.utils import *
from src.ddp import solve_trajectory, solve_trajectory_fixed_timesteps

# Start Meshcat visualizer server.
meshcat = StartMeshcat()

################################################################################
##### Digram Setup
################################################################################
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
parser = Parser(plant)
(model_instance,) = parser.AddModelsFromUrl(
    "package://drake/examples/quadrotor/quadrotor.urdf"
)

# Set up the floating base type for the quadrotor.
AddFloatingRpyJoint(
    plant,
    plant.GetFrameByName("base_link"),
    model_instance,
    use_ball_rpy=False
)
plant.Finalize()

# Set up the propellers to generate spatial force on quadrotor
body_index = plant.GetBodyByName("base_link").index()
kF = 1.0  # Force input constant
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


### TEMPORARY: CONSTANT CONTROL INPUT = mg ###
g = plant.gravity_field().gravity_vector()[2]
constant_thrust_command = [-m * g / 4] * 4
constant_input_source = builder.AddSystem(ConstantVectorSource(constant_thrust_command))
builder.Connect(constant_input_source.get_output_port(), propellers.get_command_input_port())

MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

# Build the diagram and create a simulator
diagram = builder.Build()
diagram_visualize_connections(diagram, "diagram.svg")


################################################################################
##### Simulation Setup
################################################################################
simulator = Simulator(diagram)
context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(context)

# Set initial state
# plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link"), RigidTransform([0, 0, 1]))
plant.GetJointByName("rx").set_angle(plant_context, 0.0)  # Roll
plant.GetJointByName("ry").set_angle(plant_context, 0.1)  # Pitch
plant.GetJointByName("rz").set_angle(plant_context, 0.0)  # Yaw
plant.GetJointByName("x").set_translation(plant_context, -1.5)
plant.GetJointByName("y").set_translation(plant_context, 0.0)
plant.GetJointByName("z").set_translation(plant_context, 1.0)

# Solve for trajectory
# x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace, N = solve_trajectory(plant.get_state_output_port().Eval(plant_context), np.zeros(6))
N=15
x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace = solve_trajectory_fixed_timesteps(plant.get_state_output_port().Eval(plant_context), np.zeros(6), N)

print(f"{N=}\n")
print(f"{x_trj=}\n")
print(f"{u_trj=}\n")
print(f"{cost_trace=}\n")
print(f"{regu_trace=}\n")
print(f"{redu_ratio_trace=}\n")
print(f"{redu_trace=}\n")

# Visualize Trajectory
pos_3d_matrix = x_trj[:,:3].T
# print(f"{pos_3d_matrix.T=}")
meshcat.SetLine("ddp traj", pos_3d_matrix)

# Run the simulation
meshcat.StartRecording()
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(4)
meshcat.PublishRecording()