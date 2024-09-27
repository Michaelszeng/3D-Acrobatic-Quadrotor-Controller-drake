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
    RotationMatrix,
    RobotDiagramBuilder,
    ConstantVectorSource,
    SceneGraph,
    AddMultibodyPlantSceneGraph,
    Simulator,
    StartMeshcat,
    namedview,
    JointSliders,
    Rgba,
)
from manipulation.scenarios import AddMultibodyTriad
from manipulation.meshcat_utils import AddMeshcatTriad

import numpy as np
import os
import datetime
import time
import argparse

from src.utils import *
from src.ddp import solve_trajectory, solve_trajectory_fixed_timesteps_fixed_interval, make_basic_test_traj
from src.controllerv3 import Controller

meshcat = StartMeshcat()


################################################################################
##### User-Defined Constants
################################################################################
# Set initial pose of quadrotor
x0 = -1
y0 = 0
z0 = 1
roll0 = 0
pitch0 = 0
yaw0 = 0

# This only matters if `USE_PRE_COMPUTED_TRAJECTORY` = False
pose_goal = np.array([0.5, 0, 1.25, 0, 3.14, 0])  # x,y,z,R,P,Y

USE_PRE_COMPUTED_TRAJECTORY = False
TRAJECTORY_FILENAME = "trajectories/random.pkl"

# Visualization Settings
SHOW_GHOST_QUADROTORS = True
SHOW_TRIADS = False


################################################################################
##### Diagram Setup
################################################################################
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
parser = Parser(plant)
(model_instance,) = parser.AddModelsFromUrl("package://drake/examples/quadrotor/quadrotor.urdf")

# Add visual quadrotor to show the desired pose of the main quadrotor
(visual_quadrotor_model_instance,) = parser.AddModelsFromUrl(f"file://{os.path.abspath('assets/visual_quadrotor.urdf')}")
visual_quadrotor_pose = RigidTransform(RollPitchYaw(pose_goal[3:]), pose_goal[:3])
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("visual_quadrotor_base_link"), visual_quadrotor_pose)

if SHOW_GHOST_QUADROTORS: 
    ghost_quad_instances = show_ghost_quadrotors(parser)

plant.Finalize()

# Set up the propellers to generate spatial force on quadrotor
body_index = plant.GetBodyByName("base_link").index()

AddMultibodyTriad(plant.GetFrameByName("base_link"), scene_graph)

"""
Quadrotor model:
 - Right-handed coordinate frames
 - b1 axis points between rotors 1 and 4
 - b2 axis points between rotors 1 and 2
 - b3 axis points vertically upwards 
 - Rotors 2 and 4 spin CW, Rotors 1 and 3 spin CCW to generate positive (in b3 axis) force, and therefore negative moment
"""
prop_info = [
    PropellerInfo(body_index, RigidTransform([L / np.sqrt(2), L / np.sqrt(2), 0]), kF, kM),
    PropellerInfo(body_index, RigidTransform([-L / np.sqrt(2), L / np.sqrt(2), 0]), kF, -kM),
    PropellerInfo(body_index, RigidTransform([-L / np.sqrt(2), -L / np.sqrt(2), 0]), kF, kM),
    PropellerInfo(body_index, RigidTransform([L / np.sqrt(2), -L / np.sqrt(2), 0]), kF, -kM),
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

controller = builder.AddSystem(Controller(meshcat, SHOW_TRIADS))
state_converter = builder.AddSystem(StateConverter())
desired_state_source = builder.AddSystem(TrajectoryDesiredStateSource())
builder.Connect(
    plant.GetOutputPort("quadrotor_state"),
    state_converter.GetInputPort("drone_state")
)
builder.Connect(
    state_converter.GetOutputPort("drone_state_se3"),
    controller.GetInputPort("drone_state")
)
builder.Connect(
    desired_state_source.GetOutputPort("trajectory_desired_state"),
    controller.GetInputPort("desired_state")
)
builder.Connect(
    desired_state_source.GetOutputPort("trajectory_desired_acceleration"),
    controller.GetInputPort("desired_acceleration")
)
builder.Connect(
    desired_state_source.GetOutputPort("trajectory_desired_angular_acceleration"),
    controller.GetInputPort("desired_angular_acceleration")
)
builder.Connect(
    controller.GetOutputPort("controller_output"),
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

# Set initial state
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link"), RigidTransform(RollPitchYaw(roll0, pitch0, yaw0), [x0, y0, z0]))

################################################################################
##### Run Trajectory Optimization
################################################################################
# Solve for trajectory
N=40  # Number of time steps in trajectory
if not USE_PRE_COMPUTED_TRAJECTORY:
    all_x_trj, x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace, dt, dt_array, final_translation_error, final_rotation_error = solve_trajectory(plant.get_state_output_port().Eval(plant_context), pose_goal, N)
    save_trajectory_data(TRAJECTORY_FILENAME, x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace, dt, dt_array, final_translation_error, final_rotation_error)

    # print(f"{dt=}\n")
    # print(f"{dt_array=}\n")
    # print(f"{x_trj=}\n")
    # print(f"{u_trj=}\n")
    # print(f"{cost_trace=}\n")
    # print(f"{regu_trace=}\n")
    # print(f"{redu_ratio_trace=}\n")
    # print(f"{redu_trace=}\n")
    # print(f"final_translation_error: {final_translation_error:>25}    final_rotation_error: {final_rotation_error:>25}")
else:
    x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace, dt, dt_array, final_translation_error, final_rotation_error = load_trajectory_data(TRAJECTORY_FILENAME)
    all_x_trj = [x_trj]

# For testing purposes
# x_trj, u_trj, dt_array = make_basic_test_traj(plant.get_state_output_port().Eval(plant_context), N)


################################################################################
##### Simulation and Visualization
################################################################################
# Set visual quadrotor positions
for i in range(10):
    traj_step = int(N/10 * i)
    # if 12 <= i <= 19 and i != 15 :
    #     pose = RigidTransform(RotationMatrix(x_trj[traj_step,6:15].reshape(3,3)), [100, 100, 100])  # Get em outta here
    # else:
    pose = RigidTransform(RotationMatrix(x_trj[traj_step,6:15].reshape(3,3)), x_trj[traj_step,:3])
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("visual_quadrotor_base_link", ghost_quad_instances[i]), pose)
    visual_quadrotor_joint_idx = plant.GetJointIndices(ghost_quad_instances[i])[0]
    visual_quadrotor_joint = plant.get_joint(visual_quadrotor_joint_idx)  # Joint object
    visual_quadrotor_joint.Lock(plant_context)

# Visualize Trajectory
ctr=0
num_trj = len(all_x_trj)
for x_trj in all_x_trj:
    pos_3d_matrix = x_trj[:,:3].T
    # print(f"{pos_3d_matrix.T=}")
    rgba = counter_to_rgba(ctr, num_trj)
    meshcat.SetLine(f"trajectories/traj_{ctr}", pos_3d_matrix, rgba=Rgba(rgba[0], rgba[1], rgba[2], rgba[3]))
    ctr+=1

AddMeshcatTriad(meshcat, f"Triads/Start_Pose", X_PT=RigidTransform(RotationMatrix(x_trj[0,6:15].reshape((3,3))), x_trj[0,:3]), opacity=0.5)
AddMeshcatTriad(meshcat, f"Triads/Goal_Pose", X_PT=RigidTransform(RotationMatrix(x_trj[-1,6:15].reshape((3,3))), x_trj[-1,:3]), opacity=0.5)

# Pass trajectory to TrajectoryDesiredStateSource
desired_state_source.set_time_intervals(dt_array)
desired_state_source.set_initial_state(plant.get_state_output_port().Eval(plant_context))
desired_state_source.GetInputPort("trajectory").FixValue(desired_state_source_context, x_trj)

# Run the simulation
t = 0
meshcat.StartRecording()

# # Testing DDP with open-loop control
# for i in range(np.shape(u_trj)[0]):
#     propellers.get_command_input_port().FixValue(propellers_context, u_trj[i])
#     t += dt_array[i]
#     simulator.AdvanceTo(t)

simulator.AdvanceTo(np.sum(dt_array) + 0.001)
meshcat.PublishRecording()
time.sleep(5)

date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"{date}: {meshcat.web_url()}/download")