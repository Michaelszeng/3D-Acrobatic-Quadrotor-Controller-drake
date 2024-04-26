from pydrake.all import (
    DiagramBuilder,
    LinearQuadraticRegulator,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    Propeller,
    PropellerInfo,
    RigidTransform,
    RobotDiagramBuilder,
    SceneGraph,
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

from src.utils import (
    diagram_visualize_connections,
)

rng = np.random.default_rng(seed=7)

#####################
### Meshcat Setup ###
#####################
meshcat = StartMeshcat()
meshcat.AddButton("Close")



#####################
### Diagram Setup ###
#####################
builder = DiagramBuilder()
plant = builder.AddSystem(MultibodyPlant(0.0))
parser = Parser(plant)
(model_instance,) = parser.AddModelsFromUrl(
    "package://drake/examples/quadrotor/quadrotor.urdf"
)

# # By default the multibody has a quaternion floating base.  To match
# # QuadrotorPlant, we can manually add a FloatingRollPitchYaw joint. We set
# # `use_ball_rpy` to false because the BallRpyJoint uses angular velocities
# # instead of ṙ, ṗ, ẏ.
AddFloatingRpyJoint(
    plant,
    plant.GetFrameByName("base_link"),
    model_instance,
    use_ball_rpy=False,
)
plant.Finalize()

# Default parameters from quadrotor_plant.cc:
L = 0.15        # Length of the arms (m).
kF = 1.0        # Force input constant.
kM = 0.0245     # Moment input constant.

base_body_index = plant.GetBodyByName("base_link").index()
# Note: Rotors 0 and 2 rotate one way and rotors 1 and 3 rotate the other.
prop_info = [
    PropellerInfo(base_body_index, RigidTransform([L, 0, 0]), kF, kM),
    PropellerInfo(base_body_index, RigidTransform([0, L, 0]), kF, -kM),
    PropellerInfo(base_body_index, RigidTransform([-L, 0, 0]), kF, kM),
    PropellerInfo(base_body_index, RigidTransform([0, -L, 0]), kF, -kM),
]
propellers = builder.AddSystem(Propeller(prop_info))
builder.Connect(
    propellers.get_output_port(),
    plant.get_applied_spatial_force_input_port(),
)
builder.Connect(
    plant.get_body_poses_output_port(),
    propellers.get_body_poses_input_port(),
)
builder.ExportInput(propellers.get_command_input_port(), "u")



### Finalizing diagram setup
diagram = builder.Build()
context = diagram.CreateDefaultContext()
diagram.set_name("3D-Acrobatic-Quadrotor-Controller")
diagram_visualize_connections(diagram, "diagram.svg")



# We'll use a namedview to make it easier to work with the state.
StateView = namedview("state", plant.GetStateNames(False))

# Create the LQR controller
nominal_state = StateView.Zero()
nominal_state.z_x = 1.0  # height is 1.0m
context.SetContinuousState(nominal_state[:])
mass = plant.CalcTotalMass(plant.GetMyContextFromRoot(context))
gravity = plant.gravity_field().gravity_vector()[2]
nominal_input = [-mass * gravity / 4] * 4



########################
### Simulation Setup ###
########################
simulator = Simulator(diagram)
simulator_context = simulator.get_mutable_context()
plant_context = plant.GetMyContextFromRoot(simulator_context)
propellers_context = propellers.GetMyMutableContextFromRoot(simulator_context)

diagram.GetInputPort("u").FixValue(simulator_context, nominal_input)

# plant_context.SetContinuousState(rng.random((12,)))
# diagram.GetInputPort("u").FixValue(simulator_context, np.ones(4)*0.1)  # TESTING



####################################
### Running Simulation & Meshcat ###
####################################

simulator.set_target_realtime_rate(1)
simulator.set_publish_every_time_step(True)

meshcat.StartRecording()
simulator.AdvanceTo(5)
meshcat.PublishRecording()

while not meshcat.GetButtonClicks("Close"):
    pass