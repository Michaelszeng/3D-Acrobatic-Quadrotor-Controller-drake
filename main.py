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
    JointSliders,
    FiniteHorizonLinearQuadraticRegulatorOptions,
    MakeFiniteHorizonLinearQuadraticRegulator,
    PiecewisePolynomial
)
from underactuated.scenarios import AddFloatingRpyJoint
from underactuated import ConfigureParser

import numpy as np
import os
import time
import argparse
import yaml

from src.utils import *
from src.ddp import solve_trajectory, solve_trajectory_fixed_timesteps_fixed_interval
# from src.se3_leaf import SE3Controller
# from src.controller import SE3Controller
from src.controllerv2 import SE3Controller

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
##### Run Trajectory Optimization
################################################################################
# Solve for trajectory
N=20  # Number of time steps in trajectory
# x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace, dt, dt_array, final_translation_error, final_rotation_error = solve_trajectory(np.array([x0, y0, z0, rz0, ry0, rx0, 0, 0, 0, 0, 0, 0]), pose_goal, N)

# print(f"{dt=}\n")
# print(f"{dt_array=}\n")
# print(f"{x_trj=}\n")
# print(f"{u_trj=}\n")
# print(f"{cost_trace=}\n")
# print(f"{regu_trace=}\n")
# print(f"{redu_ratio_trace=}\n")
# print(f"{redu_trace=}\n")
# print(f"final_translation_error: {final_translation_error:>25}    final_rotation_error: {final_rotation_error:>25}")


x_trj = np.array([[-1.50000000e+00,  0.00000000e+00,  1.00000000e+00,
         7.07388269e-01,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  7.06825181e-01],
       [-1.48585223e+00,  0.00000000e+00,  1.00000000e+00,
         7.07388269e-01,  0.00000000e+00, -2.20660994e-02,
         9.99900081e-01, -1.41360328e-02,  0.00000000e+00,
         1.41360328e-02,  9.99900081e-01,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
         1.71766381e+00,  3.21401770e+00,  8.01834932e-01],
       [-1.45553559e+00,  0.00000000e+00,  9.99054310e-01,
         7.07388269e-01,  0.00000000e+00, -1.15389777e-01,
         9.89290213e-01, -4.32454982e-02,  1.39408399e-01,
         5.32671797e-02,  9.96195268e-01, -6.89753197e-02,
        -1.35895116e-01,  7.56625010e-02,  9.87829795e-01,
         5.38098790e+00,  1.05505665e+01,  1.10077750e+00],
       [-1.41623625e+00,  0.00000000e+00,  9.92643767e-01,
         7.69736254e-01, -3.08480137e-02, -2.18600059e-01,
         7.42884166e-01,  2.71858895e-02,  6.68867733e-01,
         2.22023705e-01,  9.32612534e-01, -2.84498394e-01,
        -6.31528773e-01,  3.59853844e-01,  6.86787172e-01,
         3.76910989e+00,  1.16384776e+01,  7.24618787e-01],
       [-1.36725303e+00, -1.96305542e-03,  9.78732854e-01,
         1.22241711e+00, -2.23392774e-01, -3.78064287e-01,
         1.08770472e-01,  2.13743554e-01,  9.70815470e-01,
         4.69922492e-01,  8.49541133e-01, -2.39692959e-01,
        -8.75980500e-01,  4.82279542e-01, -8.03791348e-03,
        -4.87315777e+00,  5.12901396e+00,  5.44903357e-01],
       [-1.28262415e+00, -1.74287090e-02,  9.52559173e-01,
         2.03278878e+00, -4.23472386e-01, -1.06392764e+00,
        -2.39908574e-01, -1.16790758e-01,  9.63744673e-01,
         5.04688565e-01,  8.33035851e-01,  2.26584912e-01,
        -8.29296888e-01,  5.40750579e-01, -1.40909483e-01,
        -1.54411626e+01, -4.74230331e+00,  1.54522078e+00],
       [-1.13355298e+00, -4.84833506e-02,  8.74537812e-01,
         2.82240354e+00, -2.37826958e-01, -1.89877753e+00,
        -4.97924956e-02, -9.38623858e-01,  3.41329695e-01,
         7.42507700e-01,  1.93781162e-01,  6.41195116e-01,
        -6.67984298e-01,  2.85366632e-01,  6.87286594e-01,
        -1.09185307e+01, -8.70127845e+00,  3.89320597e-01],
       [-9.17722117e-01, -6.66701180e-02,  7.29337178e-01,
         3.09073276e+00,  2.66235194e-01, -2.10865799e+00,
        -1.17076374e-01, -8.89251370e-01, -4.42182228e-01,
         9.89308653e-01, -1.43413821e-01,  2.64738522e-02,
        -8.69569521e-02, -4.34355242e-01,  8.96534446e-01,
        -1.18023297e+00, -2.47904849e+00, -1.20118148e+00],
       [-6.73716899e-01, -4.56515500e-02,  5.62864179e-01,
         2.68095325e+00,  2.90769066e-01, -2.05229448e+00,
        -1.26528456e-01, -8.56757677e-01, -4.99956832e-01,
         9.83443380e-01, -4.24296896e-02, -1.76178431e-01,
         1.29729210e-01, -5.13970821e-01,  8.47941228e-01,
         3.56635671e-01,  1.87114364e+00, -7.11553612e-01],
       [-4.56687350e-01, -2.21131018e-02,  3.96726054e-01,
         2.12070014e+00,  9.33429934e-02, -1.89623188e+00,
        -1.80804238e-03, -8.74707461e-01, -4.84647902e-01,
         9.99611436e-01,  1.18999192e-02, -2.52065219e-02,
         2.78156036e-02, -4.84505160e-01,  8.74346065e-01,
         5.77173903e-01,  4.33359003e-01, -2.45035742e-01],
       [-2.81499078e-01, -1.44021589e-02,  2.40080812e-01,
         1.50312154e+00,  6.12227500e-02, -1.59245882e+00,
         3.27137743e-02, -8.96489510e-01, -4.41855596e-01,
         9.99449751e-01,  3.17657303e-02,  9.54639850e-03,
         5.47761957e-03, -4.41924764e-01,  8.97035394e-01,
         1.76811771e+00, -5.19641235e-01,  3.78736940e-02],
       [-1.55236869e-01, -9.25944787e-03,  1.06314272e-01,
         8.83418545e-01,  7.46115852e-02, -1.15840562e+00,
         1.34278121e-02, -9.52159195e-01, -3.05307323e-01,
         9.98909653e-01,  2.64272629e-02, -3.84851240e-02,
         4.47124016e-02, -3.04457661e-01,  9.51475871e-01,
         1.54263550e+00,  3.65109784e-01,  1.44626700e-01],
       [-7.99826964e-02, -2.90364618e-03,  7.63527386e-03,
         4.45886667e-01,  1.94590640e-02, -6.30524831e-01,
         9.00258590e-03, -9.84071881e-01, -1.77542912e-01,
         9.99892265e-01,  1.09174897e-02, -9.81159203e-03,
         1.15936347e-02, -1.77435454e-01,  9.84064148e-01,
         1.01747715e+00,  9.79211798e-02,  1.01236314e-01],
       [-4.15441906e-02, -1.22614066e-03, -4.67203151e-02,
         2.05674108e-01,  6.18414582e-03, -1.44792328e-01,
         1.48965827e-03, -9.95885588e-01, -9.06072665e-02,
         9.99996767e-01,  1.67026311e-03, -1.91747873e-03,
         2.06092741e-03, -9.06041171e-02,  9.95884856e-01,
         5.76893456e-01, -5.12018377e-03,  3.07951544e-02],
       [-2.36306393e-02, -6.87521508e-04, -5.93312598e-02,
         9.79122783e-02,  3.90363318e-03,  1.85222840e-01,
        -1.21568170e-03, -9.99179805e-01, -4.04751612e-02,
         9.99996547e-01, -1.12037318e-03, -2.37734053e-03,
         2.33004336e-03, -4.04779115e-02,  9.99177717e-01,
         2.76446240e-01,  2.02426246e-02, -5.36241811e-03],
       [-1.50262270e-02, -3.44474955e-04, -4.30541011e-02,
         5.81785862e-02,  1.56984343e-03,  3.04005597e-01,
        -6.94233483e-04, -9.99868636e-01, -1.61934568e-02,
         9.99999599e-01, -6.84988825e-04, -5.76428047e-04,
         5.65259989e-04, -1.61938505e-02,  9.99868711e-01,
         1.11772205e-01,  3.42937715e-03, -1.33767563e-02],
       [-9.87326646e-03, -2.05431680e-04, -1.61278911e-02,
         4.54355016e-02,  1.11623606e-03,  2.21944566e-01,
         4.93900432e-04, -9.99980069e-01, -6.29433015e-03,
         9.99999840e-01,  4.95620674e-04, -2.71743417e-04,
         2.74857601e-04, -6.29419493e-03,  9.99980154e-01,
         4.62968922e-02,  2.13082041e-03, -9.90224842e-03],
       [-5.82091091e-03, -1.05875492e-04,  3.66716485e-03,
         4.10274883e-02,  9.25930088e-04,  4.72996383e-02,
         1.37786883e-03, -9.99996707e-01, -2.16501182e-03,
         9.99999047e-01,  1.37805731e-03, -8.55654799e-05,
         8.85487085e-05, -2.16489186e-03,  9.99997653e-01,
         2.41097150e-02,  9.78965227e-04,  7.22485190e-03],
       [-2.13895683e-03, -2.27792017e-05,  7.91200419e-03,
         3.93976846e-02,  8.61517075e-04, -8.02946904e-02,
         7.29580725e-04, -9.99999734e-01, -1.22979669e-06,
         9.99999734e-01,  7.29580725e-04,  1.02165543e-08,
        -9.31931537e-09, -1.22980382e-06,  1.00000000e+00,
         1.30902666e-05, -2.36567214e-07, -6.88300655e-04],
       [ 1.41644398e-03,  5.49674612e-05,  6.65897985e-04,
         3.93965959e-02,  8.61526119e-04, -8.02946904e-02,
         7.91695644e-04, -9.99999687e-01, -4.84965297e-08,
         9.99999687e-01,  7.91695644e-04, -1.20307427e-08,
         1.20691337e-08, -4.84869896e-08,  1.00000000e+00,
         1.30902568e-05, -2.37217695e-07, -6.88300655e-04]])
u_trj=np.array([[0.51740839, 1.94667784, 3.19575647, 1.08784594],
       [0.36686893, 1.60931341, 3.20160745, 0.73735705],
       [1.77352021, 1.11587293, 2.02091661, 1.32858546],
       [3.27797037, 1.06289164, 1.53669528, 2.36473921],
       [3.76759528, 1.32117328, 1.42657238, 2.82903075],
       [2.63976867, 2.26434158, 2.058135  , 1.69647176],
       [1.51200049, 2.39059829, 2.92480499, 1.1397125 ],
       [2.07594571, 2.1302345 , 2.97541728, 1.91571256],
       [3.0409729 , 2.47858872, 2.74834196, 2.46022189],
       [3.20875262, 2.93556346, 3.01836635, 2.7921045 ],
       [3.18362207, 3.18543578, 3.35827504, 3.2124101 ],
       [3.27467452, 3.24128936, 3.21942374, 3.30258635],
       [3.02331077, 3.04347481, 3.00201603, 3.09451643],
       [2.62854101, 2.64331416, 2.63315752, 2.677811  ],
       [2.16273699, 2.15818268, 2.15956804, 2.17692218],
       [1.72292213, 1.71630373, 1.72269771, 1.72369641],
       [1.52829188, 1.51322489, 1.52808274, 1.51571268],
       [1.62216749, 1.62703064, 1.62198332, 1.62971565],
       [1.9006875 , 1.9006875 , 1.9006875 , 1.9006875 ]])
dt_array = np.array([0.020000000000000004, 0.042857142857142864, 0.05555555555555556, 0.06363636363636364, 0.06923076923076923, 0.07333333333333333, 0.07647058823529412, 0.07894736842105264, 0.08095238095238096, 0.08260869565217392, 0.084, 0.0851851851851852, 0.08620689655172414, 0.08709677419354839, 0.08787878787878789, 0.08857142857142858, 0.0891891891891892, 0.08974358974358974, 0.0902439024390244, 0.09069767441860466])




# Visualize Trajectory
pos_3d_matrix = x_trj[:,:3].T
# print(f"{pos_3d_matrix.T=}")
meshcat.SetLine("ddp traj", pos_3d_matrix)



################################################################################
##### Diagram Setup
################################################################################
builder = DiagramBuilder()
quadrotor_diagram_builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(quadrotor_diagram_builder, time_step=0.0)  # time step of 0 --> continuous time
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
propellers = quadrotor_diagram_builder.AddSystem(Propeller(prop_info))
quadrotor_diagram_builder.Connect(
    propellers.get_output_port(),
    plant.get_applied_spatial_force_input_port()
)
quadrotor_diagram_builder.Connect(
    plant.get_body_poses_output_port(),
    propellers.get_body_poses_input_port()
)
MeshcatVisualizer.AddToBuilder(quadrotor_diagram_builder, scene_graph, meshcat)
quadrotor_diagram_builder.ExportInput(propellers.get_command_input_port())
quadrotor_diagram_builder.ExportOutput(plant.GetOutputPort("state"))
quadrotor_diagram = quadrotor_diagram_builder.Build()

builder.AddSystem(quadrotor_diagram)

Q = np.diag(np.concatenate(([10] * 6, [1] * 6, [0] * 13)))
R = np.eye(4)
options = FiniteHorizonLinearQuadraticRegulatorOptions()
options.Qf = Q
break_points = np.concatenate(([0], np.cumsum(dt_array)))
x_trj = transform_state_trajectory(x_trj)  # need to convert rotation matrices to RPY and angular velocities to RPY_dot, and also add 13 zero's to account for extra state variables
x_trj = np.vstack((x_trj, x_trj[np.shape(x_trj)[0]-1]))  # Repeat last state at very end of trajectory
options.x0 = PiecewisePolynomial.FirstOrderHold(break_points, x_trj.T)
u_trj = np.vstack((u_trj, u_trj[np.shape(u_trj)[0]-1], u_trj[np.shape(u_trj)[0]-1]))  # Repeat last command 2x at very end of trajectory
options.u0 = PiecewisePolynomial.FirstOrderHold(break_points, u_trj.T)
controller = builder.AddSystem(
    MakeFiniteHorizonLinearQuadraticRegulator(
        system=quadrotor_diagram,
        context=quadrotor_diagram.CreateDefaultContext(),
        t0=options.u0.start_time(),
        tf=options.u0.end_time(),
        Q=Q,
        R=R,
        options=options,
    )
)
builder.Connect(controller.get_output_port(), quadrotor_diagram.get_input_port())
builder.Connect(quadrotor_diagram.get_output_port(), controller.get_input_port())


# se3_controller = builder.AddSystem(SE3Controller())
# state_converter = builder.AddSystem(StateConverter())
# desired_state_source = builder.AddSystem(TrajectoryDesiredStateSource())
# builder.Connect(
#     plant.GetOutputPort("quadrotor_state"),
#     state_converter.GetInputPort("drone_state")
# )
# builder.Connect(
#     state_converter.GetOutputPort("drone_state_se3"),
#     se3_controller.GetInputPort("drone_state")
# )
# builder.Connect(
#     desired_state_source.GetOutputPort("trajectory_desired_state"),
#     se3_controller.GetInputPort("x_trajectory")
# )
# builder.Connect(
#     se3_controller.GetOutputPort("controller_output"),
#     propellers.get_command_input_port()
# )


### TEMPORARY: CONSTANT CONTROL INPUT = mg ###
# g = plant.gravity_field().gravity_vector()[2]
# constant_thrust_command = [-m * g / 4] * 4
# constant_input_source = builder.AddSystem(ConstantVectorSource(constant_thrust_command))
# builder.Connect(constant_input_source.get_output_port(), propellers.get_command_input_port())

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
# desired_state_source_context = desired_state_source.GetMyMutableContextFromRoot(context)

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


# desired_state_source.set_time_intervals(dt_array)
# desired_state_source.GetInputPort("trajectory").FixValue(desired_state_source_context, x_trj)


# Run the simulation
t = 0
meshcat.StartRecording()
simulator.set_target_realtime_rate(1.0)

# # Testing DDP with open-loop control
# for i in range(np.shape(u_trj)[0]):
#     propellers.get_command_input_port().FixValue(propellers_context, u_trj[i])
#     t += dt_array[i]
#     simulator.AdvanceTo(t)

simulator.AdvanceTo(np.sum(dt_array))

meshcat.PublishRecording()