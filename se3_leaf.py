import numpy as np
from pydrake.systems.framework import LeafSystem, BasicVector

class DirtyDerivative:
    def __init__(self, order, tau, Ts):
        self.tau = tau
        self.Ts = Ts
        self.a1 = (2 * tau - Ts) / (2 * tau + Ts)
        self.a2 = 2 / (2 * tau + Ts)
        self.order = order
        self.dot = None
        self.x_d1 = None
        self.it = 1

    def calculate(self, x):
        x = np.array(x).reshape(-1, 1)  # Ensure x is a column vector
        if self.it == 1:
            self.dot = np.zeros_like(x)
            self.x_d1 = np.zeros_like(x)

        if self.it > self.order:
            self.dot = self.a1 * self.dot + self.a2 * (x - self.x_d1)

        self.it += 1
        self.x_d1 = x
        return self.dot

class SE3Controller(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        
        # Define input port for the current state of the drone
        self.input_port_drone_state = self.DeclareInputPort("drone_state", 
                                                             BasicVector(18))  #[x, y, z, x_dot, y_dot, z_dot, R1, R2, R3, R4, R5, R6, R7, R8, R9, W1, W2, W3].T

        # Define input port for the trajectory from the DDP solver
        self.input_port_u_trajectory = self.DeclareInputPort("u_trajectory", 
                                                                BasicVector(4))  # u_traj matrix [4, N]
        
        self.input_port_x_trajectory = self.DeclareInputPort("x_trajectory", 
                                                                BasicVector(18))  # x_traj matrix [18, N] 
        
        # Define input port for the current time
        # unsure if needed, put there jsut incase for updates
        self.input_port_time = self.DeclareInputPort("time", 
                                                      BasicVector(1))  # time
        
        # Define output port for the controller output
        self.output_port_controller_output = self.DeclareVectorOutputPort("controller_output",
                                                                           BasicVector(34),  # 4 for forces, 3 for moments, 3 for xd, 3 for xd_1dot, 3 for Omegac, 1 for Psi, 18 for deltaF
                                                                           self.CalcOutput)
        
        # Initialize internal state (if needed)
        self.controller_state = np.zeros(18)  # Placeholder

        # parameters of the drone
        # Control gains (taken from Lee2011, arXiv:1003.2005v4)
        self.kx = 16 * self.mass
        self.kv = 5.6 * self.mass
        self.kR = 8.81
        self.kOmega = 2.54

        # The dist from CoM to the center of ea. rotor in the b1-b2 plane
        self.d = 0.3 # random
       
        # actuator constant
        self.c_tauf = 10 #random

        # time constant for dirty derivative filte
        self.tau = 0.05
        self.Ts = 0.01
        self.Mix = np.array([[1, 1, 1, 1],
                   [0, -self.d, 0, self.d],
                   [self.d, 0, -self.d, 0],
                   [-self.c_tauf, self.c_tauf, -self.c_tauf, self.c_tauf]]) 
        
        # physical parameters of airframe
        self.gravity = 9.81
        # tune/play around with
        self.mass = 1 # random
        self.Jxx = 1
        self.Jyy = 1
        self.Jzz = 1

        self.dx1dt = DirtyDerivative(1, 0.05, 0.01)
        self.dx2dt = DirtyDerivative(2, 0.5, 0.01)
        self.dx3dt = DirtyDerivative(3, 0.5, 0.01)
        self.dx4dt = DirtyDerivative(4, 0.5, 0.01)

        self.db1dt = DirtyDerivative(1, 0.05, 0.01)
        self.db2dt = DirtyDerivative(2, 0.5, 0.01)

        self.dv1dt = DirtyDerivative(1, 0.05, 0.01)
        self.dv2dt = DirtyDerivative(2, 0.5, 0.01)
        
    def CalcOutput(self, context, output):
        """
        Computes the output of the SE3 controller.

        Args:
            context: The Context object containing the input data.
            output: The output port to which the computed controller output is set.
        """
        # Retrieve input data from input ports
        drone_state = self.input_port_drone_state.Eval(context)
        u_trajectory = self.input_port_u_trajectory.Eval(context)
        x_trajectory = self.input_port_x_trajectory.Eval(context)
        time = self.input_port_time.Eval(context)
        
        #not sure if needed, assumed for se(3) closed loop
        # Update internal state based on current input and possibly previous state
        # updated_controller_state = self.update_controller_state(drone_state, u_trajectory, x_trajectory, time)
        
        # Compute control output based on updated internal state and other inputs
        # controller_output = self.compute_control_output(updated_controller_state, drone_state, u_trajectory, x_trajectory, time)
        controller_output = self.compute_control_output(drone_state, drone_state, u_trajectory, x_trajectory, time)

        # Set output
        output.SetFromVector(controller_output)

    def vee(self, ss):
        """
        The vee function maps a skew-symmetric matrix to a vector.
        
        Args:
            ss: Skew-symmetric matrix (2x2 or 3x3).
            
        Returns:
            vec: Vector representation of the skew-symmetric matrix.
        """
        if isinstance(ss, np.ndarray):
            ss = np.array(ss)  # Convert to numpy array
            
        if ss.shape == (2, 2):
            if not np.allclose(ss[0, 1], -ss[1, 0]):
                print('The provided matrix is not skew symmetric')
            vec = ss[0, 1]
        elif ss.shape == (3, 3):
            if not (np.allclose(ss[2, 1], -ss[1, 2]) and
                    np.allclose(ss[0, 2], -ss[2, 0]) and
                    np.allclose(ss[1, 0], -ss[0, 1])):
                print('The provided matrix is not skew symmetric.')
            vec = np.array([ss[2, 1], ss[0, 2], ss[1, 0]])
        else:
            raise ValueError('Input matrix must be 2x2 or 3x3.')
            
        return vec
        
    def hat(self, vec):
        """
        Converts a 3D vector into a skew-symmetric matrix (hat operator).

        Args:
            vec (numpy.ndarray): Input vector of length 3.

        Returns:
            numpy.ndarray: Skew-symmetric matrix (3x3) if the input vector has length 3.
                            Skew-symmetric matrix (2x2) if the input vector has length 1.
        """
        if len(vec) == 3:
            return np.array([[0, -vec[2], vec[1]],
                            [vec[2], 0, -vec[0]],
                            [-vec[1], vec[0], 0]])
        elif len(vec) == 1:
            return np.array([[0, vec[0]], [-vec[0], 0]])
        else:
            raise ValueError(f"Input vector must have length 1 or 3. The length of the input vector is {len(vec)}.")

        
    def compute_control_output(self, controller_state, drone_state, u_trajectory, x_trajectory, time):
        """
        Compute the control output of the controller.
        This method computes the control output of the controller based on the updated internal 
        state and other inputs.
        
        Args:
            controller_state: Internal state of the controller.
            drone_state: Data representing the current state of the drone.
            u_trajectory: Data representing the u_traj matrix (4xN).
            x_trajectory: Data representing the x_traj matrix (18xN).
            time: Current time.

        Returns:
            controller_output: Output vector containing forces, moments, desired state, Omegac, Psi, and deltaF.
        """
        # position tracking command
        xd = np.zeros(3) 
        # fixed body frame: {b1, b2, b3}
        b1d = np.zeros(3)
        # current states
        position = drone_state[:3]
        velocity = drone_state[3:6]
        rot_matrix = drone_state[6:15].reshape(3, 3)
        Omega = drone_state[15:]
        
        # feels redundant
        if time == 0:
            dx1dt = DirtyDerivative(1, self.tau, self.Ts)
            dx2dt = DirtyDerivative(2, self.tau * 10, self.Ts)
            dx3dt = DirtyDerivative(3, self.tau * 10, self.Ts)
            dx4dt = DirtyDerivative(4, self.tau * 10, self.Ts)

            db1dt = DirtyDerivative(1, self.tau, self.Ts)
            db2dt = DirtyDerivative(2, self.tau * 10, self.Ts)

            dv1dt = DirtyDerivative(1, self.tau, self.Ts)
            dv2dt = DirtyDerivative(2, self.tau * 10, self.Ts)

        # numerical derivatives of desired position, xd
        xd_1dot = dx1dt.calculate(xd)
        xd_2dot = dx2dt.calculate(xd_1dot)
        xd_3dot = dx3dt.calculate(xd_2dot)
        xd_4dot = dx4dt.calculate(xd_3dot)

        # numerical derivatives of desired body-1 axis, b1d
        b1d_1dot = db1dt.calculate(b1d)
        b1d_2dot = db2dt.calculate(b1d_1dot)

        # numerical derivatives of current state velocity
        v_1dot = dv1dt.calculate(velocity)
        v_2dot = dv2dt.calculate(v_1dot)

        # calculate errors, eq 17-18
        ex = position - xd
        ev = velocity - xd_1dot
        ea = v_1dot - xd_2dot
        ej = v_2dot - xd_3dot

        # inertial frame 3-axis
        e3 = np.array([[0], [0], [1]])

        # thrust magnitude control, eq 19
        A = -self.kx * ex - self.kv * ev - self.mass * self.gravity * e3 + self.mass * xd_2dot
        f = -np.dot(A.T, self.dot(e3))

        # normalized feedback function, eq 23
        b3c = -A / np.linalg.norm(A)

        # b1c, eq 38
        C = np.cross(b3c.flatten(), b1d.flatten()) # not necessary to flatten but just incase
        b1c = -(1 / np.linalg.norm(C)) * np.cross(b3c.flatten(), C)

        b2c = C / np.linalg.norm(C)

        # computed attitude, eq 22 UNSURE
        Rc = np.column_stack((b1c, b2c, b3c))

        # first time derivatives of body axes
        A_1dot = -self.kx * ev - self.kv * ea + self.mass * xd_3dot
        b3c_1dot = -A_1dot / np.linalg.norm(A) + (np.dot(A, A_1dot) / np.linalg.norm(A) ** 3) * A
        C_1dot = np.cross(b3c_1dot, b1d) + np.cross(b3c, b1d_1dot)
        b2c_1dot = C / np.linalg.norm(C) - (np.dot(C, C_1dot) / np.linalg.norm(C) ** 3) * C
        b1c_1dot = np.cross(b2c_1dot, b3c) + np.cross(b2c, b3c_1dot)

        # second time derivatives of body axes
        A_2dot = -self.kx * ea - self.kv * ej + self.mass * xd_4dot
        b3c_2dot = -A_2dot / np.linalg.norm(A) + (2 / np.linalg.norm(A) ** 3) * np.dot(A, A_1dot) * A_1dot \
                + ((np.linalg.norm(A_1dot) ** 2 + np.dot(A, A_2dot)) / np.linalg.norm(A) ** 3) * A \
                - (3 / np.linalg.norm(A) ** 5) * (np.dot(A, A_1dot) ** 2) * A
        C_2dot = np.cross(b3c_2dot, b1d) + np.cross(b3c, b1d_2dot) + 2 * np.cross(b3c_1dot, b1d_1dot)
        b2c_2dot = C_2dot / np.linalg.norm(C) - (2 / np.linalg.norm(C) ** 3) * np.dot(C, C_1dot) * C_1dot \
                - ((np.linalg.norm(C_2dot) ** 2 + np.dot(C, C_2dot)) / np.linalg.norm(C) ** 3) * C \
                + (3 / np.linalg.norm(C) ** 5) * (np.dot(C, C_1dot) ** 2) * C
        b1c_2dot = np.cross(b2c_2dot, b3c) + np.cross(b2c, b3c_2dot) + 2 * np.cross(b2c_1dot, b3c_1dot)

        # Extract calculated angular velocities and their time-derivatives
        Rc_1dot = np.column_stack((b1c_1dot, b2c_1dot, b3c_1dot))
        Rc_2dot = np.column_stack((b1c_2dot, b2c_2dot, b3c_2dot))
        Omegac = self.vee(Rc.T @ Rc_1dot)
        Omegac_1dot = self.vee(Rc.T @ Rc_2dot - self.hat(Omegac) @ self.hat(Omegac))

        # inertia matrix
        J = np.diag([self.Jxx, self.Jyy, self.Jzz])

        # eq 21
        eR = 0.5 * self.vee(Rc.T @ rot_matrix - rot_matrix.T @ Rc).reshape(-1, 1)
        eOmega = Omega - rot_matrix.T @ Rc @ Omegac

        # moment vector control
        M = -self.kR * eR - self.kOmega * eOmega + np.cross(Omega, J @ (Omega)) \
            - J.dot(self.hat(Omega) @ (rot_matrix.T) @ Rc @ Omegac - rot_matrix.T @ Rc @ Omegac_1dot)

        # calculate SO(3) error function, Psi
        Psi = 0.5 * np.trace(np.identity(3) - Rc.T @ rot_matrix)

        deltaF = self.Mix * np.concatenate((f, M))

        controller_output = np.concatenate((f, M, xd, xd_1dot, Omegac, Psi, deltaF))
        
        return controller_output
    
    def update_controller_state(self, drone_state, u_trajectory, x_trajectory, time):
        """
        Update the internal state of the controller.
        This method is responsible for updating the internal state of the controller based on 
        the current input data and possibly the previous controller state.

        Args:
            drone_state: Data representing the current state of the drone.
            u_trajectory: Data representing the u_traj matrix (4xN).
            x_trajectory: Data representing the x_traj matrix (18xN).
            time: Current time.

        Returns:
            updated_controller_state: Updated internal state of the controller.
        """
        # Placeholder implementation (replace with actual update logic)
        updated_controller_state = np.zeros(10)  # Placeholder, adjust size as needed
        
        return updated_controller_state
    
    # def dxdt(self, x, reset, tau, Ts):
    #     if reset:
    #         self.it = 0
    #         self.dot = np.zeros((3, 1))
    #         self.x_d1 = np.zeros((3, 1))
    #         self.ddot = np.zeros((3, 1))
    #         self.dot_d1 = np.zeros((3, 1))
    #         self.d3dot = np.zeros((3, 1))
    #         self.ddot_d1 = np.zeros((3, 1))
    #         self.d4dot = np.zeros((3, 1))
    #         self.d3dot_d1 = np.zeros((3, 1))

    #     self.it += 1

    #     d1 = 2 * tau - Ts
    #     d2 = 2 * tau + Ts

    #     if self.it > 1:
    #         self.dot = (d1 / d2) * self.dot + (2 / d2) * (x - self.x_d1)

    #     if self.it > 2:
    #         self.ddot = (d1 / d2) * self.ddot + (2 / d2) * (self.dot - self.dot_d1)

    #     if self.it > 3:
    #         self.d3dot = (d1 / d2) * self.d3dot + (2 / d2) * (self.ddot - self.ddot_d1)

    #     if self.it > 4:
    #         self.d4dot = (d1 / d2) * self.d4dot + (2 / d2) * (self.d3dot - self.d3dot_d1)

    #     self.x_d1 = x
    #     self.dot_d1 = self.dot
    #     self.ddot_d1 = self.ddot
    #     self.d3dot_d1 = self.d3dot

    #     return self.dot, self.ddot, self.d3dot, self.d4dot

