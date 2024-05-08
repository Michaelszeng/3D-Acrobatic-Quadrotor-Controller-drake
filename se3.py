import numpy as np

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


def hat(vec):
    if len(vec) == 3:
        return np.array([[0, -vec[2], vec[1]],
                         [vec[2], 0, -vec[0]],
                         [-vec[1], vec[0], 0]])
    elif len(vec) == 1:
        return np.array([[0, vec[0]], [-vec[0], 0]])


def dxdt(x, reset, tau, Ts):
    if reset:
        dxdt.it = 0
        dxdt.dot = np.zeros((3, 1))
        dxdt.x_d1 = np.zeros((3, 1))
        dxdt.ddot = np.zeros((3, 1))
        dxdt.dot_d1 = np.zeros((3, 1))
        dxdt.d3dot = np.zeros((3, 1))
        dxdt.ddot_d1 = np.zeros((3, 1))
        dxdt.d4dot = np.zeros((3, 1))
        dxdt.d3dot_d1 = np.zeros((3, 1))

    dxdt.it += 1

    d1 = 2 * tau - Ts
    d2 = 2 * tau + Ts

    if dxdt.it > 1:
        dxdt.dot = (d1 / d2) * dxdt.dot + (2 / d2) * (x - dxdt.x_d1)

    if dxdt.it > 2:
        dxdt.ddot = (d1 / d2) * dxdt.ddot + (2 / d2) * (dxdt.dot - dxdt.dot_d1)

    if dxdt.it > 3:
        dxdt.d3dot = (d1 / d2) * dxdt.d3dot + (2 / d2) * (dxdt.ddot - dxdt.ddot_d1)

    if dxdt.it > 4:
        dxdt.d4dot = (d1 / d2) * dxdt.d4dot + (2 / d2) * (dxdt.d3dot - dxdt.d3dot_d1)

    dxdt.x_d1 = x
    dxdt.dot_d1 = dxdt.dot
    dxdt.ddot_d1 = dxdt.ddot
    dxdt.d3dot_d1 = dxdt.d3dot

    return dxdt.dot, dxdt.ddot, dxdt.d3dot, dxdt.d4dot

dx1dt = DirtyDerivative(1, 0.05, 0.01)
dx2dt = DirtyDerivative(2, 0.5, 0.01)
dx3dt = DirtyDerivative(3, 0.5, 0.01)
dx4dt = DirtyDerivative(4, 0.5, 0.01)
db1dt = DirtyDerivative(1, 0.05, 0.01)
db2dt = DirtyDerivative(2, 0.5, 0.01)
dv1dt = DirtyDerivative(1, 0.05, 0.01)
dv2dt = DirtyDerivative(2, 0.5, 0.01)

def controller(u, P):
    global dx1dt, dx2dt, dx3dt, dx4dt, db1dt, db2dt, dv1dt, dv2dt 
    # process inputs
    xd = u[:3]
    b1d = u[3:6]
    # current state
    x = u[6:9]
    v = u[9:12]
    R = u[12:21].reshape(3, 3)
    Omega = u[21:24]
    t = u[-1]

    if t == 0:
        dx1dt = DirtyDerivative(1, P.tau, P.Ts)
        dx2dt = DirtyDerivative(2, P.tau * 10, P.Ts)
        dx3dt = DirtyDerivative(3, P.tau * 10, P.Ts)
        dx4dt = DirtyDerivative(4, P.tau * 10, P.Ts)

        db1dt = DirtyDerivative(1, P.tau, P.Ts)
        db2dt = DirtyDerivative(2, P.tau * 10, P.Ts)

        dv1dt = DirtyDerivative(1, P.tau, P.Ts)
        dv2dt = DirtyDerivative(2, P.tau * 10, P.Ts)

    # numerical derivatives of desired position, xd
    xd_1dot = dx1dt.calculate(xd)
    xd_2dot = dx2dt.calculate(xd_1dot)
    xd_3dot = dx3dt.calculate(xd_2dot)
    xd_4dot = dx4dt.calculate(xd_3dot)

    # numerical derivatives of desired body-1 axis, b1d
    b1d_1dot = db1dt.calculate(b1d)
    b1d_2dot = db2dt.calculate(b1d_1dot)

    # numerical derivatives of current state velocity, v
    v_1dot = dv1dt.calculate(v)
    v_2dot = dv2dt.calculate(v_1dot)

    # calculate errors, eq 17-18
    ex = x - xd
    ev = v - xd_1dot
    ea = v_1dot - xd_2dot
    ej = v_2dot - xd_3dot

    # inertial frame 3-axis
    e3 = np.array([[0], [0], [1]])

    # thrust magnitude control, eq 19
    A = -P.kx * ex - P.kv * ev - P.mass * P.gravity * e3 + P.mass * xd_2dot
    f = -np.dot(A.T, R.dot(e3))

    # normalized feedback function, eq 23
    b3c = -A / np.linalg.norm(A)

    # Construct b1c, eq 38
    C = np.cross(b3c.flatten(), b1d.flatten())
    b1c = -(1 / np.linalg.norm(C)) * np.cross(b3c.flatten(), C)

    b2c = C / np.linalg.norm(C)

    # computed attitude, eq 22
    Rc = np.column_stack((b1c, b2c, b3c))

    # time derivatives of body axes
    A_1dot = -P.kx * ev - P.kv * ea + P.mass * xd_3dot
    b3c_1dot = -A_1dot / np.linalg.norm(A) + (np.dot(A, A_1dot) / np.linalg.norm(A) ** 3) * A
    C_1dot = np.cross(b3c_1dot, b1d) + np.cross(b3c, b1d_1dot)
    b2c_1dot = C / np.linalg.norm(C) - (np.dot(C, C_1dot) / np.linalg.norm(C) ** 3) * C
    b1c_1dot = np.cross(b2c_1dot, b3c) + np.cross(b2c, b3c_1dot)

    # second time derivatives of body axes
    A_2dot = -P.kx * ea - P.kv * ej + P.mass * xd_4dot
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
    Omegac = np.cross(Rc.T.dot(Rc_1dot), Rc.T.dot(R))
    Omegac_1dot = np.cross(Rc.T.dot(Rc_2dot) - hat(Omegac).dot(hat(Omegac)), Rc.T.dot(R))

    # inertia matrix
    J = np.diag([P.Jxx, P.Jyy, P.Jzz])

    # more error, eq 21
    eR = 0.5 * np.array([np.dot(Rc.T, R) - np.dot(R.T, Rc)]).reshape(-1, 1)
    eOmega = Omega - Rc.T.dot(R).dot(Omegac)

    # moment vector control
    M = -P.kR * eR - P.kOmega * eOmega + np.cross(Omega, J.dot(Omega)) \
        - J.dot(hat(Omega).dot(R.T).dot(Rc).dot(Omegac) - Rc.T.dot(R).dot(Omegac_1dot))

    # calculate SO(3) error function, Psi
    Psi = 0.5 * np.trace(np.eye(3) - Rc.T.dot(R))

    deltaF = P.Mix * np.concatenate((f, M))

    return np.concatenate((f, M, xd, xd_1dot, Omegac.flatten(), Psi, deltaF))


# Example of use
class Params:
    def __init__(self):
        self.kx = 1
        self.kv = 1
        self.mass = 1
        self.gravity = 9.81
        self.Jxx = 1
        self.Jyy = 1
        self.Jzz = 1
        self.tau = 0.05
        self.Ts = 0.01
        self.Mix = 1  # Mix is just a placeholder here


# Example usage:
P = Params()
u = np.random.rand(18)  # Placeholder input
out = controller(u, P)
print(out)
