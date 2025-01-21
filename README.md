# Acrobatic 6-DOF-Constrained iLQR Trajectory Optimization and Control for a Quadrotor
Cool acrobatic drone (using the [Drake](https://drake.mit.edu/) simulator/toolbox, iLQR trajectory optimization, and [Geometric SE(3) Controller](https://ieeexplore.ieee.org/document/5717652)) by Michael Zeng and Spring Lin.

![](dronebackflip.gif)

## Technical Achievements
 - Performs trajectory optimization from any starting pose to any goal pose over a fixed horizon, using Iterative LQR (iLQR) algorithm (also known as Differential Dynamic Programming (DDP)). Particularly worthy of note is that this algorithm solves a highly non-convex problem using only convex optimization (to a local minima), penalizing error in all 6 degrees of freedom, which is not possible with other popular works that use polynomial parameterizations of the trajectory and the differential flatness property of quadrotors [1]. Simply explained, iLQR solves the optimal trajectory `(x(t), u(t)` iteratively; each iteration, it analyticaly solves the Bellman Equation using a local linear approximation of the quadrotor dynamics and quadratic approximation of the cost function around the current guess of `(x(t), u(t)` to derive an improved control-input trajectory `u(t)`, then applies `u(t)` to the quadrotor's true dynamics to arrive at a new guess of the state trajectory `x(t)`. This process repeats until convergence.
 - Accurately tracks the optimal trajectory using a nearly globally stable geometric controller, which is not subject to linearization errors like with a simple LQR. We effectively re-implemented the seminal work on quadrotorc control [2].


<sub><sup>[1] D. Mellinger and V. Kumar, "Minimum snap trajectory generation and control for quadrotors," 2011 IEEE International Conference on Robotics and Automation, Shanghai, China, 2011, pp. 2520-2525, doi: 10.1109/ICRA.2011.5980409. keywords: {Trajectory;Angular velocity;Acceleration;Rotors;Aerodynamics;Force;Optimization}</sup></sub>

<sub><sup>[2] T. Lee, M. Leok and N. H. McClamroch, "Geometric tracking control of a quadrotor UAV on SE(3)," 49th IEEE Conference on Decision and Control (CDC), Atlanta, GA, USA, 2010, pp. 5420-5425, doi: 10.1109/CDC.2010.5717652. keywords: {Unmanned aerial vehicles;Stability analysis;Attitude control;Asymptotic stability;Propellers;Trajectory;Rotors}</sup></sub>

## Future work: 
- Improve the iLQR cost formulation as trajectories tend to be too aggressive and prefer straight lines
- Tuning the Geometric SE(3) controller


## Installation

Our recommended setup requires using a Linux machine, or using WSL2 on Windows, with the following requirements:

    `python` 3.10 or higher
    `pip` 23.3.1 or higher

Install dependencies from `requirements.txt`:

    pip list --format=freeze > requirements.txt

Or install manually:

    pip install numpy
    pip install manipulation
    pip install underactuated
    pip install ipython
    pip install pyvirtualdisplay


## Running

Adjust Parameters at the top of `main.py` as needed.

Run using:

    python3 main.py
