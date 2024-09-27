# 3D-Acrobatic-Quadrotor-Controller-drake
Cool acrobatic drone (using the [Drake](https://drake.mit.edu/) simulator/toolbox, iLQR trajectory optimization, and [Geometric SE(3) Controller](https://ieeexplore.ieee.org/document/5717652)) by Michael Zeng and Spring Lin.

![](dronebackflip.gif)

Controls of an underactuated quadrotor from any starting pose to "any" (6-DOF constrained) goal pose.

Future work: Improve the iLQR-generated trajectories as they tend to be too aggressive for the poorly tuned Geometric SE(3) Controller, and seem to prefer straight lines due to cost function formulation.

## Installation

Our recommended setup requires using a Linux machine, or using WSL2 on Windows, with the following requirements:

    `python` 3.10 or higher
    `pip` 23.3.1 or higher

Install Dependencies:

    `pip list --format=freeze > requirements.txt`

## Running

Adjust Parameters at the top of `main.py` as needed.

Run using:

    `python3 main.py`