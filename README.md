Repository for Particle Filter Being Used in Pybullet

## Overview
This repository implements a Particle Filter for robot localization in a PyBullet simulation environment. The robot uses sensor data to estimate its position within a probabilistic occupancy grid map (OGM). The simulation visualizes how the filter converges over time as the robot moves and collects data.


## Key Files:

├── PfSimulation.py         Main simulation loop entry point<br>
├── ParticleFilter.py       Core particle filter logic<br>
├── OGM.py                  Probabilistic Occupancy Grid Map class<br>
├── utils.py                Raycasting and helper utilities<br>
├── requirements.txt        Python dependencies<br>
└── README.md               Project documentation<br>

## Installing Dependencies
To install the necessary libraries, run the command below in your terminal after cloning the repository: 
```bash
pip install -r requirements.txt
```


