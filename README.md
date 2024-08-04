

### Research Papers List of [Jiawei Gao](https://winston-gu.github.io/)

I will try to summarize each paper in one sentence. Important papers will be marked with :star:. If you find something interesting or want to discuss it with me, feel free to contact me via [Email](mailto:winstongu20@gmail.com) or Github issues. 

Inspired by my friend Ze's [Reading List](https://github.com/YanjieZe/Paper-List).


# Legged Robots
- :star: arXiv 2024.7, UMI on Legs: Making Manipulation Policies Mobile with Manipulation-Centric Whole-body Controllers. [Website](https://umi-on-legs.github.io/), [Code](https://umi-on-legs.github.io/).
    - Use end-effector trajectories in the task frame as interface between manipulation policy and wholebody controller.
- arXiv 2024.07, Berkeley Humanoid: A Research Platform for Learning-based Control. [Website](https://berkeley-humanoid.com/). A low-cost, DIY-style, mid-scale humanoid robot.
- RSS 2024. RL2AC: Reinforcement Learning-based Rapid Online Adaptive Control for Legged Robot Robust Locomotion. [Paper](https://enriquecoronadozu.github.io/rssproceedings2024/rss20/p060.pdf).
    - Derive an adaptive torque compensator to mitigate the effects of external disturbances or model mismatches. 
    - Related to Disturbance rejection control, adaptive control, etc. Can be directly plug and play on top of exsiting framework.
    - May be analogous to adjust the pd gains parameter in a real-time manner. Adding feed forward into PD controller.
- L4DC 2024, Learning and Deploying Robust Locomotion Policies with Minimal Dynamics Randomization. [arXiv](https://arxiv.org/abs/2209.12878). 
    - RFI (Random Force Injection)
    - $\tau_j^r=K_p\left(\mathrm{q}_j^*-\mathrm{q}_j\right)-K_d \dot{\mathrm{q}}_j+\tau_{r_j}$


# Robotics Manipulation
- :star: RSS 2024, Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots. [Website](https://umi-gripper.github.io/), [Code](https://github.com/real-stanford/universal_manipulation_interface).
    - A framwork for data collection and policy learning. Use Go-Pro as camera & IMU, Diffusion Policy as policy. Latency in both observation space and action are dealt with.
    - The design of side mirrors is really brilliant.
- RA-L 2024, On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer, [Arxiv](https://arxiv.org/abs/2312.03673). 
    - Benchmarked 13 action spaces in FRANKA manipulation skills learning. Used 7 dof Franka Emika Panda & PPO & Isaac Gym
    - Found that the Joint Velocity action spaces seems to perform overall better. $\tau=f_{\mathrm{JIC}}(a)=K\left(q_d-q\right)+D\left(\dot{q_d}-\dot{q}\right)$, $\dot{q}_d=s(a)$, $q_d$ is computed as a first-order integration of $q_d$.

# Random Papers

## Talks