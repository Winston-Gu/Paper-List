### Research Papers List of [Jiawei Gao](https://winston-gu.github.io/)

I will try to summarize each paper in one sentence. Important papers will be marked with :star:. If you find something interesting or want to discuss it with me, feel free to contact me via [Email](mailto:winstongu20@gmail.com) or Github issues.

Inspired by my friend Ze's [Reading List](https://github.com/YanjieZe/Paper-List).

# Legged Robots
- arXiv 2024.07, Berkeley Humanoid: A Research Platform for Learning-based Control. [Website](https://berkeley-humanoid.com/). A low-cost, DIY-style, mid-scale humanoid robot.
- :star: arXiv 2024.7, UMI on Legs: Making Manipulation Policies Mobile with Manipulation-Centric Whole-body Controllers. [Website](https://umi-on-legs.github.io/), [Code](https://umi-on-legs.github.io/).Use end-effector trajectories in the task frame as interface between manipulation policy and wholebody controller.
- arXiv 2024.3, VBC: Visual Whole-Body Control for Legged Loco-Manipulation. [Website](https://wholebody-b1.github.io/), [Code](https://github.com/Ericonaldo/visual_wholebody). Decouple into low-level and high-level policy, end-effector positions are the interface.
- arXiv 2024.3, RoboDuet: A Framework Affording Mobile-Manipulation and Cross-Embodiment. [Website](https://locomanip-duet.github.io/). First train locomotion when arm is fixed, and then train loco policy and arm policy jointly.
- RSS 2024, RL2AC: Reinforcement Learning-based Rapid Online Adaptive Control for Legged Robot Robust Locomotion. [Paper](https://enriquecoronadozu.github.io/rssproceedings2024/rss20/p060.pdf). Adding feed forward into PD controller.
- L4DC 2024, Learning and Deploying Robust Locomotion Policies with Minimal Dynamics Randomization. [arXiv](https://arxiv.org/abs/2209.12878). RFI (Random Force Injection)
- TRO 2024, Adaptive Force-Based Control of Dynamic Legged Locomotion over Uneven Terrain. [Paper](https://arxiv.org/abs/2307.04030). Incorporating adaptive control into a force-based control system.
- :star: CoRL 2022, Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion. [Website](https://manipulation-locomotion.github.io/), [Code](https://github.com/MarkFzp/Deep-Whole-Body-Control). Advantage mixing and Regularized Online Adaptation.
- ICRA 2024, Learning Force Control for Legged Manipulation. [Website](https://tif-twirl-13.github.io/learning-compliance), [Thesis Paper](https://tif-twirl-13.github.io/learning-compliance/learning_compliance_thesis.pdf). End effector force tracking.
- :star: TRO 2024, Not Only Rewards but Also Constraints: Applications on Legged Robot Locomotion. [Paper](https://ieeexplore.ieee.org/abstract/document/10530429). Utilize constraints instead of reward function. Use IPO to solve the constrained RL problem.
- RSS 2023, Demonstrating a Walk in the Park: Learning to Walk in 20 Minutes With Model-Free Reinforcement Learning. [Website](https://sites.google.com/berkeley.edu/walk-in-the-park), [Code](https://github.com/ikostrikov/walk_in_the_park). Learning locomotion directly in real world, using SAC algorithms in Jax.
- IROS 2023, Hierarchical Adaptive Control for Collaborative Manipulation of a Rigid Object by Quadrupedal Robots. [Paper](https://arxiv.org/abs/2303.06741).
- ICRA 2023, Hierarchical Adaptive Loco-manipulation Control for Quadruped Robots. [Paper](https://arxiv.org/abs/2209.13145). An adaptive controller to solve the locomotion and manipulation tasks simultaneously. Use the position and velocity error to update the adaptive controller for manipulations.
- CoRL 2022, Oral. Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior. [Website](https://sites.google.com/view/gait-conditioned-rl/), [Github](https://github.com/Improbable-AI/walk-these-ways). Multiplicity of Behavior (MoB): learning a single policy that encodes a structured family of locomotion strategies that solve training tasks in different ways.
- RSS 2022, Rapid Locomotion via Reinforcement Learning. [Website](https://agility.csail.mit.edu/), [Code](https://github.com/Improbable-AI/rapid-locomotion-rl). Implicit System Identification.
- IROS 2022, Adapting Rapid Motor Adaptation for Bipedal Robots. [Paper](https://arxiv.org/abs/2205.15299). Further finetune the base policy $\pi_1$ with the imperfect extrinsics predicted by the adaptation module $\phi$.
- RA-L 2022, Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion. [Paper](https://arxiv.org/abs/2202.05481)
- :star: Science Robotics 2022, Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild. [Paper](https://www.science.org/doi/10.1126/scirobotics.abk2822). Adding a belief state encoder based on attention mechanism, which can fuse perceptive information and proprioceptive information.
- IROS 2021, Adaptive Force-based Control for Legged Robots. [Paper](https://arxiv.org/abs/2011.06236). L1 adaptive control law, force-based control.
- CoRL 2021, Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. [Paper](https://arxiv.org/abs/2109.11978).
- Science Robotics 2020, Learning Quadrupedal Locomotion over Challenging Terrain. [Paper](https://arxiv.org/abs/2010.11251).
- ICRA 2019, ALMA - Articulated Locomotion and Manipulation for a Torque-Controllable Robot. [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8794273). Track operational space motion and force references with a wholebody control algorithm that generates torque references for all the controllable joints by using hierarchical optimization.
- ACC 2015, L1 Adaptive Control for Bipedal Robots with Control Lyapunov Function based Quadratic Programs. [Paper](https://ieeexplore.ieee.org/document/7170842).
- ICRA 2015, Whole-body Pushing Manipulation with Contact Posture Planning of Large and Heavy Object for Humanoid Robot. [Paper](https://ieeexplore.ieee.org/abstract/document/7139995) . Generate pushing motion for humanoid robots, based on ZMP.


# Robotics Manipulation
- :star: RSS 2024, Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots. [Website](https://umi-gripper.github.io/), [Code](https://github.com/real-stanford/universal_manipulation_interface). A data collection framework.
- RA-L 2024, On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer, [Arxiv](https://arxiv.org/abs/2312.03673). Benchmarked 13 action spaces in FRANKA manipulation skills learning.
- IROS 2001, Adaptive force control of position/velocity controlled robots: theory and experiment. [Paper](https://ieeexplore.ieee.org/document/976374). Popose 2 velocity based implicit force trajectory tracking controllers
- TMECH 1999, A Survey of Robot Interaction Control Schemes with Experimental Comparison. [Paper](https://ieeexplore.ieee.org/abstract/document/789685).
- 1987, Dynamic Hybrid Position/Force Control of Robot Manipulators-Description of Hand Constraints and Calculation of Joint Driving Force. [Paper](https://ieeexplore.ieee.org/abstract/document/1087120).
- 1981, Hybrid Position/Force Control of Manipulators. [Paper](https://asmedigitalcollection.asme.org/dynamicsystems/article/103/2/126/400298/Hybrid-Position-Force-Control-of-Manipulators).


# Reinforcement Learning

- arXiv 2018.10, Exploration by Random Network Distillation. [Paper](https://arxiv.org/abs/1810.12894). RND for exploration.
- :star: ICML 2017, Curiosity-driven Exploration by Self-supervised Prediction. [Website](https://pathak22.github.io/noreward-rl/), [Code](https://github.com/pathak22/noreward-rl). Formulate curiosity as the error in an agent’s ability to predict the consequence of its own actions in a visual feature space learned by a self-supervised inverse dynamics model.

# Random Papers

- CoRL 2023, DATT: Deep Adaptive Trajectory Tracking for Quadrotor Control. [Website](https://sites.google.com/view/deep-adaptive-traj-tracking), [Code](https://github.com/KevinHuang8/DATT). Use L1 adaptive control to estimate disturbance.
- :star: RSS 2017, Preparing for the Unknown: Learning a Universal Policy with Online System Identification. Using an online system identification model to predict parameter $\mu$ given history, and $\mu$ is the input to the actual policy.

## Talks
