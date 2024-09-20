### Research Papers List of [Jiawei Gao](https://winston-gu.github.io/)

I will try to summarize each paper in one sentence. Important papers will be marked with :star:. If you find something interesting or want to discuss it with me, feel free to contact me via [Email](mailto:winstongu20@gmail.com) or Github issues.

Inspired by my friend Ze's [Reading List](https://github.com/YanjieZe/Paper-List).

# Legged Robots

- Humanoids 2024, Know your limits! Optimize the behavior of bipedal robots through self-awareness. [Website](https://evm7.github.io/Self-AWare/). Generate many reference trajectories given textual commands, and use a self-awareness module to rank them.
- arXiv 2024.09, PIP-Loco: A Proprioceptive Infinite Horizon Planning Framework for Quadrupedal Robot Locomotion. [Paper](https://arxiv.org/abs/2409.09441), [Code](https://github.com/StochLab/PIP-Loco). Training future horizon prediction in simulation, and use MPPI when deployment.
- arXiv 2024.09, Learning Skateboarding for Humanoid Robots through Massively Parallel Reinforcement Learning. [Paper](https://arxiv.org/abs/2409.07846).
- arXiv 2024.09, Learning to Open and Traverse Doors with a Legged Manipulator. [Paper](https://arxiv.org/abs/2409.04882).
- arXiv 2024.09, One Policy to Run Them All: an End-to-end Learning Approach to Multi-Embodiment Locomotion. [Paper](https://arxiv.org/abs/2409.06366). Learn an abstract locomotion controller, encoder-decoder architecture.
- SCA 2024, PartwiseMPC: Interactive Control of Contact-Guided Motions. [Website](https://www.cs.ubc.ca/~van/papers/2024-partwiseMPC/index.html). Utilize contact keyframes as task description and partwise MPC.
- arXiv 2024.07, Learning Multi-Modal Whole-Body Control for Real-World Humanoid Robots. [Website](https://masked-humanoid.github.io/mhc/). Mask commands so that the robots can track different command modalities.
- arXiv 2024.04, Learning H-Infinity Locomotion Control. [Website](https://junfeng-long.github.io/HINF/). Adding a learnable disturber network to achieve the robustness of the policy.
- arXiv 2024.08, PIE: Parkour with Implicit-Explicit Learning Framework for Legged Robots. [Paper](https://arxiv.org/abs/2408.13740). Use A2C framework, implicit state estimation by VAE, explicit state estimation by regression.
- arXiv 2024.07, Berkeley Humanoid: A Research Platform for Learning-based Control. [Website](https://berkeley-humanoid.com/). A low-cost, DIY-style, mid-scale humanoid robot.
- arXiv 2024.07, Wheeled Humanoid Bilateral Teleoperation with Position-Force Control Modes for Dynamic Loco-Manipulation. [Paper](https://arxiv.org/abs/2407.12189). Designed a system to retarget human loco-manipulation into a small wheeled robot with hands.
- :star: arXiv 2024.7, UMI on Legs: Making Manipulation Policies Mobile with Manipulation-Centric Whole-body Controllers. [Website](https://umi-on-legs.github.io/), [Code](https://umi-on-legs.github.io/).Use end-effector trajectories in the task frame as interface between manipulation policy and wholebody controller.
- arXiv 2024.06, SLR: Learning Quadruped Locomotion without Privileged Information. [Website](https://11chens.github.io/SLR/). Learning some representation and state transition model. Quite confused because of the poor writing.
- arXiv 2024.05, Impedance Matching: Enabling an RL-Based Running Jump in a Quadruped Robot. [Paper](https://arxiv.org/abs/2404.15096v1). Sim-to-real synchronization based on frequency-domain analysis.
- arXiv 2024.05, Combining Teacher-Student with Representation Learning: A Concurrent Teacher-Student Reinforcement Learning Paradigm for Legged Locomotion. [Paper](https://arxiv.org/abs/2405.10830).
- arXiv 2024.04, DiffuseLoco: Real-Time Legged Locomotion Control with Diffusion from Offline Datasets. [Paper](https://arxiv.org/abs/2404.19264). Use DDPM to learn from offline dataset, introducing delayed input and action predictions tricks for real-time deployment.
- arXiv 2024.03, VBC: Visual Whole-Body Control for Legged Loco-Manipulation. [Website](https://wholebody-b1.github.io/), [Code](https://github.com/Ericonaldo/visual_wholebody). Decouple into low-level and high-level policy, end-effector positions are the interface.
- arXiv 2024.03, RoboDuet: A Framework Affording Mobile-Manipulation and Cross-Embodiment. [Website](https://locomanip-duet.github.io/). First train locomotion when arm is fixed, and then train loco policy and arm policy jointly.
- arXiv 2024.01, Adaptive Mobile Manipulation for Articulated Objects In the Open World. [Website](https://open-world-mobilemanip.github.io/). Learning at test time. Use CLIP to generate reward: compute the similarity score of the observed image and 2 prompts, "door that is closed" and "door that is open"
- :star: RSS 2024, RL2AC: Reinforcement Learning-based Rapid Online Adaptive Control for Legged Robot Robust Locomotion. [Paper](https://enriquecoronadozu.github.io/rssproceedings2024/rss20/p060.pdf). Adding feed forward into PD controller.
- RSS 2024, (Best Paper Award Finalist), Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning. [Paper](https://arxiv.org/abs/2408.14472).
- RSS 2024, Rethinking Robustness Assessment: Adversarial Attacks on Learning-based Quadrupedal Locomotion Controllers. [Website](https://fanshi14.github.io/me/rss24.html).
- L4DC 2024, Learning and Deploying Robust Locomotion Policies with Minimal Dynamics Randomization. [arXiv](https://arxiv.org/abs/2209.12878). RFI (Random Force Injection)
- TRO 2024, Adaptive Force-Based Control of Dynamic Legged Locomotion over Uneven Terrain. [Paper](https://arxiv.org/abs/2307.04030). Incorporating adaptive control into a force-based control system.
- :star: CoRL 2022, Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion. [Website](https://manipulation-locomotion.github.io/), [Code](https://github.com/MarkFzp/Deep-Whole-Body-Control). Advantage mixing and Regularized Online Adaptation.
- ICRA 2024, Learning Force Control for Legged Manipulation. [Website](https://tif-twirl-13.github.io/learning-compliance), [Thesis Paper](https://tif-twirl-13.github.io/learning-compliance/learning_compliance_thesis.pdf). End effector force tracking.
- :star: TRO 2024, Not Only Rewards but Also Constraints: Applications on Legged Robot Locomotion. [Paper](https://ieeexplore.ieee.org/abstract/document/10530429). Utilize constraints instead of reward function. Use IPO to solve the constrained RL problem.
- arXiv 2023.04, Torque-based Deep Reinforcement Learning for Task-and-Robot Agnostic Learning on Bipedal Robots Using Sim-to-Real Transfer. [Paper](https://arxiv.org/abs/2304.09434). The policy outputs torque directly at 250 HZ.
- arXiv 2023.03, Learning Bipedal Walking for Humanoids with Current Feedback. [Paper](https://arxiv.org/abs/2303.03724). Simulation poor torque tracking in simulation, measure and track torque in real robots.
- CoRL 2023, Learning to See Physical Properties with Active Sensing Motor Policies. [Website](https://gmargo11.github.io/active-sensing-loco/). Active Sensing: adding the error of physical properties estimation into reward function.
- RSS 2023, Demonstrating a Walk in the Park: Learning to Walk in 20 Minutes With Model-Free Reinforcement Learning. [Website](https://sites.google.com/berkeley.edu/walk-in-the-park), [Code](https://github.com/ikostrikov/walk_in_the_park). Learning locomotion directly in real world, using SAC algorithms in Jax.
- IROS 2023, Hierarchical Adaptive Control for Collaborative Manipulation of a Rigid Object by Quadrupedal Robots. [Paper](https://arxiv.org/abs/2303.06741).
- ICRA 2023, Legs as Manipulator: Pushing Quadrupedal AgilityBeyond Locomotion. [Website](https://robot-skills.github.io/). Use one front leg as manipulator. First train locomotion and manipulation policy respectively, and then learn a behavior tree from demonstration to stitch previous skills together.
- ICRA 2023, Hierarchical Adaptive Loco-manipulation Control for Quadruped Robots. [Paper](https://arxiv.org/abs/2209.13145). An adaptive controller to solve the locomotion and manipulation tasks simultaneously. Use the position and velocity error to update the adaptive controller for manipulations.
- arXiv 2022.03, RoLoMa: Robust Loco-Manipulation for Quadruped Robots with Arms. [Paper](https://arxiv.org/abs/2203.01446)
- ISRR 2022, Reference-Free Learning Bipedal Motor Skills via Assistive Force Curricula. [Paper](https://link.springer.com/chapter/10.1007/978-3-031-25555-7_21). Learning bipedal locomotion utilizing assistive force.
- CoRL 2022, Oral. Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior. [Website](https://sites.google.com/view/gait-conditioned-rl/), [Github](https://github.com/Improbable-AI/walk-these-ways). Multiplicity of Behavior (MoB): learning a single policy that encodes a structured family of locomotion strategies that solve training tasks in different ways.
- RSS 2022, Rapid Locomotion via Reinforcement Learning. [Website](https://agility.csail.mit.edu/), [Code](https://github.com/Improbable-AI/rapid-locomotion-rl). Implicit System Identification.
- IROS 2022, PI-ARS: Accelerating Evolution-Learned Visual-Locomotion with Predictive Information Representations. [Paper](https://arxiv.org/abs/2207.13224). Predictive Information Representations: Learn an encoder to maximize predictive information (the mutual information between past and future.)
- IROS 2022, Adapting Rapid Motor Adaptation for Bipedal Robots. [Paper](https://arxiv.org/abs/2205.15299). Further finetune the base policy $\pi_1$ with the imperfect extrinsics predicted by the adaptation module $\phi$.
- RA-L 2022, Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion. [Paper](https://arxiv.org/abs/2202.05481)
- :star: Science Robotics 2022, Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild. [Paper](https://www.science.org/doi/10.1126/scirobotics.abk2822). Adding a belief state encoder based on attention mechanism, which can fuse perceptive information and proprioceptive information.
- IROS 2021, Adaptive Force-based Control for Legged Robots. [Paper](https://arxiv.org/abs/2011.06236). L1 adaptive control law, force-based control.
- CoRL 2021, Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. [Paper](https://arxiv.org/abs/2109.11978).
- RA-L 2020, Learning Fast Adaptation with Meta Strategy Optimization. [Paper](https://arxiv.org/abs/1909.12995). Finding the latent representations as an optimization problem.
- Science Robotics, 2020, Multi-Expert Learning of Adaptive Legged Locomotion. [Paper](https://www.science.org/doi/10.1126/scirobotics.abb2174). Use gating neural network to learn the combination of expert skill networks.
- Science Robotics 2020, Learning Quadrupedal Locomotion over Challenging Terrain. [Paper](https://arxiv.org/abs/2010.11251).
- arXiv 2020.04, Learning Agile Robotic Locomotion Skills by Imitating Animals. [Paper](https://arxiv.org/abs/2004.00784), [Code](https://github.com/erwincoumans/motion_imitation).
- IROS 2019, Sim-to-Real Transfer for Biped Locomotion. [Paper](https://arxiv.org/abs/1903.01390). Pre-sysID and post-sysID.
- ICRA 2019, ALMA - Articulated Locomotion and Manipulation for a Torque-Controllable Robot. [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8794273). Track operational space motion and force references with a wholebody control algorithm that generates torque references for all the controllable joints by using hierarchical optimization.
- ACC 2015, L1 Adaptive Control for Bipedal Robots with Control Lyapunov Function based Quadratic Programs. [Paper](https://ieeexplore.ieee.org/document/7170842).
- ICRA 2015, Whole-body Pushing Manipulation with Contact Posture Planning of Large and Heavy Object for Humanoid Robot. [Paper](https://ieeexplore.ieee.org/abstract/document/7139995) . Generate pushing motion for humanoid robots, based on ZMP.


# Robotics Manipulation

- arXiv 2024.09, Hand-object interaction pretraining from videos. [Website](https://hgaurav2k.github.io/hop/), [Paper](https://arxiv.org/abs/2409.08273).
- arXiv 2024.09, Closed-Loop Visuomotor Control with Generative Expectation for Robotic Manipulation. [Paper](https://arxiv.org/abs/2409.09016). Use a decoder to generate action from error embeddings.
- arXiv 2024.09, Adaptive Control based Friction Estimation for Tracking Control of Robot Manipulators. [Paper](https://arxiv.org/abs/2409.05054). Adaptive control methods.
- arXiv 2024.09, Fast Payload Calibration for Sensorless Contact Estimation Using Model Pre-training. [Paper](https://arxiv.org/abs/2409.03369).
- arXiv 2024.08, RP1M: A Large-Scale Motion Dataset for Piano Playing with Bi-Manual Dexterous Robot Hands. [Website](https://rp1m.github.io/). A dataset built on RoboPianist with shadow hands.
- arXiv 2024.08, ACE: A Cross-Platform Visual-Exoskeletons System for Low-Cost Dexterous Teleoperation. [Website](https://ace-teleop.github.io/), [Code](https://github.com/ACETeleop/ACETeleop). A teleoperation system.
- arXiv 2024.08, A Survey of Embodied Learning for Object-Centric Robotic Manipulation. [Paper](https://arxiv.org/abs/2408.11537).
- arXiv 2024.08, Real-time Dexterous Telemanipulation with an End-Effect-Oriented Learning-based Approach. [Paper](https://arxiv.org/abs/2408.00853). First using DDPG to train robots to follow operator's command offline, then test it online.
- arXiv 2024.03, CoPa: General Robotic Manipulation through Spatial Constraints of Parts with Foundational Model. [Website](https://copa-2024.github.io/).
- :star: RSS 2024, Dynamic On-Palm Manipulation via Controlled Sliding. [Website](https://dynamic-controlled-sliding.github.io/), [Code](https://github.com/DAIRLab/dairlib/tree/plate_balancing/examples/franka). Using hierarchical control methods: The system is modeled as LCS (Linear Complementarity Model), and then use C3 (Complementary Consensus Control) algorithms to solve. Low-level OSC tracking controller track the end-effector positions and force given by MPC.
- :star: RSS 2024, Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots. [Website](https://umi-gripper.github.io/), [Code](https://github.com/real-stanford/universal_manipulation_interface). A data collection framework.
- RSS 2024, Learning Manipulation by Predicting Interaction. [Website](https://opendrivelab.github.io/mpi.github.io/), [Code](https://github.com/OpenDriveLab/MPI).
- RSS 2024, Any-point Trajectory Modeling for Policy Learning. [Website](https://xingyu-lin.github.io/atm/), [Code](https://github.com/Large-Trajectory-Model/ATM). Utilize points tracking in human videos for policy learning.
- RA-L 2024, On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer, [Arxiv](https://arxiv.org/abs/2312.03673). Benchmarked 13 action spaces in FRANKA manipulation skills learning.
- CoRL 2023 Oral, VoxPoser: Composable 3D Value Map for Robotic Manipulation with Language Models. [Website](https://voxposer.github.io/), [Code](https://github.com/huangwl18/VoxPoser). Utilizing LLM and VLM to write code, thus generating affordance maps and constraint maps in 3D scene.
- NeurIPS 2023 Spotlight, Learning Universal Policies via Text-Guided Video Generation. [Website](https://universal-policy.github.io/unipi/), [Code](https://github.com/flow-diffusion/AVDC), [Openreview](https://openreview.net/forum?id=bo8q5MRcwy). Formulate the sequential decision making problem as a text-conditioned video generation problem.
- CoRL 2020, Transporter Networks: Rearranging the Visual World for Robotic Manipulation. [Website](https://transporternets.github.io/). Learning pick-conditioned placing via transporting for robotics manipulation.
- IROS 2001, Adaptive force control of position/velocity controlled robots: theory and experiment. [Paper](https://ieeexplore.ieee.org/document/976374). Popose 2 velocity based implicit force trajectory tracking controllers
- TMECH 1999, A Survey of Robot Interaction Control Schemes with Experimental Comparison. [Paper](https://ieeexplore.ieee.org/abstract/document/789685).
- 1987, Dynamic Hybrid Position/Force Control of Robot Manipulators-Description of Hand Constraints and Calculation of Joint Driving Force. [Paper](https://ieeexplore.ieee.org/abstract/document/1087120).
- 1981, Hybrid Position/Force Control of Manipulators. [Paper](https://asmedigitalcollection.asme.org/dynamicsystems/article/103/2/126/400298/Hybrid-Position-Force-Control-of-Manipulators).


# Reinforcement Learning

- arXiv 2018.10, Exploration by Random Network Distillation. [Paper](https://arxiv.org/abs/1810.12894). RND for exploration.
- :star: ICML 2017, Curiosity-driven Exploration by Self-supervised Prediction. [Website](https://pathak22.github.io/noreward-rl/), [Code](https://github.com/pathak22/noreward-rl). Formulate curiosity as the error in an agent’s ability to predict the consequence of its own actions in a visual feature space learned by a self-supervised inverse dynamics model.

# Random Papers

- arXiv 2024.09, Fine Manipulation Using a Tactile Skin: Learning in Simulation and Sim-to-Real Transfer. [Paper](https://arxiv.org/abs/2409.12735).
- arXiv 2024.09, A Learning-based Quadcopter Controller with Extreme Adaptation. [Paper](https://arxiv.org/abs/2409.12949). RMA for adaptation, combine BC and RL on drones.
- arXiv 2024.08, All Robots in One: A New Standard and Unified Dataset for Versatile, General-Purpose Embodied Agents. [Website](https://imaei.github.io/project_pages/ario/).
- arXiv 2024.08, Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation. [Website](https://crossformer-model.github.io/), [Code](https://github.com/rail-berkeley/crossformer). Train a transformer policy for cross embodied robots by tokenizing observations and treating actions as readout tokens.
- arXiv 2024.07, MAGIC-VFM: Meta-learning Adaptation for Ground Interaction Control with Visual Foundation Models. [Paper](https://arxiv.org/abs/2407.12304).
- arXiv 2024.05, Hierarchical World Models as Visual Whole-Body Humanoid Controllers. [Website](https://www.nicklashansen.com/rlpuppeteer/), [Code](https://github.com/nicklashansen/puppeteer).First train a low-level tracking model using MoCapAct using TD-MPC2, and then train skills on down-stream tasks.
- CoRL 2024, PianoMime: Learning a Generalist, Dexterous Piano Player from Internet Demonstrations. [Website](https://pianomime.github.io/).
- arXiv 2024.02, Pushing the Limits of Cross-Embodiment Learning for Manipulation and Navigation. [Website](https://extreme-cross-embodiment.github.io/), [Code](https://github.com/JonathanYang0127/omnimimic/tree/release). A cross embodied transformer policy. Tokenize visual observations and generate actions through a conditional diffusion process.
- RSS 2024, RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots. [Website](https://robocasa.ai/), [Code](https://github.com/robocasa/robocasa). A large-scale simulation framework, a lot of kitchens.
- ICLR 2024, Spotlight. TD-MPC2: Scalable, Robust World Models for Continuous Control. [Website](https://www.tdmpc2.com/), [Code](https://github.com/nicklashansen/tdmpc2). Adding some tricks on top of TD-MPC.
- ICLR 2024, RFCL: Reverse Forward Curriculum Learning for Extreme Sample and Demonstration Efficiency in RL. [Website](https://reverseforward-cl.github.io/), [Code](https://github.com/stonet2000/rfcl).
- ICRA 2024, Safe Deep Policy Adaptation. [Website](https://sites.google.com/view/safe-deep-policy-adaptation), [Code](https://github.com/LeCAR-Lab/SafeDPA). Jointly learns adaptive policy and dynamics models in simulation, predicts environment configurations, and fine-tunes dynamics models with few-shot real-world data.
- CoRL 2023 Oral, DATT: Deep Adaptive Trajectory Tracking for Quadrotor Control. [Website](https://sites.google.com/view/deep-adaptive-traj-tracking), [Code](https://github.com/KevinHuang8/DATT). Use L1 adaptive control to estimate disturbance.
- NeurIPS 2023, Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning. [Website](https://nakamotoo.github.io/Cal-QL/), [Code](https://github.com/nakamotoo/Cal-QL).
- CVPR 2022, Masked Autoencoders Are Scalable Vision Learners. [Paper](https://arxiv.org/abs/2111.06377).  Mask random patches of the input image and reconstruct the missing pixels.
- ICML 2022, Temporal Difference Learning for Model Predictive Control. [Website](https://www.nicklashansen.com/td-mpc/), [Code](https://github.com/nicklashansen/tdmpc). Learning the dynamics model that are predictive of reward, and learning a terminal-value function by TD-learning. Use MPPI.
- :star: RSS 2017, Preparing for the Unknown: Learning a Universal Policy with Online System Identification. Using an online system identification model to predict parameter $\mu$ given history, and $\mu$ is the input to the actual policy.
- arXiv 2016, Building Machines That Learn and Think Like People. [Paper](https://arxiv.org/abs/1604.00289).
- NeurIPS 2016, Learning Physical Intuition of Block Towers by Example. [Paper](https://proceedings.mlr.press/v48/lerer16.html).

## Talks
