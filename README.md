### Research Papers List of [Jiawei Gao](https://winston-gu.github.io/)

I will try to summarize each paper in one sentence. Important papers will be marked with :star:. If you find something interesting or want to discuss it with me, feel free to contact me via [Email](mailto:winstongu20@gmail.com) or Github issues.

Inspired by my friend Ze's [Reading List](https://github.com/YanjieZe/Paper-List).

# Methods

## Adapdation

### State Estimation / Sys-ID

- Dynamics as Prompts: In-Context Learning for Sim-to-Real System Identifications. [Website](https://sim2real-capture.github.io). Collect data in simulation, and use history data to predict the environment parameters in real time.
- IROS 2024, Physically Consistent Online Inertial Adaptation for Humanoid Loco-manipulation. [Paper](https://arxiv.org/abs/2405.07901). Use EKF to estimate the inertial parameters of the payload on robot's hand. Integrate with a model-based controller on a humanoid.

## Offline-to-Online

- CoRL 2024, TRANSIC: Sim-to-Real Policy Transfer by Learning from Online Correction. [Website](https://transic-robot.github.io), [Code](https://github.com/transic-robot/transic-envs). Collect online human-in-the-loop teleoperation correction data, and learned an residual policy on top of base policy trained in simulation.

## Force Control

- arXiv 2024.10, Physics-Informed Learning for the Friction Modeling of High-Ratio Harmonic Drives. [Arxiv](https://arxiv.org/abs/2410.12685#page=3.55). Estimate friction and compensate for huamnoid robots.

## World Models

- arXiv 2024.10, Language Agents Meet Causality -- Bridging LLMs and Causal World Models. [Website](https://j0hngou.github.io/LLMCWM/). Using causal representation learning to learn casual variables, and then let LLM agent to plan.
- CoRL 2022, DayDreamer: World Models for Physical Robot Learning. [Website](https://danijar.com/project/daydreamer/). Imagined rollouts in latent space.


# Tasks

## Foundation Model for Robotics

- arXiv 2024.10, Run-time Observation Interventions Make Vision-Language-Action Models More Visually Robust. [Website](https://arxiv.org/abs/2410.01971). Knowing which part of image is more sensitive to the task.
- CoRL 2023, Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners. [Website](https://robot-help.github.io/). Draw inspiration from conformal prediction theory. Letting LLM output multipule plans and also the likelihood of each choices, then calibrate by using datasets or executing these plans.
- arXiv 2024.03, Explore until Confident: Efficient Exploration for Embodied Question Answering. [Website](https://explore-eqa.github.io/). Leveraging VLM to output multiple possible plans, let the robot to explore until there is only one output from VLM.


## Humanoid Robots

- arXiv 2024.10, HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots. [Website](https://hover-versatile-humanoid.github.io). First train a privileged oracle policy, then distill in to policies in different command modes by masking.
- arXiv 2024.10, Whole-Body Dynamic Throwing with Legged Manipulators. [Paper](https://arxiv.org/abs/2410.05681). Whole-body object throwing by training with RL, the key is reward shaping.
- arXiv 2024.09, iWalker: Imperative Visual Planning for Walking Humanoid Robot. [Paper](https://arxiv.org/abs/2409.18361). Depth perception to planning, and then use model-based control.
- arXiv 2024.07, Learning Multi-Modal Whole-Body Control for Real-World Humanoid Robots. [Website](https://masked-humanoid.github.io/mhc/). Mask commands so that the robots can track different command modalities.
- arXiv 2024.06, PlaMo: Plan and Move in Rich 3D Physical Environments. [Paper](https://arxiv.org/abs/2406.18237). Integrate path planner and motion controller for humanoid characters to navigate in 3d scenes.
- arXiv 2024.05, Hierarchical World Models as Visual Whole-Body Humanoid Controllers. [Website](https://www.nicklashansen.com/rlpuppeteer/), [Code](https://github.com/nicklashansen/puppeteer). First train a tracking agent that takes abstact command as input, then train hierarchical RL for downstream tasks.
- NeurIPS 2024, Harmony4D: A Video Dataset for In-The-Wild Close Human Interactions. [Website](https://jyuntins.github.io/harmony4d/), [Code](https://github.com/jyuntins/harmony4d). Multi human interaction dataset.
- :star: SIGGRAPH Asia 2024, MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting. [Website](https://research.nvidia.com/labs/par/maskedmimic/), [Code](https://github.com/NVlabs/ProtoMotions). First train a priviledged motion imitation policy, then distill into different command modes. Policy architecture is CVAE. Encoder can provide an offset on top of leared priors (learned by a transformer), and only prior and decoder are used during test time.
- arXiv 2023.12, PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction. [Website](https://wyhuai.github.io/physhoi-page/), [Code](https://github.com/wyhuai/PhysHOI). Utilize contact-graph rewards for better tracking.
- ICLR 2019, Neural probabilistic motor primitives for humanoid control. [Paper](https://arxiv.org/abs/1811.11711). Encode motion dataset into a latent space, and use decoder as a policy.
- MIG 2018, Physics-based motion capture imitation with deep reinforcement learning. [Paper](https://dl.acm.org/doi/10.1145/3274247.3274506). Training RL to control humanoid characters. Maybe the start point of isaacgym.


## Legged Robots

- arXiv 2024.10, FRASA: An End-to-End Reinforcement Learning Agent for Fall Recovery and Stand Up of Humanoid Robots. [Paper](https://arxiv.org/abs/2410.08655). Training a DRL policy for fall recovery on kid size humanoid robots.
- arXiv 2024.10, Learning Humanoid Locomotion over Challenging Terrain. [Arxiv](https://arxiv.org/abs/2410.03654). First pretrain using transformer to do next-sequnece prediction, then fine-tune with RL.
- arXiv 2024.9 Whole-body end-effector pose tracking. [Arxiv](https://arxiv.org/abs/2409.16048). Training pose tracking task with command sampling.
- arXiv 2024.09, Real-Time Whole-Body Control of Legged Robots with Model-Predictive Path Integral Control. [Website](https://whole-body-mppi.github.io/). MPPI on quadrupeds.
- Humanoids 2024, Know your limits! Optimize the behavior of bipedal robots through self-awareness. [Website](https://evm7.github.io/Self-AWare/). Generate many reference trajectories given textual commands, and use a self-awareness module to rank them.
- arXiv 2024.09, PIP-Loco: A Proprioceptive Infinite Horizon Planning Framework for Quadrupedal Robot Locomotion. [Paper](https://arxiv.org/abs/2409.09441), [Code](https://github.com/StochLab/PIP-Loco). Training future horizon prediction in simulation, and use MPPI when deployment.
- arXiv 2024.09, Learning Skateboarding for Humanoid Robots through Massively Parallel Reinforcement Learning. [Paper](https://arxiv.org/abs/2409.07846).
- arXiv 2024.09, Learning to Open and Traverse Doors with a Legged Manipulator. [Paper](https://arxiv.org/abs/2409.04882).
- arXiv 2024.09, One Policy to Run Them All: an End-to-end Learning Approach to Multi-Embodiment Locomotion. [Paper](https://arxiv.org/abs/2409.06366). Learn an abstract locomotion controller, encoder-decoder architecture.
- SCA 2024, PartwiseMPC: Interactive Control of Contact-Guided Motions. [Website](https://www.cs.ubc.ca/~van/papers/2024-partwiseMPC/index.html). Utilize contact keyframes as task description and partwise MPC.
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
- arXiv 2022.05, Bridging Model-based Safety and Model-free Reinforcement Learning through System Identification of Low Dimensional Linear Models. [Paper](https://arxiv.org/abs/2205.05787). Dynamics of Cassie under RL policies can be seen as a low dimensional linear system.
- arXiv 2022.03, RoLoMa: Robust Loco-Manipulation for Quadruped Robots with Arms. [Paper](https://arxiv.org/abs/2203.01446)
- arXiv 2022.01, Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees. [Paper](https://arxiv.org/abs/2201.08355). Safet-aware dual policy structure.
- ISRR 2022, Reference-Free Learning Bipedal Motor Skills via Assistive Force Curricula. [Paper](https://link.springer.com/chapter/10.1007/978-3-031-25555-7_21). Learning bipedal locomotion utilizing assistive force.
- CoRL 2022, Oral. Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior. [Website](https://sites.google.com/view/gait-conditioned-rl/), [Github](https://github.com/Improbable-AI/walk-these-ways). Multiplicity of Behavior (MoB): learning a single policy that encodes a structured family of locomotion strategies that solve training tasks in different ways.
- RSS 2022, Rapid Locomotion via Reinforcement Learning. [Website](https://agility.csail.mit.edu/), [Code](https://github.com/Improbable-AI/rapid-locomotion-rl). Implicit System Identification.
- IROS 2022, PI-ARS: Accelerating Evolution-Learned Visual-Locomotion with Predictive Information Representations. [Paper](https://arxiv.org/abs/2207.13224). Predictive Information Representations: Learn an encoder to maximize predictive information (the mutual information between past and future.)
- IROS 2022, Adapting Rapid Motor Adaptation for Bipedal Robots. [Paper](https://arxiv.org/abs/2205.15299). Further finetune the base policy $\pi_1$ with the imperfect extrinsics predicted by the adaptation module $\phi$.
- RA-L 2022, Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion. [Paper](https://arxiv.org/abs/2202.05481)
- RA-L 2022, Combining Learning-Based Locomotion Policy With Model-Based Manipulation for Legged Mobile Manipulators. [Paper](https://ieeexplore.ieee.org/abstract/document/9684679?casa_token=jsU-9TWLq4oAAAAA%3AX5RmTX2AwUeHTlbAbKVHh_8-djFj-9JLlyXduwvMKe0nsyoPiGUiko5wwS0Rl3VL7HQKAm4). Decouple the manipulator control from base policy training by modeling the disturbances from the manipulator as predictable external wrenches.
- :star: Science Robotics 2022, Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild. [Paper](https://www.science.org/doi/10.1126/scirobotics.abk2822). Adding a belief state encoder based on attention mechanism, which can fuse perceptive information and proprioceptive information.
- IROS 2021, Adaptive Force-based Control for Legged Robots. [Paper](https://arxiv.org/abs/2011.06236). L1 adaptive control law, force-based control.
- CoRL 2021, Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. [Paper](https://arxiv.org/abs/2109.11978).
- RA-L 2020, Learning Fast Adaptation with Meta Strategy Optimization. [Paper](https://arxiv.org/abs/1909.12995). Finding the latent representations as an optimization problem.
- Science Robotics, 2020, Multi-Expert Learning of Adaptive Legged Locomotion. [Paper](https://www.science.org/doi/10.1126/scirobotics.abb2174). Use gating neural network to learn the combination of expert skill networks.
- Science Robotics 2020, Learning Quadrupedal Locomotion over Challenging Terrain. [Paper](https://arxiv.org/abs/2010.11251).
- arXiv 2020.04, Learning Agile Robotic Locomotion Skills by Imitating Animals. [Paper](https://arxiv.org/abs/2004.00784), [Code](https://github.com/erwincoumans/motion_imitation).
- IROS 2019, Sim-to-Real Transfer for Biped Locomotion. [Paper](https://arxiv.org/abs/1903.01390). Pre-sysID and post-sysID.
- ICRA 2019, ALMA - Articulated Locomotion and Manipulation for a Torque-Controllable Robot. [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8794273). Track operational space motion and force references with a wholebody control algorithm that generates torque references for all the controllable joints by using hierarchical optimization.
- :star: RSS 2017, Preparing for the Unknown: Learning a Universal Policy with Online System Identification. Using an online system identification model to predict parameter $\mu$ given history, and $\mu$ is the input to the actual policy.
- ACC 2015, L1 Adaptive Control for Bipedal Robots with Control Lyapunov Function based Quadratic Programs. [Paper](https://ieeexplore.ieee.org/document/7170842).
- ICRA 2015, Whole-body Pushing Manipulation with Contact Posture Planning of Large and Heavy Object for Humanoid Robot. [Paper](https://ieeexplore.ieee.org/abstract/document/7139995) . Generate pushing motion for humanoid robots, based on ZMP.


## Manipulation

- arXiv 2024.10, Multi-Task Interactive Robot Fleet Learning with Visual World Models. [Website](https://ut-austin-rpl.github.io/sirius-fleet/). First train a model to predict future trajectories, and then finetune using deployment data.,
- arXiv 2024.10, HuDOR: Bridging the Human to Robot Dexterity Gap through Object-Oriented Rewards. [Website](https://object-rewards.github.io). Get object trajectories from videos, and calculate the reward based on the object's state.
- arXiv 2024.10, Robots Pre-Train Robots: Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets. [Website](https://robots-pretrain-robots.github.io).
- arXiv 2024.10, Local Policies Enable Zero-shot Long Horizon Manipulation. [Website](https://mihdalal.github.io/manipgen/). Distill ~1000 RL experts into a generalist policy. Use a variant of dagger for better performance.
- arXiv 2024.10, One-Step Diffusion Policy: Fast Visuomotor Policies via Diffusion Distillation. [Website](https://research.nvidia.com/labs/dir/onedp/), [Paper](https://arxiv.org/abs/2410.21257). Distill diffusion policy into a one-step action generator. Fomularize the gradient of KL divergence into a score-difference loss.
- arXiv 2024.10, M3Bench: Benchmarking Whole-body Motion Generation for Mobile Manipulation in 3D Scenes. [Website](https://arxiv.org/abs/2410.06678). A benchmark for mobile manipulation in household scenes with many tasks.
- arXiv 2024.10, Discovering Robotic Interaction Modes with Discrete Representation Learning. [Website](https://actaim2.github.io). Self-supervised, Gaussian Mixture VAE. Splits the policy into a discrete mode selector and an action predictor.
- arXiv 2024.10, Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning. [Website](https://hil-serl.github.io), [Code](https://github.com/rail-berkeley/hil-serl). Real world RL with human in the loop corrections. Many designs made this data-efficeint.
- arXiv 2024.10, Combining Deep Reinforcement Learning with a Jerk-Bounded Trajectory Generator for Kinematically Constrained Motion Planning. [Paper](https://arxiv.org/abs/2410.20907). Refine RL output actions to make it jerk-bounded.
- arXiv 2024.10, DA-VIL: Adaptive Dual-Arm Manipulation with Reinforcement Learning and Variable Impedance Control. [Website](https://dualarmvil.github.io/Dual-Arm-VIL/), [Paper](https://arxiv.org/abs/2410.19712v1). RL policy output stiffness, and then passed into a QP solver to generate torque.
- arXiv 2024.10, MILES: Making Imitation Learning Easy with Self-Supervision. [Paper](https://arxiv.org/abs/2410.19693). Automatic data collection and augmentation.
- arXiv 2024.10, Learning Diffusion Policies from Demonstrations For Compliant Contact-rich Manipulation. [Paper](https://arxiv.org/abs/2410.19235). Diffusion policy output Cartesian end-effector poses and arm stiffness.
- arXiv 2024,10, ForceMimic: Force-Centric Imitation Learning with Force-Motion Capture System for Contact-Rich Manipulation. Force capture using UMI and train a hybrid force-position control policy.
- arXiv 2024.10, DROP: Dexterous Reorientation via Online Planning. [Website](https://caltech-amber.github.io/drop/). Sampling-based online planning for dexterous manipulation.
- arXiv 2024.10, Adaptive Compliance Policy: Learning Approximate Compliance for Diffusion Guided Control. [Website](https://adaptive-compliance.github.io/). Using diffusion policy as backbone, output a scalar value representing the stiffness manitude for compliance controller. Force sensors integrated.
- :star: arXiv 2024.10, Overcoming Slow Decision Frequencies in Continuous Control: Model-Based Sequence Reinforcement Learning for Model-Free Control. [Paper](https://arxiv.org/abs/2410.08979). Policy model output a sequence of actions.
- arXiv 2024.10, ARCap: Collecting High-quality Human Demonstrations for Robot Learning with Augmented Reality Feedback. [Website](https://stanford-tml.github.io/ARCap/), [Code](https://github.com/Ericcsr/ARCap). Using AR feedback when collecting human demonstrations.
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
- :star: CoRL 2023, AdaptSim: Task-Driven Simulation Adaptation for Sim-to-Real Transfer. [Website](https://irom-lab.github.io/AdaptSim/), [Code](https://github.com/irom-lab/AdaptSim). Iterately real2sim sysID.
- CoRL 2023 Oral, VoxPoser: Composable 3D Value Map for Robotic Manipulation with Language Models. [Website](https://voxposer.github.io/), [Code](https://github.com/huangwl18/VoxPoser). Utilizing LLM and VLM to write code, thus generating affordance maps and constraint maps in 3D scene.
- NeurIPS 2023 Spotlight, Learning Universal Policies via Text-Guided Video Generation. [Website](https://universal-policy.github.io/unipi/), [Code](https://github.com/flow-diffusion/AVDC), [Openreview](https://openreview.net/forum?id=bo8q5MRcwy). Formulate the sequential decision making problem as a text-conditioned video generation problem.
- CoRL 2020, Transporter Networks: Rearranging the Visual World for Robotic Manipulation. [Website](https://transporternets.github.io/). Learning pick-conditioned placing via transporting for robotics manipulation.
- IROS 2001, Adaptive force control of position/velocity controlled robots: theory and experiment. [Paper](https://ieeexplore.ieee.org/document/976374). Popose 2 velocity based implicit force trajectory tracking controllers
- TMECH 1999, A Survey of Robot Interaction Control Schemes with Experimental Comparison. [Paper](https://ieeexplore.ieee.org/abstract/document/789685).
- 1987, Dynamic Hybrid Position/Force Control of Robot Manipulators-Description of Hand Constraints and Calculation of Joint Driving Force. [Paper](https://ieeexplore.ieee.org/abstract/document/1087120).
- 1981, Hybrid Position/Force Control of Manipulators. [Paper](https://asmedigitalcollection.asme.org/dynamicsystems/article/103/2/126/400298/Hybrid-Position-Force-Control-of-Manipulators).

## Visual Robot Learning

- arXiv 2024.09, Neural Fields in Robotics: A Survey. [Website](https://robonerf.github.io/survey/index.html).


## Planning

- arXiv 2024.10, DARE: Diffusion Policy for Autonomous Robot Exploration. [Paper](https://arxiv.org/abs/2410.16687). One-step diffusion process for planning and exploration.

# Reinforcement Learning

- arXiv 2018.10, Exploration by Random Network Distillation. [Paper](https://arxiv.org/abs/1810.12894). RND for exploration.
- :star: ICML 2017, Curiosity-driven Exploration by Self-supervised Prediction. [Website](https://pathak22.github.io/noreward-rl/), [Code](https://github.com/pathak22/noreward-rl). Formulate curiosity as the error in an agent’s ability to predict the consequence of its own actions in a visual feature space learned by a self-supervised inverse dynamics model.


# Language Models


# Learning

- arXiv 2020.12, Rethinking Bias-Variance Trade-off for Generalization of Neural Networks. [Paper](https://arxiv.org/abs/2002.11328).The variance is unimodal **or** bell-shaped.

# Random Papers

- arXiv 2024.10, Motion Planning for Robotics: A Review for Sampling-based Planners. [Paper](https://arxiv.org/abs/2410.19414).
- arXiv 2024.09, Fine Manipulation Using a Tactile Skin: Learning in Simulation and Sim-to-Real Transfer. [Paper](https://arxiv.org/abs/2409.12735).
- arXiv 2024.09, A Learning-based Quadcopter Controller with Extreme Adaptation. [Paper](https://arxiv.org/abs/2409.12949). RMA for adaptation, combine BC and RL on drones.
- arXiv 2024.08, All Robots in One: A New Standard and Unified Dataset for Versatile, General-Purpose Embodied Agents. [Website](https://imaei.github.io/project_pages/ario/).
- arXiv 2024.08, Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation. [Website](https://crossformer-model.github.io/), [Code](https://github.com/rail-berkeley/crossformer). Train a transformer policy for cross embodied robots by tokenizing observations and treating actions as readout tokens.
- arXiv 2024.07, MAGIC-VFM: Meta-learning Adaptation for Ground Interaction Control with Visual Foundation Models. [Paper](https://arxiv.org/abs/2407.12304).
- arXiv 2024.05, Hierarchical World Models as Visual Whole-Body Humanoid Controllers. [Website](https://www.nicklashansen.com/rlpuppeteer/), [Code](https://github.com/nicklashansen/puppeteer).First train a low-level tracking model using MoCapAct using TD-MPC2, and then train skills on down-stream tasks.
- CoRL 2024, PianoMime: Learning a Generalist, Dexterous Piano Player from Internet Demonstrations. [Website](https://pianomime.github.io/).
- IROS 2024m Robot Synesthesia: A Sound and Emotion Guided AI Painter. [Website](https://convexalpha.github.io/Robot-Synesthesia/). Let a robot manipulator to paint something.
- arXiv 2024.02, Pushing the Limits of Cross-Embodiment Learning for Manipulation and Navigation. [Website](https://extreme-cross-embodiment.github.io/), [Code](https://github.com/JonathanYang0127/omnimimic/tree/release). A cross embodied transformer policy. Tokenize visual observations and generate actions through a conditional diffusion process.
- RSS 2024, RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots. [Website](https://robocasa.ai/), [Code](https://github.com/robocasa/robocasa). A large-scale simulation framework, a lot of kitchens.
- ICLR 2024, Spotlight. TD-MPC2: Scalable, Robust World Models for Continuous Control. [Website](https://www.tdmpc2.com/), [Code](https://github.com/nicklashansen/tdmpc2). Adding some tricks on top of TD-MPC.
- ICLR 2024, RFCL: Reverse Forward Curriculum Learning for Extreme Sample and Demonstration Efficiency in RL. [Website](https://reverseforward-cl.github.io/), [Code](https://github.com/stonet2000/rfcl).
- ICRA 2024, Safe Deep Policy Adaptation. [Website](https://sites.google.com/view/safe-deep-policy-adaptation), [Code](https://github.com/LeCAR-Lab/SafeDPA). Jointly learns adaptive policy and dynamics models in simulation, predicts environment configurations, and fine-tunes dynamics models with few-shot real-world data.
- CoRL 2023 Oral, DATT: Deep Adaptive Trajectory Tracking for Quadrotor Control. [Website](https://sites.google.com/view/deep-adaptive-traj-tracking), [Code](https://github.com/KevinHuang8/DATT). Use L1 adaptive control to estimate disturbance.
- NeurIPS 2023, Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning. [Website](https://nakamotoo.github.io/Cal-QL/), [Code](https://github.com/nakamotoo/Cal-QL).
- CVPR 2022, Masked Autoencoders Are Scalable Vision Learners. [Paper](https://arxiv.org/abs/2111.06377).  Mask random patches of the input image and reconstruct the missing pixels.
- ICML 2022, Temporal Difference Learning for Model Predictive Control. [Website](https://www.nicklashansen.com/td-mpc/), [Code](https://github.com/nicklashansen/tdmpc). Learning the dynamics model that are predictive of reward, and learning a terminal-value function by TD-learning. Use MPPI.
- arXiv 2016, Building Machines That Learn and Think Like People. [Paper](https://arxiv.org/abs/1604.00289).
- NeurIPS 2016, Learning Physical Intuition of Block Towers by Example. [Paper](https://proceedings.mlr.press/v48/lerer16.html).
