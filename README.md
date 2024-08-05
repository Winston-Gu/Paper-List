

### Research Papers List of [Jiawei Gao](https://winston-gu.github.io/)

I will try to summarize each paper in one sentence. Important papers will be marked with :star:. If you find something interesting or want to discuss it with me, feel free to contact me via [Email](mailto:winstongu20@gmail.com) or Github issues. 

Inspired by my friend Ze's [Reading List](https://github.com/YanjieZe/Paper-List).


# Legged Robots

- arXiv 2024.07, Berkeley Humanoid: A Research Platform for Learning-based Control. [Website](https://berkeley-humanoid.com/). A low-cost, DIY-style, mid-scale humanoid robot.
- :star: arXiv 2024.7, UMI on Legs: Making Manipulation Policies Mobile with Manipulation-Centric Whole-body Controllers. [Website](https://umi-on-legs.github.io/), [Code](https://umi-on-legs.github.io/).
    - Use end-effector trajectories in the task frame as interface between manipulation policy and wholebody controller.
- arXiv 2024.3, VBC: Visual Whole-Body Control for Legged Loco-Manipulation. [Website](https://wholebody-b1.github.io/), [Code](https://github.com/Ericonaldo/visual_wholebody).
    - low-level policy using all degrees of freedom to track the end-effector manipulator position: tracks a root velocity command and a target endeffector pose
    - high-level policy proposing the end-effector position based on visual inputs: provides velocity and end-effector pose command to the lowlevel policy.
- arXiv 2024.3, RoboDuet: A Framework Affording Mobile-Manipulation and Cross-Embodiment. [Website](https://locomanip-duet.github.io/). First train locomotion when arm is fixed, and then train loco policy and arm policy jointly.
- RSS 2024, RL2AC: Reinforcement Learning-based Rapid Online Adaptive Control for Legged Robot Robust Locomotion. [Paper](https://enriquecoronadozu.github.io/rssproceedings2024/rss20/p060.pdf).
    - Derive an adaptive torque compensator to mitigate the effects of external disturbances or model mismatches. 
    - Related to Disturbance rejection control, adaptive control, etc. Can be directly plug and play on top of exsiting framework.
    - May be analogous to adjust the pd gains parameter in a real-time manner. Adding feed forward into PD controller.
- L4DC 2024, Learning and Deploying Robust Locomotion Policies with Minimal Dynamics Randomization. [arXiv](https://arxiv.org/abs/2209.12878). 
    - RFI (Random Force Injection)
    - $\tau_j^r=K_p\left(\mathrm{q}_j^*-\mathrm{q}_j\right)-K_d \dot{\mathrm{q}}_j+\tau_{r_j}$
- :star: CoRL 2022, Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion. [Website](https://manipulation-locomotion.github.io/), [Code](https://github.com/MarkFzp/Deep-Whole-Body-Control).
    - Advantage mixing: mixing advantage functions for manipulation and locomotion to speed up policy learning. A kind of curriculum learning using only 1 paramter: first encourage the policy to learn manipulation and locomotion seperately, and then learn them jointly.
    - Regularized Online Adaptation: teache-student training in only 1 phase. 
        - The difference compared to RMA: regularize the teacher's encoder at training stage.
        - Regularize the teacher's encoder $z^{\mu}$ to avoid large deviation from $z^{\phi}$ estimated by the adaptation module.
        - Intuition: "The teacher policy will not learn information that is not learnable by the student policy which can only observe partial onboard information."
- RSS 2022, Rapid Locomotion via Reinforcement Learning. [Website](https://agility.csail.mit.edu/), [Code](https://github.com/Improbable-AI/rapid-locomotion-rl).
    - Implicit System Identification: The dynamic paramers $\mathrm{d}_t$ is the input of the teacher policy $\pi_{T}(\mathrm{x}_t, \mathrm{d}_t)$. The teacher policy is dividied into 2 components: $\pi_{T}(\mathrm{x}_t, \mathrm{d}_t)=\pi_{\theta_b}(\mathrm{x}_t, g_{\theta_d}(\mathrm{d}_t))$, where $g_{\theta_d}$ is the encoder that compresses $\mathrm{d}_t$ into an intermediate latent vector $\mathrm{z}_t$. The student policy tries to mimic the tecaher's action by implicitly infer domain parameters from a state history, $\hat{\mathrm{z}_t}=h_{\theta_a}(\mathrm{x}_{[t-h:t-1]})$.



# Robotics Manipulation
- :star: RSS 2024, Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots. [Website](https://umi-gripper.github.io/), [Code](https://github.com/real-stanford/universal_manipulation_interface).
    - A framwork for data collection and policy learning. Use Go-Pro as camera & IMU, Diffusion Policy as policy. Latency in both observation space and action are dealt with.
    - The design of side mirrors is really brilliant.
- RA-L 2024, On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer, [Arxiv](https://arxiv.org/abs/2312.03673). 
    - Benchmarked 13 action spaces in FRANKA manipulation skills learning. Used 7 dof Franka Emika Panda & PPO & Isaac Gym
    - Found that the Joint Velocity action spaces seems to perform overall better. $\tau=f_{\mathrm{JIC}}(a)=K\left(q_d-q\right)+D\left(\dot{q_d}-\dot{q}\right)$, $\dot{q}_d=s(a)$, $q_d$ is computed as a first-order integration of $q_d$.
- 2001 IROS, Adaptive force control of position/velocity controlled robots: theory and experiment. [Paper](https://ieeexplore.ieee.org/document/976374).
    - Popose 2 velocity based implicit force trajectory tracking controllers
    - Adding the derivative of the desired force into the control law, and some parameters can be estimated if unknwon.


# Random Papers

## Talks