# A Paper List of Winston Jiawei Gao



# Offline RL

- SAQ: Action-Quantized Offline Reinforcement Learning for Robotic Skill Learning.
    - Arxiv 2023, UC Berkeley(Sergey Levine)
    - [Website](https://saqrl.github.io/), [Code](https://github.com/jianlanluo/SAQ)
    - Description: Use VQ-VAE to learn state-conditioned **action discretization**, and then implement this discretization on top of Offline RL algorithms such as IQL, CQL, and BRAC.
- What Matters in Learning from Offline Human Demonstrations for Robot Manipulation
    - CoRL 2021 Oral, Stanford (Feifei Li) & UT Austin (Yuke Zhu)
    - [Website](https://robomimic.github.io/), [Code](https://github.com/ARISE-Initiative/robomimic), [Blogpost](https://robomimic.github.io/study/)
    - Description: Conduct an extensive study on utilizing offline RL algorithms for robot manipulation tasks with datasets of varying quality. Also, propose **robomimic**, a framework for robot learning from demonstration.




# Human-Scene Interaction

- AMASS: Archive of Motion Capture As Surface Shapes.
    - ICCV 2019, MPI.
    - [Website](https://amass.is.tue.mpg.de/), [Code](https://github.com/nghorbani/amass), [Dataset](https://amass.is.tue.mpg.de/download.php)
    - Description: develops a method (MoSh++) which converts mocap marker data to SMPL format model; therefore unifies 15 different optical marker-based mocap datasets to create the largest publicly available database of human motions
- AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control.
    - SIGGRAPH 2021, UC Berkeley(Xuebin Peng & Pieter Abbeel)
    - [Website](https://xbpeng.github.io/projects/AMP/index.html), [Original Code](https://github.com/xbpeng/DeepMimic)(using Tensorflow), [Frequently used Code](https://github.com/nv-tlabs/ASE) (using PyTorch and Isaac Gym)
    - Description: using GAIL-like method to learn "**adversarial motion prior**" from a dataset of unstructured motion clips, which pecifies **style-rewards** for training the character through RL.
- RFC: Residual Force Control for Agile Human Behavior Imitation and Extended Motion Synthesis
    - NeurIPS 2020, CMU RI(Kris M. Kitani)
    - [Website](https://www.ye-yuan.com/rfc), [Code](https://github. com/Khrylx/RFC)
    - Description: Add a **residual** force control term to alleviate the **dynamics mismatch** problem in physics-based humanoid motion synthesis.

- PHC: Perpetual Humanoid Control for Real-time Simulated Avatars
    - ICCV 2023, CMU RI (Zhengyi Luo)
    - [Website](https://zhengyiluo.github.io/PHC/), [Zhihu](https://zhuanlan. zhihu. com/p/663592305)
    - Description: Learn from large-scale motion databases using a single policy within the general framework of **goal-conditioned RL**, similar to previous works such as AMP. To alleviate the 'catastrophic forgetting' problem, employ **progressive learning methods**, including hard negative mining and progressive neural networks.


# LLM Robotics

- EUREKA: Human-level reward design vida coding large language models. 
    - Arxiv 2023, Nvidia etc.
    - [Website](https://eureka-research.github.io/), [Arxiv](https://arxiv.org/abs/2310.12931), [Code](https://github.com/eureka-research/Eureka). 
    - Description: Utilize Code LLMs to write reward function code, curriculum learning and RLHF (watch videos and select which is better) makes it even better.

# Robotics Manipulation

- MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations
    - CoRL 2023, Nvidia & Yuke Zhu
    - [Website](https://mimicgen.github.io/)
    - Description: "A system for automatically synthesizing large-scale, rich datasets from only a small number of human demonstrations by adapting them to new contexts. "

# Robotics Locomotion

## Bipedal Robots

- Virtual Passive Dynamic Walking and Energy-based Control Laws
    - Fumihiko Asano et al.
    - IROS 2000
- Extended Virtual Passive Dynamic Walking and Virtual Passitivity-mimicking Control Laws 
    - Fumihiko Asano et al.
    - ICRA 2001
- Parametric Excitation Mechanisms for Dynamic Bipedal Walking
    - Fumihiko Asano et al.
    - ICRA 2005
- Biped Gait Generation and Control Based on a Unified Property of Passive Dynamic Walking
    - Fumihiko Asano et al.
    - TRO 2005





# Rencent Random Papers