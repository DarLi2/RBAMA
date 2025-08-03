# Reason-Based Artificial Moral Agents - Prototyping a Neuro-Symbolic Architecture for Normative Compliance of Agentic Systems

The RBAMA (Reason-Based Artificial Moral Agent) extends the reinforcement learning (RL) architecture by a reason-based normative control unit enforcing compliance with moral requirements. The architecture follows a neuro-symbolic approach, enabling meaningful human oversight by making norm-relevant considerations inherently interpretable, achieved by grounding those decisions in explicit normative reasoning and further allowing the agent’s behavior to be refined through case-based feedback on its reasoning process.

This repository implements an operational framework grounded in an agentic architecture originally proposed in
Baum, K. et al. (2024). ([paper](https://arxiv.org/abs/2409.15014)), and further developed in Dargasz, L. (2025). ([thesis](https://arxiv.org/abs/2507.15895)). Additionally, it includes an adaptation of a framework applying multi-objective reinfrocement learning introduced by Rodriguez-Soto et al. (2021) ([paper](https://www.iiia.csic.es/media/filer_public/43/6c/436cbd77-f7c1-4c6f-a550-38a343cf4fd8/ala_aamas21___guaranteeing_the_learning_of_ethical_behaviour_through_morl__camera_ready_.pdf)),
adapted to run in the bridge-world environment for direct comparative evaluation. Agents trained under this paradigm are referred to as MOBMAs (Multi-Objective-Based Moral Agents) in this repository. Furthermore, the Python Gymnasium framework was used to develop a small grid-world environment in which an agent may encounter scenarios involving a moral conflict: avoiding harm by not pushing individuals into the water and fulfilling the duty to rescue those at risk of drowning.

## 🛠 Installation

1. **Install Conda**  
   Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.

2. **Clone the repository**

3. **Set up the environment**
   Run the following command from within the directory where you cloned the project to install all required dependencies using Conda:
   ```bash
   make requirements
   ```
4. **Activate the Environment**
   ```bash
   conda activate RBAMA
   ```

## ▶️ Example Usage

<!-- add commands for training scripts here -->

1. Train an RBAMA on an instance (bridge1_v1) of the bridge environment:

   ```bash
   python scripts/RBAMA/training/complete.py bridge1_v1 30000 3000 3000 100 prioR
   ```

2. Evaluate the RBAMA:

   - in terms of numerical return values on 1000 test episodes:

   ```bash
   python scripts/eval/returns_RBAMA.py bridge1_v1modularR3000W30000I3000R100 1000
   ```

   - by visualizing its moral decision-making:

   ```bash
   python scripts/eval/reasoning.py bridge1_v1modularR3000W30000I3000R100 --state_reset '[17, 24, 49, 49, 40]'
   ```

## 🤖 Features

1. Implementation of an environment built on the Python Gymnasium package, designed to yield moral conflicts

2. Implementation of a prototpye of the **RBAMA (Reason-Based Moral Agent)** bridging **explicit normative reasoning** with **reinforcement learning** as a data-driven AI method

   - symbolic processing via a formalized, logic-based decision-making framework
   - training of neural networks for learning efficient strategies

3. Simulating the Feedback Process by integrating a **moral judge**

   - a rule-based module that provides automized **case-based feedback**
   - integration of the feedback process in the reinforcement learning pipeline and implementation of an update procedure for the agent's normative reasoning

4. Adaptation of the approach proposed by Rodriguez-Soto et al. to the environment to enable experimental comparison

## 📈 Results

1. **Training of the RBAMA on an instance of the bridge environment:**

   <img src="pics/bridge1_v1instr.png" alt="Instrumental policy" width="300"/>

   Instrumental policy training: the agent learns to navigate to its goal

   <img src="pics/bridge1_v1resc.png" alt="Rescuing policy" width="300"/>

   Rescuing policy training: the agent learns to perform rescues

   <img src="pics/reason-theory.png" alt="Learned reason theory" width="200"/>

   Learned reason theory

2. **Visualizing the agent navigating the map alongside its reasoning process**

   <img src="pics/person_on_bridge.png" alt="My Screenshot" width="200"/>
   <img src="pics/reasoning_waiting.png" alt="My Screenshot" width="200"/>

   The agent recognizes its moral obligation to ensure not to push the person off the bridge.

   <img src="pics/drowning_person.png" alt="My Screenshot" width="200"/>
   <img src="pics/reasoning_rescuing.png" alt="My Screenshot" width="200"/>

   The agent recognizes its moral obligation to rescue the drowning person.

   <img src="pics/moral_dilemma.png" alt="My Screenshot" width="200"/>
   <img src="pics/reasoning_conflicted.png" alt="My Screenshot" width="200"/>

   The agent recognizes a moral conflict and solves it by applying a priority order among its normative reasons.

<!-- If making the code public add citation(?)

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@article{baum2024actingrightreasonscreating,
  title={Acting for the Right Reasons: Creating Reason-Sensitive Artificial Moral Agents},
  author={Baum, Kevin and Dargasz, Lisa and Jahn, Felix and Gros, Timo P and Wolf, Verena},
  journal={arXiv preprint arXiv:2409.15014},
  year={2024}
}
```

-->
