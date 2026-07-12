# DQN Principles

## Quick start

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

Alternative:

```bash
python -m pip install pygame torch numpy matplotlib opencv-python pillow
```

For headless training:

```bash
python main_Train_Headless.py
```

## File overview

- Calc_Five_Simulate.py — Runs a simulation pass to evaluate the trained agent's behavior.
- Calc_Q-Training.py — Computes Q-learning training steps for the environment.
- Calc_Q-value.py — Loads a model and prints or evaluates Q-values for a given state.
- Calc_StepOne.py — Implements the first-step environment logic and movement simulation.
- constant.py — Stores shared constants such as environment dimensions, actions, and paths.
- DQN.py — Contains the Deep Q-Network model, training loop, environment logic, and rendering helpers.
- main_Load_Model.py — Loads a saved model and runs inference in the environment.
- main_Obstacle_Editor.py — Provides a simple editor for creating and editing obstacle layouts.
- main_Train_Headless.py — Trains the agent without the graphical UI for faster execution.
- main_Train_UI.py — Launches the interactive training interface for the agent.
- manual_control.py — Lets a user manually control the agent for testing and debugging.


## How to Train and Load Model

- **Step 1:** Run `main_Obstacle_Editor.py` – press **S** to save obstacle. **right click** to remove obtacle or add moving obstacle. **left click** to select obstacle or add static obstacle.
- **Step 2:** Run `main_Train_Headless.py` – set the training configs (episodes, epsilon, etc.) from `constant.py`.
- **Step 3:** Run `main_Load_Model.py` – to see the trained model.
