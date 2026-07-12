import os
import torch
import numpy as np

from DQN import DQN
from constant import HIDDEN_SIZE, LOAD_MODEL, STATE_SIZE, ACTION_SIZE

# INPUTS
Angle = 3
Sensor1 = 1
Sensor2 = 1
Sensor3 = 0.76
Sensor4 = 0.70
Sensor5 = 0.76
Sensor6 = 1
Sensor7 = 1
Prev_action = 1

MODEL_PATH = LOAD_MODEL


def validate_inputs(angle, sensors, prev_action):
    if not 1 <= angle <= 5:
        raise ValueError("Angle must be between 1 and 5.")
    for idx, sensor in enumerate(sensors, start=1):
        if not 0.0 <= sensor <= 1.0:
            raise ValueError(f"Sensor {idx} value must be between 0.0 and 1.0.")
    if not 0 <= prev_action <= 3:
        raise ValueError("Prev_action must be 0, 1, 2, or 3.")


def load_model(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = DQN(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    sensors = [Sensor1, Sensor2, Sensor3, Sensor4, Sensor5, Sensor6, Sensor7]
    validate_inputs(Angle, sensors, Prev_action)

    model_path = MODEL_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    input_vector = np.array([Angle] + sensors + [Prev_action], dtype=np.float32)
    input_tensor = torch.from_numpy(input_vector).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(input_tensor).cpu().numpy().flatten()

    print("Model path:", os.path.abspath(model_path))
    print("\n" + "=" * 50)
    print("INPUT VECTOR:")
    print("-" * 50)
    print(f"  Angle        : {input_vector[0]:.0f}")
    print(f"  Sensors      : {[f'{x:.2f}' for x in input_vector[1:8]]}")
    print(f"  Prev_action  : {input_vector[8]:.0f}")
    print("=" * 50)
    print("\nQ-VALUES OUTPUT:")
    print("-" * 50)
    for i, q in enumerate(q_values):
        action_name = ["Forward", "Turn Left", "Turn Right"][i]
        print(f"  Action {i} ({action_name:>10}) : {q:.2f}")
    print("-" * 50)
    best_action = int(np.argmax(q_values))
    action_names = ["Forward", "Turn Left", "Turn Right"]
    print(f"\nBEST ACTION: {best_action} ({action_names[best_action]})")
    print("=" * 50)


if __name__ == "__main__":
    main()
