# Frozen Lake RL Visualizer

An interactive, responsive desktop app (Tkinter + ttkbootstrap) to visualize Value Iteration and Policy Iteration on the Gym FrozenLake-v1 environment.

Features:
- Choose environment size (4x4 or 8x8)
- Toggle slippery dynamics
- Run Value Iteration or Policy Iteration with configurable discount (gamma)
- See state values and greedy policy arrows
- Click a cell to inspect state index, coordinates, value, and greedy action
- Animate the agent following the computed policy
- Fully resizable and responsive canvas

## Setup

1) Create and activate a virtual environment (recommended).

```cmd
python -m venv .venv
.venv\Scripts\activate
```

2) Install dependencies.

```cmd
pip install -r requirements.txt
```

Note: The app supports both `gymnasium` (recommended) and `gym`. If you prefer classic gym, adjust the requirements accordingly.

## Run

```cmd
python main_app.py
```

## Project Structure

```
frozenlake_rl_app/
├── main_app.py           # GUI app (ttkbootstrap)
├── rl_algorithms.py      # Value & Policy iteration
├── requirements.txt
└── README.md
```

## Troubleshooting
- If the window is blank on startup, resize the window once or click "Run" to trigger a redraw.
- If you see errors related to FrozenLake, ensure the package (gymnasium/gym) version supports `FrozenLake-v1`.
- On some systems, you may need to install `pygame` for rendering dependencies; it's included in requirements.

