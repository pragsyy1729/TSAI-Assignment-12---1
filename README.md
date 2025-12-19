# Grid World Value Iteration

This project implements the Value Iteration algorithm for solving a Markov Decision Process (MDP) in a 5x5 grid world environment using PyTorch.

## Problem Setup

- **Grid Size**: 5x5 (25 states total)
- **States**: Represented as coordinates (i, j), where i and j range from 0 to 4
- **Actions**: Up, Down, Left, Right (4 possible actions)
- **Terminal State**: (4, 4) - the bottom-right corner
- **Rewards**: Determined by the column index of the next state:
  - Columns 0-3: Reward = -1
  - Column 4: Reward = 1
- **Discount Factor (γ)**: 1.0
- **Convergence Threshold (θ)**: 1e-4

## Algorithm

The code uses Value Iteration to compute the optimal value function V(s) for each state s (except the terminal state, which is fixed at 0).

The Bellman optimality equation is applied iteratively:

\[ V(s) = \max_a \left[ R(s, a, s') + \gamma V(s') \right] \]

Where:
- s is the current state
- a is the action
- s' is the resulting state after taking action a
- R is the reward received upon reaching s'
- γ is the discount factor

The algorithm repeats until the maximum change in value function across all states is less than θ.

## Dependencies

- PyTorch

## How to Run

1. Ensure PyTorch is installed (e.g., via `uv add torch`)
2. Run the script: `uv run main.py`

## Output

After convergence, the value function is printed:

```
Value function:
tensor([[-2., -1.,  0.,  0.],
        [-2., -1.,  0.,  0.],
        [-2., -1.,  0.,  0.],
        [-2., -1.,  0.,  0.]])
```

This shows the optimal value for each state in the grid, with the terminal state (4,4) remaining at 0.