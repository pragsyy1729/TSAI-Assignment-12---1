import torch


N = 4
grid = torch.zeros((N, N), dtype=torch.float32)
zeros_part = torch.full((N-1,), -1)
one_part = torch.tensor([0])
R_s = torch.cat((zeros_part, one_part), dim=0)
V = torch.zeros((N, N), dtype=torch.float32)
discount = 1
theta = 1e-4

# Define possible actions
actions = ['up', 'down', 'left', 'right']

# Terminal state
terminal = (N-1, N-1)

# Function to get next state
def get_next_state(i, j, action):
    if action == 'up':
        return max(0, i-1), j
    elif action == 'down':
        return min(N-1, i+1), j
    elif action == 'left':
        return i, max(0, j-1)
    elif action == 'right':
        return i, min(N-1, j+1)

# Reward function
def get_reward(ni, nj):
    return R_s[nj]

# Value iteration
while True:
    delta = 0.0
    V_new = V.clone()
    for i in range(N):
        for j in range(N):
            if (i, j) == terminal:
                continue
            max_val = float('-inf')
            for action in actions:
                ni, nj = get_next_state(i, j, action)
                val = get_reward(ni, nj) + discount * V[ni, nj]
                if val > max_val:
                    max_val = val
            V_new[i, j] = max_val
            delta = max(delta, abs(V_new[i, j] - V[i, j]))
    V = V_new
    if delta < theta:
        break

# Print the value function
print("Value function:")
print(V)