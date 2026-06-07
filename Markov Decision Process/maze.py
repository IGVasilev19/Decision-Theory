import numpy as np
import matplotlib.pyplot as plt

# 6 wide (x = 1..6), 4 tall (y = 1..4). Every cell is a state.
WIDTH, HEIGHT = 6, 4
START = (1, 1)
GOAL = (4, 3)  # the one cell that lets the robot escape (terminal)

STEP_REWARD = -1   # each move costs 1 -> pushes toward the shortest path
GOAL_REWARD = 10   # reward for stepping onto the escape cell

# action -> (dx, dy). y grows upward.
ACTIONS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
ARROWS = {"up": "^", "down": "v", "left": "<", "right": ">"}

STATES = [(x, y) for x in range(1, WIDTH + 1) for y in range(1, HEIGHT + 1)]


def in_bounds(x, y):
    return 1 <= x <= WIDTH and 1 <= y <= HEIGHT


def valid_actions(state):
    # Cells on the walls expose only the moves that stay inside the grid.
    x, y = state
    return [a for a, (dx, dy) in ACTIONS.items() if in_bounds(x + dx, y + dy)]


def step_model(state, action):
    # Deterministic move. Returns (next_state, reward, done).
    dx, dy = ACTIONS[action]
    next_state = (state[0] + dx, state[1] + dy)
    if next_state == GOAL:
        return next_state, GOAL_REWARD, True
    return next_state, STEP_REWARD, False


def q_value(state, action, state_values, future_reward_discount):
    next_state, reward, done = step_model(state, action)
    future = 0.0 if done else state_values[next_state]
    return reward + future_reward_discount * future


def value_iteration(future_reward_discount, convergence_threshold):
    state_values = {state: 0.0 for state in STATES}
    sweeps = 0
    while True:
        sweeps += 1
        largest_change = 0.0
        for state in STATES:
            if state == GOAL:
                continue  # terminal: value stays 0, no actions
            previous_value = state_values[state]
            state_values[state] = max(
                q_value(state, action, state_values, future_reward_discount)
                for action in valid_actions(state)
            )
            largest_change = max(largest_change, abs(previous_value - state_values[state]))
        if largest_change < convergence_threshold:
            break

    best_action_per_state = {
        state: max(
            valid_actions(state),
            key=lambda action: q_value(state, action, state_values, future_reward_discount),
        )
        for state in STATES
        if state != GOAL
    }
    return state_values, best_action_per_state, sweeps


def choose_action(state, q_table, epsilon, rng):
    actions = valid_actions(state)
    if rng.random() < epsilon:
        return actions[int(rng.integers(len(actions)))]
    return max(actions, key=lambda action: q_table[state][action])


def q_learning(episodes, future_reward_discount, epsilon_start, epsilon_end,
               epsilon_decay, max_steps, learning_rate, seed=0):
    rng = np.random.default_rng(seed)
    q_table = {
        state: {action: 0.0 for action in valid_actions(state)}
        for state in STATES if state != GOAL
    }
    epsilon = epsilon_start

    for _ in range(episodes):
        state = START
        for _ in range(max_steps):
            action = choose_action(state, q_table, epsilon, rng)
            next_state, reward, done = step_model(state, action)

            best_next_value = 0.0 if done else max(q_table[next_state].values())
            td_target = reward + future_reward_discount * best_next_value
            q_table[state][action] += learning_rate * (td_target - q_table[state][action])

            state = next_state
            if done:
                break
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    learned_values = {state: max(actions.values()) for state, actions in q_table.items()}
    learned_policy = {
        state: max(actions, key=actions.get) for state, actions in q_table.items()
    }
    return q_table, learned_values, learned_policy


def print_value_grid(title, state_values):
    print(f"\n{title}")
    for y in range(HEIGHT, 0, -1):
        cells = []
        for x in range(1, WIDTH + 1):
            if (x, y) == GOAL:
                cells.append("  GOAL ")
            else:
                cells.append(f"{state_values[(x, y)]:7.2f}")
        print(" ".join(cells))


def print_policy_grid(title, policy):
    print(f"\n{title}")
    for y in range(HEIGHT, 0, -1):
        cells = []
        for x in range(1, WIDTH + 1):
            if (x, y) == GOAL:
                cells.append("G")
            else:
                cells.append(ARROWS[policy[(x, y)]])
        print("  ".join(cells))


def _draw_maze(ax, values, policy, title, vmin, vmax):
    grid = np.full((HEIGHT, WIDTH), np.nan)
    for (x, y), value in values.items():
        grid[y - 1, x - 1] = value
    grid[GOAL[1] - 1, GOAL[0] - 1] = GOAL_REWARD

    im = ax.imshow(grid, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    for (x, y), action in policy.items():
        dx, dy = ACTIONS[action]
        ax.arrow(x - 1, y - 1, dx * 0.3, dy * 0.3, head_width=0.12,
                 color="white", length_includes_head=True)
    ax.scatter(GOAL[0] - 1, GOAL[1] - 1, marker="*", s=300, color="red", label="escape")
    ax.scatter(START[0] - 1, START[1] - 1, marker="o", s=120, color="orange", label="start")
    ax.set_xticks(range(WIDTH)); ax.set_xticklabels(range(1, WIDTH + 1))
    ax.set_yticks(range(HEIGHT)); ax.set_yticklabels(range(1, HEIGHT + 1))
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)
    return im


def plot_comparison(vi_values, vi_policy, ql_values, ql_policy):
    all_values = list(vi_values.values()) + list(ql_values.values()) + [GOAL_REWARD]
    vmin, vmax = min(all_values), max(all_values)

    fig, (ax_vi, ax_ql) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Maze value + policy: value iteration vs Q-learning",
                 fontsize=13, fontweight="bold")
    _draw_maze(ax_vi, vi_values, vi_policy,
               "Value iteration (knows the map)", vmin, vmax)
    im = _draw_maze(ax_ql, ql_values, ql_policy,
                    "Q-learning (learned from (1,1) by trial)", vmin, vmax)
    ax_vi.legend(loc="upper left")
    fig.colorbar(im, ax=(ax_vi, ax_ql), label="V(state) = max_a Q — expected discounted return",
                 shrink=0.8)
    plt.savefig("maze.png", dpi=120)
    print("\nSaved maze comparison plot to maze.png")
    plt.show()


if __name__ == "__main__":
    future_reward_discount = 0.9  # shared by both algorithms

    vi_values, vi_policy, sweeps = value_iteration(
        future_reward_discount=future_reward_discount,
        convergence_threshold=1e-6,
    )
    print(f"Value iteration converged in {sweeps} sweeps")
    print_value_grid("V(state) [value iteration]", vi_values)
    print_policy_grid("Optimal policy [value iteration]", vi_policy)
    print(f"\nV(start {START}) = {vi_values[START]:.3f}")

    _, ql_values, ql_policy = q_learning(
        episodes=5000,
        future_reward_discount=future_reward_discount,
        epsilon_start=0.9,
        epsilon_end=0.01,
        epsilon_decay=0.998,
        max_steps=100,
        learning_rate=0.1,
    )
    print_policy_grid("Learned policy [q-learning]", ql_policy)

    agree = sum(vi_policy[s] == ql_policy[s] for s in vi_policy)
    print(f"\nPolicy agreement: {agree}/{len(vi_policy)} cells match value iteration")

    plot_comparison(vi_values, vi_policy, ql_values, ql_policy)
