import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces

DEAD, HIGH, LOW = 0, 1, 2
CHARGE, WORK = 0, 1

STATE_NAMES = ["dead", "high", "low"]
ACTION_NAMES = ["charge", "work"]

TRANSITIONS = {LOW: {WORK: [(0.8, LOW, 20), (0.2, DEAD, -1000)], CHARGE: [(1.0, HIGH, -1)],},HIGH: {WORK: [(0.95, HIGH, 20), (0.05, LOW, 20)], CHARGE: [(1.0, HIGH, -2)],},DEAD: {WORK: [(1.0, DEAD, 0)],CHARGE: [(1.0, DEAD, 0)],},}


class CleanerEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, start_state=HIGH, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)
        self.start_state = start_state
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = self.start_state
        return self.current_state, {}

    def step(self, action):
        outcomes = TRANSITIONS[self.current_state][action]
        probabilities = [probability for probability, _, _ in outcomes]
        chosen_index = self.np_random.choice(len(outcomes), p=probabilities)
        _, next_state, reward = outcomes[chosen_index]

        self.current_state = int(next_state)
        is_dead = self.current_state == DEAD
        if self.render_mode == "human":
            self.render()
        return self.current_state, reward, is_dead, False, {}

    def render(self):
        print(f"state = {STATE_NAMES[self.current_state]}")


def q_value(state, action, state_values, future_reward_discount):
    return sum(
        probability * (reward + future_reward_discount * state_values[next_state])
        for probability, next_state, reward in TRANSITIONS[state][action]
    )


def value_iteration(future_reward_discount, convergence_threshold):
    state_values = np.zeros(3)
    sweeps = 0
    value_history = [state_values.copy()]
    while True:
        sweeps += 1
        largest_change = 0.0
        for state in (HIGH, LOW):
            previous_value = state_values[state]
            state_values[state] = max(
                q_value(state, action, state_values, future_reward_discount) for action in (CHARGE, WORK)
            )
            largest_change = max(largest_change, abs(previous_value - state_values[state]))

        value_history.append(state_values.copy())
        if largest_change < convergence_threshold:
            break

    best_action_per_state = {
        state: max(
            (CHARGE, WORK),
            key=lambda action: q_value(state, action, state_values, future_reward_discount),
        )
        for state in (HIGH, LOW)
    }

    return state_values, best_action_per_state, sweeps, np.array(value_history)


def choose_action(q_row, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(2))
    return int(np.argmax(q_row))


def q_learning(env, episodes, future_reward_discount, epsilon_start, epsilon_end, epsilon_decay, max_steps):
    q_table = np.zeros((3, 2))
    epsilon = epsilon_start
    value_history = [q_table.max(axis=1).copy()]
    epsilon_history = [epsilon]

    for _ in range(episodes):
        env.start_state = int(env.np_random.choice([HIGH, LOW]))
        state, _ = env.reset()
        for _ in range(max_steps):
            action = choose_action(q_table[state], epsilon, env.np_random)
            next_state, reward, is_dead, _, _ = env.step(action)

            learning_rate = 0.1
            best_next_value = 0.0 if is_dead else q_table[next_state].max()
            td_target = reward + future_reward_discount * best_next_value
            q_table[state, action] += learning_rate * (td_target - q_table[state, action])

            state = next_state
            if is_dead:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        value_history.append(q_table.max(axis=1).copy())
        epsilon_history.append(epsilon)

    learned_values = {state: float(np.max(q_table[state])) for state in (HIGH, LOW)}
    learned_policy = {state: int(np.argmax(q_table[state])) for state in (HIGH, LOW)}
    return (
        q_table,
        learned_values,
        learned_policy,
        np.array(value_history),
        np.array(epsilon_history),
    )


def rollout(env, policy, future_reward_discount=0.9, max_steps=200):
    state, _ = env.reset()
    discounted_return = 0.0
    discount = 1.0
    for _ in range(max_steps):
        if state == DEAD:
            break
        state, reward, is_dead, _, _ = env.step(policy[state])
        discounted_return += discount * reward
        discount *= future_reward_discount
        if is_dead:
            break
    return discounted_return


def plot_convergence(vi_history, ql_history, ql_epsilon, optimal_values):
    colors = {HIGH: "tab:blue", LOW: "tab:orange"}
    fig, (ax_vi, ax_ql) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Value convergence: model-based (value iteration) vs model-free (Q-learning)",
                 fontsize=13, fontweight="bold")

    # Value iteration: V per state per sweep.
    for state in (HIGH, LOW):
        ax_vi.plot(vi_history[:, state], color=colors[state], marker="o", markersize=3,
                   label=f"V({STATE_NAMES[state]}) — value-iteration estimate per sweep")
    for state in (HIGH, LOW):
        ax_vi.axhline(optimal_values[state], color=colors[state], linestyle="--", alpha=0.6,
                      label=f"V({STATE_NAMES[state]}) = {optimal_values[state]:.2f} (converged optimum)")
    ax_vi.set_title("Value iteration (knows transition probabilities)")
    ax_vi.set_xlabel("sweep (full Bellman backup over all states)")
    ax_vi.set_ylabel("V(state) — expected discounted return")
    ax_vi.legend(loc="lower right", fontsize=8, title="value iteration")
    ax_vi.grid(True, alpha=0.3)

    # Q-learning: max-Q per state per episode, with VI optimum as dashed target.
    ql_lines = []
    for state in (HIGH, LOW):
        line, = ax_ql.plot(ql_history[:, state], color=colors[state],
                           label=f"V({STATE_NAMES[state]}) = max_a Q — learned estimate per episode")
        ql_lines.append(line)
    for state in (HIGH, LOW):
        line = ax_ql.axhline(optimal_values[state], color=colors[state], linestyle="--", alpha=0.7,
                             label=f"V({STATE_NAMES[state]}) = {optimal_values[state]:.2f} (value-iteration target)")
        ql_lines.append(line)
    ax_ql.set_title("Q-learning (learns by trial, no probabilities given)")
    ax_ql.set_xlabel("episode")
    ax_ql.set_ylabel("V(state) = max_a Q(state, a)")
    ax_ql.grid(True, alpha=0.3)

    # Epsilon decay on a twin axis so exploration schedule is visible.
    ax_eps = ax_ql.twinx()
    eps_line, = ax_eps.plot(ql_epsilon, color="gray", alpha=0.6, linestyle=":",
                            label="epsilon — exploration rate (decays each episode)")
    ax_eps.set_ylabel("epsilon (P[random action])", color="gray")
    ax_eps.set_ylim(0, 1)
    ax_eps.tick_params(axis="y", labelcolor="gray")

    # Single combined legend so Q-value curves and epsilon read together.
    ax_ql.legend(handles=ql_lines + [eps_line], loc="lower right", fontsize=8,
                 title="Q-learning")

    fig.tight_layout()
    plt.savefig("convergence.png", dpi=120)
    print("\nSaved convergence plot to convergence.png")
    plt.show()


if __name__ == "__main__":
    future_reward_discount = 0.9  # discount factor: shared by both algorithms (see explanation)

    optimal_values, policy, sweeps, vi_history = value_iteration(
        future_reward_discount=future_reward_discount,
        convergence_threshold=1e-3,
    )
    print(f"Converged in {sweeps} sweeps")
    for state in (HIGH, LOW):
        print(
            f"  {STATE_NAMES[state]:5s} V = {optimal_values[state]:7.3f}  "
            f"policy = {ACTION_NAMES[policy[state]]}"
        )

    num_runs = 200
    for start_state in (HIGH, LOW):
        env = CleanerEnv(start_state=start_state)
        average_return = np.mean([rollout(env, policy) for _ in range(num_runs)])
        print(
            f"Rollout from {STATE_NAMES[start_state]:5s}: "
            f"avg = {average_return:7.3f}  (V = {optimal_values[start_state]:7.3f})"
        )

    print("\nQ-learning (model-free):")
    learning_env = CleanerEnv()
    learning_env.reset(seed=0)
    _, learned_values, learned_policy, ql_history, ql_epsilon = q_learning(
        learning_env,
        episodes=91,
        future_reward_discount=future_reward_discount,
        epsilon_start=0.9,
        epsilon_end=0.01,
        epsilon_decay=0.998,
        max_steps=91,
    )
    for state in (HIGH, LOW):
        print(
            f"  {STATE_NAMES[state]:5s} Q = {learned_values[state]:7.3f}  "
            f"policy = {ACTION_NAMES[learned_policy[state]]}  "
            f"(VI: V = {optimal_values[state]:7.3f}, {ACTION_NAMES[policy[state]]})"
        )

    plot_convergence(vi_history, ql_history, ql_epsilon, optimal_values)
