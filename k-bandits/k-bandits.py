import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, n_states=5, n_actions=10, seed=42, drift_std=0.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.drift_std = drift_std
        self.rng = np.random.default_rng(seed)
        self._initial_rewards = self.rng.normal(0, 1, (n_states, n_actions))
        self.true_rewards = self._initial_rewards.copy()

    def reset(self):
        self.true_rewards = self._initial_rewards.copy()

    def sample_state(self):
        return int(self.rng.integers(0, self.n_states))

    def step(self, state, action):
        reward = self.rng.normal(self.true_rewards[state, action], 1)
        if self.drift_std > 0.0:
            self.true_rewards += self.rng.normal(0, self.drift_std, self.true_rewards.shape)
        return reward

    def optimal_action(self, state):
        return int(np.argmax(self.true_rewards[state]))


class Agent:
    """
    Strategies:
      greedy    — always pick best estimated action per state
      egreedy   — pick random action with probability epsilon, else greedy
      optimistic — greedy on optimistic initial Q values (forces early exploration)

    Update rule (all strategies):
      alpha=None  → sample average: step size = 1/N  (equal weight to all past rewards)
      alpha=float → constant step:  step size = alpha (recent rewards weighted more)
    """

    def __init__(self, n_states, n_actions, strategy="greedy", epsilon=0.1,
                 alpha=None, optimistic_init=5.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.strategy = strategy
        self.epsilon = epsilon
        self.alpha = alpha
        self.optimistic_init = optimistic_init
        self.reset()

    def reset(self):
        if self.strategy == "optimistic":
            self.Q = np.full((self.n_states, self.n_actions), self.optimistic_init, dtype=float)
        else:
            self.Q = np.zeros((self.n_states, self.n_actions), dtype=float)
        self.N = np.zeros((self.n_states, self.n_actions), dtype=int)

    def select_action(self, state):
        if self.strategy in ("greedy", "optimistic"):
            return int(np.argmax(self.Q[state]))

        if self.strategy == "egreedy":
            if np.random.random() < self.epsilon:
                return np.random.randint(self.n_actions)
            return int(np.argmax(self.Q[state]))

        return np.random.randint(self.n_actions)

    def update(self, state, action, reward):
        self.N[state, action] += 1
        step = self.alpha if self.alpha is not None else 1.0 / self.N[state, action]
        self.Q[state, action] += step * (reward - self.Q[state, action])


def run_experiment(env, agent, n_steps=1000, n_runs=200):
    rewards_log = np.zeros((n_runs, n_steps))
    optimal_log = np.zeros((n_runs, n_steps))

    for run in range(n_runs):
        env.reset()
        agent.reset()
        for t in range(n_steps):
            state = env.sample_state()
            action = agent.select_action(state)
            reward = env.step(state, action)
            agent.update(state, action, reward)
            rewards_log[run, t] = reward
            optimal_log[run, t] = int(action == env.optimal_action(state))

    return rewards_log, optimal_log


def plot_results(results, n_steps, title_suffix="", filename="strategy_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Multi-Armed Bandit: Strategy Comparison{title_suffix}", fontsize=14)

    for label, (rewards, optimal) in results.items():
        axes[0].plot(rewards.mean(axis=0), label=label)
        axes[1].plot(optimal.mean(axis=0) * 100, label=label)

    axes[0].set_title("Average Reward over Time")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Average Reward")
    axes[0].legend()

    axes[1].set_title("% Optimal Action over Time")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("% Optimal Action")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.show()


def plot_reward_distributions(results, title_suffix="", filename="reward_distributions.png"):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    fig.suptitle(f"Final-Step Reward Distributions per Strategy{title_suffix}", fontsize=13)

    for ax, (label, (rewards, _)) in zip(axes, results.items()):
        final_rewards = rewards[:, -1]
        ax.hist(final_rewards, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(final_rewards.mean(), color="red", linestyle="--",
                   label=f"mean={final_rewards.mean():.2f}")
        ax.set_title(label)
        ax.set_xlabel("Reward")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.show()


def make_strategies(n_states, n_actions):
    return {
        "Greedy":                    Agent(n_states, n_actions, strategy="greedy"),
        "E-Greedy (e=0.1)":          Agent(n_states, n_actions, strategy="egreedy", epsilon=0.1),
        "Const. Step (a=0.1)":       Agent(n_states, n_actions, strategy="egreedy", epsilon=0.1, alpha=0.1),
        "Optimistic Init (Q0=5)":    Agent(n_states, n_actions, strategy="optimistic", optimistic_init=5.0),
    }


def run_scenario(label, drift_std, n_steps, n_runs, n_states, n_actions,
                 plot_suffix, cmp_file, dist_file):
    print(f"\n=== {label} (states={n_states}, actions={n_actions}, drift_std={drift_std}) ===")
    env = Environment(n_states=n_states, n_actions=n_actions, seed=0, drift_std=drift_std)
    strategies = make_strategies(n_states, n_actions)
    results = {}
    for name, agent in strategies.items():
        print(f"  Running: {name} ...")
        results[name] = run_experiment(env, agent, n_steps=n_steps, n_runs=n_runs)
    plot_results(results, n_steps, title_suffix=plot_suffix, filename=cmp_file)
    plot_reward_distributions(results, title_suffix=plot_suffix, filename=dist_file)
    print(f"  Saved: {cmp_file}, {dist_file}")


if __name__ == "__main__":
    N_STEPS = 1000
    N_RUNS = 500
    N_STATES = 5
    N_ACTIONS = 10

    run_scenario(
        label="Static Distribution",
        drift_std=0.0,
        n_steps=N_STEPS, n_runs=N_RUNS, n_states=N_STATES, n_actions=N_ACTIONS,
        plot_suffix=" — Static",
        cmp_file="strategy_comparison_static.png",
        dist_file="reward_distributions_static.png",
    )

    run_scenario(
        label="Fast Changing Distribution",
        drift_std=0.05,
        n_steps=N_STEPS, n_runs=N_RUNS, n_states=N_STATES, n_actions=N_ACTIONS,
        plot_suffix=" — Non-Stationary (drift=0.05)",
        cmp_file="strategy_comparison_nonstationary_fast.png",
        dist_file="reward_distributions_nonstationary_fast.png",
    )