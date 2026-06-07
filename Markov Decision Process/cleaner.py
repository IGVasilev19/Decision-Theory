import gymnasium as gym
import numpy as np
from gymnasium import spaces

DEAD, HIGH, LOW = 0, 1, 2
CHARGE, WORK = 0, 1

STATE_NAMES = ["dead", "high", "low"]
ACTION_NAMES = ["charge", "work"]

TRANSITIONS = {
    LOW: {
        WORK: [(0.8, LOW, 20), (0.2, DEAD, -1000)],
        CHARGE: [(1.0, HIGH, -1)],
    },
    HIGH: {
        WORK: [(0.95, HIGH, 20), (0.05, LOW, 20)],
        CHARGE: [(1.0, HIGH, -2)],
    },
    DEAD: {
        WORK: [(1.0, DEAD, 0)],
        CHARGE: [(1.0, DEAD, 0)],
    },
}


class CleanerEnv(gym.Env):
    """Cleaning-robot MDP. Robot has battery (high/low) or is dead."""

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


def q_value(state, action, state_values, gamma):
    """Expected return of taking `action` in `state`: reward + discounted future value."""
    return sum(
        probability * (reward + gamma * state_values[next_state])
        for probability, next_state, reward in TRANSITIONS[state][action]
    )


def value_iteration(gamma=0.9, convergence_threshold=1e-3):
    state_values = np.zeros(3)
    sweeps = 0
    while True:
        sweeps += 1
        largest_change = 0.0
        for state in (HIGH, LOW):
            previous_value = state_values[state]
            state_values[state] = max(
                q_value(state, action, state_values, gamma) for action in (CHARGE, WORK)
            )
            largest_change = max(largest_change, abs(previous_value - state_values[state]))

        if largest_change < convergence_threshold:
            break

    best_action_per_state = {
        state: max(
            (CHARGE, WORK),
            key=lambda action: q_value(state, action, state_values, gamma),
        )
        for state in (HIGH, LOW)
    }

    return state_values, best_action_per_state, sweeps


def q_learning(
    env,
    episodes=200,
    gamma=0.9,
    learning_rate=0.1,
    exploration_rate=0.1,
    max_steps=200,
):
    """Model-free: learn Q-values by interacting with env, no access to TRANSITIONS."""
    q_table = np.zeros((3, 2))

    for _ in range(episodes):
        env.start_state = int(env.np_random.choice([HIGH, LOW])) 
        state, _ = env.reset()
        for _ in range(max_steps):
            if env.np_random.random() < exploration_rate:
                action = env.np_random.integers(2)
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, is_dead, _, _ = env.step(action)

            best_next_value = 0.0 if is_dead else np.max(q_table[next_state])
            target = reward + gamma * best_next_value
            q_table[state][action] += learning_rate * (target - q_table[state][action])

            state = next_state
            if is_dead:
                break

    learned_values = {state: float(np.max(q_table[state])) for state in (HIGH, LOW)}
    learned_policy = {state: int(np.argmax(q_table[state])) for state in (HIGH, LOW)}
    return q_table, learned_values, learned_policy


def rollout(env, policy, gamma=0.9, max_steps=200):
    state, _ = env.reset()
    discounted_return = 0.0
    discount = 1.0
    for _ in range(max_steps):
        if state == DEAD:
            break
        state, reward, is_dead, _, _ = env.step(policy[state])
        discounted_return += discount * reward
        discount *= gamma
        if is_dead:
            break
    return discounted_return


if __name__ == "__main__":
    optimal_values, policy, sweeps = value_iteration()
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
    _, learned_values, learned_policy = q_learning(learning_env)
    for state in (HIGH, LOW):
        print(
            f"  {STATE_NAMES[state]:5s} Q = {learned_values[state]:7.3f}  "
            f"policy = {ACTION_NAMES[learned_policy[state]]}  "
            f"(VI: V = {optimal_values[state]:7.3f}, {ACTION_NAMES[policy[state]]})"
        )
