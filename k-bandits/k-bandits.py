import numpy as np
import matplotlib.pyplot as plt

K = 3                       
TRUE_MEANS = [5, 6, 7]      
SD = 2                      
STEPS = 1000                
RUNS = 500                  
EPSILON = 0.1               
ALPHA = 0.1                 
WALK_SD = 0.5               
WALK_EVERY = 10             

means = np.array(TRUE_MEANS, dtype=float)
Q = np.zeros(K)
counts = np.zeros(K)


def greedy():
    action = np.argmax(Q)

    reward = np.random.normal(loc=means[action], scale=SD)

    counts[action] += 1
    Q[action] += (1 / counts[action]) * (reward - Q[action])

    return action, reward


def e_greedy(explore_chance):
    if np.random.random() <= explore_chance:
        action = np.random.randint(K)
    else:
        action = np.argmax(Q)

    reward = np.random.normal(loc=means[action], scale=SD)

    counts[action] += 1
    Q[action] += (1 / counts[action]) * (reward - Q[action])

    return action, reward


def const_step(explore_chance, alpha):
    if np.random.random() <= explore_chance:
        action = np.random.randint(K)
    else:
        action = np.argmax(Q)

    reward = np.random.normal(loc=means[action], scale=SD)

    counts[action] += 1
    Q[action] += alpha * (reward - Q[action])

    return action, reward


def run_policy(step_fn, init_q, non_stationary, steps=STEPS):
    """step_fn() is a no-arg call doing one pull, returning (action, reward)."""
    global means, Q, counts
    total = np.zeros(steps)

    for _ in range(RUNS):
        means = np.array(TRUE_MEANS, dtype=float)
        Q = np.full(K, float(init_q))
        counts = np.zeros(K)

        for t in range(steps):
            if non_stationary and t > 0 and t % WALK_EVERY == 0:
                means += np.random.normal(0.0, WALK_SD, size=K)

            _, reward = step_fn()
            total[t] += reward

    return total / RUNS


POLICIES = [
    ("Greedy (sample-avg)", greedy),
    ("Epsilon-greedy (sample-avg)", lambda: e_greedy(EPSILON)),
    ("Const step size", lambda: const_step(EPSILON, ALPHA)),
]


def make_graph(ax, title, init_q, non_stationary, steps=STEPS):
    for label, step_fn in POLICIES:
        avg = run_policy(step_fn, init_q, non_stationary, steps)
        ax.plot(avg, label=label)
    ax.axhline(max(TRUE_MEANS), color="gray", ls="--", lw=1, label="best mean")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Average reward")
    ax.legend(fontsize=8)


SHORT_STEPS = 50

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

make_graph(axes[0, 0], "Stationary, Q init = 0",       init_q=0, non_stationary=False)
make_graph(axes[0, 1], "Stationary, optimistic Q = 5", init_q=5, non_stationary=False)
make_graph(axes[1, 0], f"Non-stationary (means drift every {WALK_EVERY} steps), Q init = 0",
           init_q=0, non_stationary=True)
make_graph(axes[1, 1], f"Stationary, Q init = 0, only {SHORT_STEPS} steps",
           init_q=0, non_stationary=False, steps=SHORT_STEPS)

plt.tight_layout()
plt.savefig("bandit_comparison.png", dpi=120)
print("Saved bandit_comparison.png")
plt.show()
