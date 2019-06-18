import numpy as np
import matplotlib.pyplot as plt
from ucb_c import UCB_C
from ucb_h import UCB_H

# settings
np.random.seed = 10
arms = [0.01, 0.05, 0.10]
simulations = 100
horizon = 1000
policies = [UCB_H(horizon), UCB_C(horizon)]

def sample(i):
    return 1 if np.random.rand() < arms[i] else 0

def plot(data, label):
    plt.rcParams.update({'font.size': 14,
                         'axes.titlesize': 12})
    plt.plot(np.arange(0, horizon), data, label=label)
    plt.xlabel('horizon')
    plt.ylabel('cum. reward')
    plt.legend(loc='best')
    plt.title('UCB-H vs. UCB-C')
    
for policy in policies:
    R = 0 # total reward
    Rt = np.zeros(horizon) # total rewards per time step
    empirical_means = [0, 0, 0]
    arm_pulls = [0, 0, 0]
    print(f'policy: {policy.__class__.__name__}')
    for i in range(simulations):
        print("\rrun {}... ".format(i + 1), end="", flush=True)
        policy.__init__(horizon)
        for t in range(horizon):
            i = policy.select_arm(t+1)
            r = sample(i)
            policy.update(i, r, t)
        for i in range(len(arms)):
            empirical_means[i] += policy.empirical_means()[i]
            arm_pulls[i] += policy.arm_pulls()[i]
        R += policy.R
        Rt += policy.Rt
    print(f'\rempirical means: {np.array(empirical_means)/simulations}')
    print(f'arm pulls: {np.array(arm_pulls)/simulations}')
    print(f'total reward: {R/simulations}\n')
    plot(Rt/simulations, policy.__class__.__name__)
    