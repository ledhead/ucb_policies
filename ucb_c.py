import numpy as np
from ucb import UCB

class UCB_C (UCB):
    
# =============================================================================
#     UCB-C: derived from Chebyshev's inequality
# =============================================================================
    
    def compute_ucb(self, arm_mean, arm_pulls, time_step):
        return arm_mean + np.power(time_step, 2) * \
                np.sqrt(arm_mean*(1-arm_mean)*arm_pulls)/arm_pulls