import numpy as np
from ucb import UCB

class UCB_H (UCB):
    
# =============================================================================
#     UCB-H: derived from Hoeffding's inequality (equivalent to UCB1)
# =============================================================================
    
    def compute_ucb(self, arm_mean, arm_pulls, time_step):
        return arm_mean + np.sqrt(2*np.log(time_step)/arm_pulls)