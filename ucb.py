import numpy as np

class UCB:
    
# =============================================================================
#     Base UCB class
# =============================================================================
    
    def __init__(self, horizon):
        self.s = [1, 1, 1] # successes
        self.f = [1, 1, 1] # failures
        self.R = 0 # total reward
        self.Rt = np.zeros(horizon) # rewards per time step
        
    def compute_ucb(self, arm_mean, arm_pulls, time_step):
        raise NotImplementedError
    
    def select_arm(self, t):
        scores = []
        for i in range(len(self.s)):
            pulls = self.s[i]+self.f[i]
            mean = self.s[i]/pulls
            ucb_score = self.compute_ucb(mean, pulls, t)
            scores.append(ucb_score)
        return np.argmax(scores)
    
    def update(self, i, r, t):
        # update the model with reward r obtained for arm i at time step t
        if r == 1:
            self.s[i] += 1
            self.R += 1
        else:
            self.f[i] += 1
        self.Rt[t] += self.R
            
    def arm_pulls(self):
        return [self.s[i]+self.f[i] for i in range(len(self.s))]
    
    def empirical_means(self):
        return [self.s[i]/(self.s[i]+self.f[i]) for i in range(len(self.s))]