"""Finite bandit agents."""

import numpy as np
import random as rd
from scipy.stats import beta

from base.agent import Agent
from base.agent import random_argmax

_SMALL_NUMBER = 1e-10
##############################################################################


class FiniteBernoulliBanditEpsilonGreedy(Agent):
  """Simple agent made for finite armed bandit problems."""

  def __init__(self, n_arm, a0=1, b0=1, epsilon=0.0):
    self.n_arm = n_arm
    self.epsilon = epsilon
    self.prior_success = np.array([a0 for arm in range(n_arm)])
    self.prior_failure = np.array([b0 for arm in range(n_arm)])

  def set_prior(self, prior_success, prior_failure):
    # Overwrite the default prior
    self.prior_success = np.array(prior_success)
    self.prior_failure = np.array(prior_failure)

  def get_posterior_mean(self):
    return self.prior_success / (self.prior_success + self.prior_failure)

  def get_posterior_sample(self):
    return np.random.beta(self.prior_success, self.prior_failure)

  def update_observation(self, observation, action, reward):
    # Naive error checking for compatibility with environment
    assert observation == self.n_arm

    if np.isclose(reward, 1):
      self.prior_success[action] += 1
    elif np.isclose(reward, 0):
      self.prior_failure[action] += 1
    else:
      raise ValueError('Rewards should be 0 or 1 in Bernoulli Bandit')

  def pick_action(self, observation):
    """Take random action prob epsilon, else be greedy."""
    if np.random.rand() < self.epsilon:
      action = np.random.randint(self.n_arm)
    else:
      posterior_means = self.get_posterior_mean()
      action = random_argmax(posterior_means)

    return action


##############################################################################


class FiniteBernoulliBanditTS(FiniteBernoulliBanditEpsilonGreedy):
  """Thompson sampling on finite armed bandit."""

  def __init__(self, n_arm, a0=1, b0=1, epsilon=0.0):
    super().__init__(n_arm, a0=1, b0=1, epsilon=0.0)
    self.count = 0

  def pick_action(self, observation):
    """Thompson sampling with Beta posterior for action selection."""
    # self.count += 1
    # print('TS Count:', self.count)
    sampled_means = self.get_posterior_sample()
    action = random_argmax(sampled_means)
    return action

##############################################################################

class FiniteBernoulliBanditUCB(FiniteBernoulliBanditEpsilonGreedy):

  def __init__(self, n_arm, a0=1, b0=1, epsilon=0.0):
    super().__init__(n_arm, a0=1, b0=1, epsilon=0.0)
    self.c = 3
    self.count = 0

  def pick_action(self, observation):
    # Pick the best one with consideration of upper confidence bounds.
    self.count += 1
    # print("Count:", self.count)
    i = max(
        range(self.n_arm),
        key=lambda x: self.prior_success[x] / float(self.prior_success[x] + self.prior_failure[x]) + beta.std(
            self.prior_success[x], self.prior_failure[x]) * self.c
    )
    return i

##############################################################################

class FiniteBernoulliBanditIDS(FiniteBernoulliBanditEpsilonGreedy):
  """IDS"""

  def __init__(self, n_arm, a0=1, b0=1, epsilon=0.0):
    super().__init__(n_arm, a0=1, b0=1, epsilon=0.0)
    self.optimal_arm = None
    self.flag = False
    self.threshold = 0.99

  def init_prior(self, a0=1, a1=1):
        """
        Init Beta prior
        :param a0: int, multiplicative factor for alpha
        :param a1: int, multiplicative factor for beta
        :return: np.arrays, prior values (alpha, beta) for each earm
        """
        beta1 = a0 * np.ones(self.n_arm).astype(int)
        beta2 = a1 * np.ones(self.n_arm).astype(int)
        return beta1, beta2

  def pick_action(self, observation, M=10000, VIDS=False):
    beta1, beta2 = self.init_prior()
    Maap, p_a = np.zeros((self.n_arm, self.n_arm)), np.zeros(self.n_arm)
    thetas = np.array([np.random.beta(beta1[arm], beta2[arm], M) for arm in range(self.n_arm)])
    if not self.flag:
        if np.max(p_a) >= self.threshold:
            # Stop learning policy
            self.flag = True
            arm = self.optimal_arm
        else:
            arm, p_a = self.computeIDS(Maap, p_a, thetas, M, VIDS)
    else:
        arm = self.optimal_arm
    return arm
  
  def computeIDS(self, Maap, p_a, thetas, M, VIDS=False):
    mu, theta_hat = np.mean(thetas, axis=1), np.argmax(thetas, axis=0)
    for a in range(self.n_arm):
        mu[a] = np.mean(thetas[a])
        for ap in range(self.n_arm):
            t = thetas[ap, np.where(theta_hat == a)]
            Maap[ap, a] = np.nan_to_num(np.mean(t))
            if ap == a:
                p_a[a] = t.shape[1]/M
    if np.max(p_a) >= self.threshold:
        # Stop learning policy
        self.optimal_arm = np.argmax(p_a)
        arm = self.optimal_arm
    else:
        rho_star = sum([p_a[a] * Maap[a, a] for a in range(self.n_arm)])
        delta = rho_star - mu
        if VIDS:
            v = np.array([sum([p_a[ap] * (Maap[a, ap] - mu[a]) ** 2 for ap in range(self.n_arm)])
                          for a in range(self.n_arm)])
            arm = self.rd_argmax(-delta ** 2 / v)
        else:
            g = np.array([sum([p_a[ap] * (Maap[a, ap] * np.log(Maap[a, ap]/mu[a]+1e-10) +
                                          (1-Maap[a, ap]) * np.log((1-Maap[a, ap])/(1-mu[a])+1e-10))
                                for ap in range(self.n_arm)]) for a in range(self.n_arm)])
            arm = self.IDSAction(delta, g)
    return arm, p_a

  def IDSAction(self, delta, g):
    Q = np.zeros((self.n_arm, self.n_arm))
    IR = np.ones((self.n_arm, self.n_arm)) * np.inf
    q = np.linspace(0, 1, 1000)
    for a in range(self.n_arm - 1):
        for ap in range(a + 1, self.n_arm):
            if g[a] < 1e-6 or g[ap] < 1e-6:
                return self.rd_argmax(-g)
            da, dap, ga, gap = delta[a], delta[ap], g[a], g[ap]
            qaap = q[self.rd_argmax(-(q * da + (1 - q) * dap) ** 2 / (q * ga + (1 - q) * gap))]
            IR[a, ap] = (qaap * (da - dap) + dap) ** 2 / (qaap * (ga - gap) + gap)
            Q[a, ap] = qaap
    amin = self.rd_argmax(-IR.reshape(self.n_arm * self.n_arm))
    a, ap = amin // self.n_arm, amin % self.n_arm
    b = np.random.binomial(1, Q[a, ap])
    arm = int(b * a + (1 - b) * ap)
    return arm

  def rd_argmax(self, vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return rd.choice(indices)

##############################################################################


class FiniteBernoulliBanditBootstrap(FiniteBernoulliBanditTS):
  """Bootstrapped Thompson sampling on finite armed bandit."""

  def get_posterior_sample(self):
    """Use bootstrap resampling instead of posterior sample."""
    total_tries = self.prior_success + self.prior_failure
    prob_success = self.prior_success / total_tries
    boot_sample = np.random.binomial(total_tries, prob_success) / total_tries
    return boot_sample

##############################################################################


class FiniteBernoulliBanditLaplace(FiniteBernoulliBanditTS):
  """Laplace Thompson sampling on finite armed bandit."""

  def get_posterior_sample(self):
    """Gaussian approximation to posterior density (match moments)."""
    (a, b) = (self.prior_success + 1e-6 - 1, self.prior_failure + 1e-6 - 1)
    # The modes are not well defined unless alpha, beta > 1
    assert np.all(a > 0)
    assert np.all(b > 0)
    mode = a / (a + b)
    hessian = a / mode + b / (1 - mode)
    laplace_sample = mode + np.sqrt(1 / hessian) * np.random.randn(self.n_arm)
    return laplace_sample

##############################################################################


class DriftingFiniteBernoulliBanditTS(FiniteBernoulliBanditTS):
  """Thompson sampling on finite armed bandit."""

  def __init__(self, n_arm, a0=1, b0=1, gamma=0.01):
    self.n_arm = n_arm
    self.a0 = a0
    self.b0 = b0
    self.prior_success = np.array([a0 for arm in range(n_arm)])
    self.prior_failure = np.array([b0 for arm in range(n_arm)])
    self.gamma = gamma

  def update_observation(self, observation, action, reward):
    # Naive error checking for compatibility with environment
    assert observation == self.n_arm

    # All values decay slightly, observation updated
    self.prior_success = self.prior_success * (
        1 - self.gamma) + self.a0 * self.gamma
    self.prior_failure = self.prior_failure * (
        1 - self.gamma) + self.b0 * self.gamma
    self.prior_success[action] += reward
    self.prior_failure[action] += 1 - reward

##############################################################################
class FiniteBernoulliBanditLangevin(FiniteBernoulliBanditTS):
  '''Langevin method for approximate posterior sampling.'''
  
  def __init__(self,n_arm, step_count=100,step_size=0.01,a0=1, b0=1, epsilon=0.0):
    FiniteBernoulliBanditTS.__init__(self,n_arm, a0, b0, epsilon)
    self.step_count = step_count
    self.step_size = step_size
  
  def project(self,x):
    '''projects the vector x onto [_SMALL_NUMBER,1-_SMALL_NUMBER] to prevent
    numerical overflow.'''
    return np.minimum(1-_SMALL_NUMBER,np.maximum(x,_SMALL_NUMBER))
  
  def compute_gradient(self,x):
    grad = (self.prior_success-1)/x - (self.prior_failure-1)/(1-x)
    return grad
    
  def compute_preconditioners(self,x):
    second_derivatives = (self.prior_success-1)/(x**2) + (self.prior_failure-1)/((1-x)**2)
    second_derivatives = np.maximum(second_derivatives,_SMALL_NUMBER)
    preconditioner = np.diag(1/second_derivatives)
    preconditioner_sqrt = np.diag(1/np.sqrt(second_derivatives))
    return preconditioner,preconditioner_sqrt
    
  def get_posterior_sample(self):
        
    (a, b) = (self.prior_success + 1e-6 - 1, self.prior_failure + 1e-6 - 1)
    # The modes are not well defined unless alpha, beta > 1
    assert np.all(a > 0)
    assert np.all(b > 0)
    x_map = a / (a + b)
    x_map = self.project(x_map)
    
    preconditioner, preconditioner_sqrt=self.compute_preconditioners(x_map)
   
    x = x_map
    for i in range(self.step_count):
      g = self.compute_gradient(x)
      scaled_grad = preconditioner.dot(g)
      scaled_noise= preconditioner_sqrt.dot(np.random.randn(self.n_arm)) 
      x = x + self.step_size*scaled_grad + np.sqrt(2*self.step_size)*scaled_noise
      x = self.project(x)
      
    return x
      
    
    