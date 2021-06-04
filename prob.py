from scipy.stats import norm
from numpy import linalg as LA
import numpy as np
from math import log
from scipy import optimize
from loss import NewBrownian
from loss import Linear

def lin_alpha(g : Linear, pF : float) -> float:
    return LA.norm(g.a) * norm.ppf(1.0 - pF)

def lin_pf(g : Linear, alpha : float) -> float:
    return 1.0 - norm.cdf(alpha / LA.norm(g.a))

# only is accurate for small p_F
def new_brown_pF(g : NewBrownian, alpha: float) -> float:
    return -sum(list(map(
        lambda sigma_i: log(norm.cdf(alpha / sigma_i)),
        g.b
    )))

# only is accurate for small p_F
def new_brown_alpha(g: NewBrownian, pF: float) -> float:
    def fprime(alpha): 
        return -sum(list(map(
            lambda sigma_i : 
                norm.pdf(alpha/sigma_i) / sigma_i,
            g.b
        )))

    def f(alpha):
        return (new_brown_pF(g, alpha) - pF)
    
    return optimize.newton(f, 0, fprime)
