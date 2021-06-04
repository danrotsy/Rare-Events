from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from loss import *
from scipy import stats
from scipy.stats import norm
from numpy import linalg as LA

def mc(
    g : Loss, 
    gCalls : int, 
    alpha : float
) -> float:
    """Monte Carlo estimate of pF for g w.r.t. alpha in gCalls."""
    fails = 0
    for i in range(0, gCalls):
        x = np.array(np.random.normal(0.0, 1.0, g.n))
        if (g.compute(x.T) > alpha): 
            fails += 1
    return float(fails) / float(gCalls)

def svm(
    g : Loss, 
    gCalls : int, 
    alpha : float
) -> float:
    """SVM estimate of pF for g w.r.t. alpha in gCalls."""
    x = []
    y = []
    for i in range(0, gCalls):
        row = np.random.uniform(-5.0, 5.0, g.n)
        x.append(list(row))
        if (g.compute(row) > alpha):
            y.append(1)
        else:
            y.append(-1)
    y = np.array(y)
    x = np.array(x)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(x, y)
    
    fails = 0
    for i in range(0, gCalls):
        test = np.array(np.random.normal(0.0, 1.0, g.n))
        if clf.predict([test]) > 0:
            fails += 1
    
    return float(fails)/float(gCalls)

def rf(
    g : Loss, 
    gCalls : int, 
    alpha : float,
    depth=2
) -> float:
    """Random Forest estimate of pF for g w.r.t. alpha in gCalls."""
    x = []
    y = []
    for i in range(0, gCalls):
        row = np.random.uniform(-5.0, 5.0, g.n)
        x.append(list(row))
        if (g.compute(row) > alpha):
            y.append(1)
        else:
            y.append(-1)
    y = np.array(y)
    x = np.array(x)
    clf = RandomForestClassifier(max_depth = depth, random_state = 0)
    clf.fit(x,y)
    
    fails = 0
    for i in range(0, gCalls):
        test = np.array(np.random.normal(0.0, 1.0, g.n))
        if clf.predict([test]) > 0:
            fails += 1
    
    return float(fails)/float(gCalls)

def linreg(
    g : Loss, 
    gCalls : int, 
    alpha : float
) -> float:
    x = []
    y = []
    for i in range(0, gCalls):
        row = np.random.uniform(-5.0, 5.0, g.n)
        x.append(list(row))
        y.append(g.compute(row))
    y = np.array(y)
    x = np.array(x)
    clf = linear_model.LinearRegression()
    clf.fit(x, y)
    
    pF = 1.0 - norm.cdf((alpha - clf.intercept_) / LA.norm(clf.coef_))
    
    return pF

def polyreg(
    g : Loss, 
    gCalls : int,
    alpha : float,
    deg = 3,
    reg = 0.0001,
    samples = 1000
) -> float:
    x = []
    y = []
    for i in range(0, gCalls):
        row = np.random.uniform(-5.0, 5.0, g.n)
        x.append(list(row))
        y.append(g.compute(row))
    y = np.array(y)
    x = np.array(x)
    clf = KernelRidge(alpha=reg, kernel = 'poly', degree = deg)
    clf.fit(x, y)
    
    fails = 0
    for i in range(0, samples):
        test = np.array(np.random.normal(0.0, 1.0, g.n))
        if clf.predict([test]) > alpha:
            fails += 1
    
    return float(fails)/float(samples)
    
    return pF
