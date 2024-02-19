# %% set up


import numpy as np
from numpy import random as rd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import time
from joblib import Parallel, delayed


# %% surrogate loss for negative value


## logistic loss for I(Af < 0)
def surr_loss(x, gamma):
    return np.log(1 + np.exp(- x / gamma))


## derivative of the surrogate loss function
def dsurr_loss(x, gamma):
    return (- 1 / gamma) * np.exp(- x / gamma) / (1 + np.exp(- x / gamma))


# %% ramp loss for positive value


## ramp loss for I(Af > 0), convex function 1
def ramp_loss1(x, eta):
    return np.maximum(x / eta + 1, 0)


## ramp loss for I(Af > 0), convex function 2
def ramp_loss2(x, eta):
    return np.maximum(x / eta, 0)


## ramp loss for I(Af > 0)
def ramp_loss(x, eta):
    return ramp_loss1(x, eta) - ramp_loss2(x, eta)


# %% linear kernel


## linear kernel function
def f(X, theta):
    return X @ theta


# %% value function for training set using IPW


## estimating the value function using normalized IPW
def valuek_func_ipw(thetak, dat, k):
    numerator = sum(
        dat[:, pR[k]] / dat[:, pP] *
        (dat[:, pA] * f(dat[:, pX], thetak) > 0)
    )
    denominator = sum(
        1 / dat[:, pP] *
        (dat[:, pA] * f(dat[:, pX], thetak) > 0)
    )
    return numerator / denominator


# %% value function, accuracy and agreement rate for test set


## value function on the test set by taking the average outcome
def value_func_test(dat_test):
    return np.sum(dat_test[:, qR], 0) / dat_test.shape[0]


## accuracy on the test set
def acc_func_test(dat_test):
    return np.mean(dat_test[:, qA] == opt_trt(dat_test[:, qX]), 0)


## agreement rate on the test set
def agree_func_test(dat_test, theta1, theta2, J_theta):
    A1 = np.column_stack([
        2 * (dat_test[:, qX] @ theta1[:, j] > 0) - 1
        for j in range(J_theta)
    ])
    A2 = np.column_stack([
        2 * (dat_test[:, qX] @ theta2[:, j] > 0) - 1
        for j in range(J_theta)
    ])
    return np.mean(A1 == A2, 0)


# %% functions for SepL


## objective function for SepL
def obj_func_single(thetak, dat, lamk, k):
    val = sum(
        dat[:, pR[k]] / dat[:, pP] *
        surr_loss(dat[:, pA] * f(dat[:, pX], thetak), 1)
    ) / dat.shape[0]
    reg = lamk * thetak @ thetak
    return val + reg


## derivative of the objective function
def dobj_func_single(thetak, dat, lamk, k):
    val = dat[:, pX].T @ (
        dat[:, pR[k]] / dat[:, pP] *
        dsurr_loss(dat[:, pA] * f(dat[:, pX], thetak), 1) * 
        dat[:, pA]
    ) / dat.shape[0]
    reg = lamk * 2 * thetak
    return val + reg


# %% functions for FITR-Ramp


## objective function for FITR-Ramp
def obj_func_ramp_iter(theta, dat, L, lamk, mu, eta, k):
    val = sum(
        dat[:, pR[k]] / dat[:, pP] *
        surr_loss(dat[:, pA] * f(dat[:, pX], theta[:, k]), 1)
    ) / dat.shape[0]
    reg = lamk * theta[:, k] @ theta[:, k]
    ## fusion penalty
    lap = mu * sum([
        2 * L[k, l] * sum(
            ramp_loss(f(dat[:, pX], theta[:, k]) *
                      f(dat[:, pX], theta[:, l]), eta)
        )
        for l in np.delete(np.arange(K), k)
    ]) / dat.shape[0]
    return val + reg + lap


# %% optimization functions


## estimate ITR using SepL
def solve_single(theta0, dat, lam, k):
    ## remove the conditional mean of outcome
    dat_mod = dat.copy()
    mod_lm = LinearRegression().fit(dat[:, pX], dat[:, pR[k]])
    resid = dat[:, pR[k]] - mod_lm.predict(dat[:, pX])
    ## flip the sign of R and A if R < 0
    dat_mod[:, pR[k]] = abs(resid)
    dat_mod[:, pA] = dat[:, pA] * np.sign(resid)
    ## define the objective function and its derivative
    obj = lambda q: obj_func_single(q, dat_mod, lam, k)
    dobj = lambda q: dobj_func_single(q, dat_mod, lam, k)
    mod_opt = minimize(
        obj, jac=dobj, x0 = theta0[:, k], method='BFGS'
    )
    return mod_opt.x


## estimate ITR using FITR-IntL
def solve_intl(theta_init, dat, L, lam, mu, k):
    ## calculate the pseudo outcome
    idx_l = np.delete(np.arange(K), k)
    fx = dat[:, pX] @ theta_init[:, idx_l]
    consis = dat[:, pA].reshape(-1, 1) * np.sign(fx)
    tmp_R = dat[:, pR[k]] - mu * dat[:, pP] * (consis @ L[idx_l, k])
    ## remove the conditional mean of outcome
    dat_mod = dat.copy()
    mod_lm = LinearRegression().fit(dat[:, pX], tmp_R)
    resid = tmp_R - mod_lm.predict(dat[:, pX])
    ## flip the sign of R and A if R < 0
    dat_mod[:, pR[k]] = abs(resid)
    dat_mod[:, pA] = dat[:, pA] * np.sign(resid)
    ## define the objective function and its derivative
    obj = lambda q: obj_func_single(q, dat_mod, lam, k)
    dobj = lambda q: dobj_func_single(q, dat_mod, lam, k)
    mod_opt = minimize(
        obj, jac=dobj, x0 = theta_init[:, k], method='BFGS'
    )
    return mod_opt.x

            
## estimate ITR using FITR-Ramp
def solve_ramp(theta_init, dat, L, lam, mu, eta, k):
    theta_curr = theta_init.copy()
    ## remove the conditional mean of outcome
    dat_mod = dat.copy()
    mod_lm = LinearRegression().fit(dat[:, pX], dat[:, pR[k]])
    resid = dat[:, pR[k]] - mod_lm.predict(dat[:, pX])
    ## flip the sign of R and A if R < 0
    dat_mod[:, pR[k]] = abs(resid)
    dat_mod[:, pA] = dat[:, pA] * np.sign(resid)
    ## define the objective function
    def obj(q):
        theta = theta_curr.copy()
        theta[:, k] = q
        return obj_func_ramp_iter(theta, dat_mod, L, lam, mu, eta, k)
    mod_opt = minimize(
        obj, x0 = theta_init[:, k], method='Powell'
    )
    return mod_opt.x


# %% cross validation


## cross validation to find lambda
def cv_lam(theta0, dat, ncv, lam_list, k, seed):
    rd.seed(seed)
    v_cv = np.zeros((ncv, len(lam_list)))
    fold_idx = rd.choice(ncv, n, replace=True)
    for m in range(ncv):
        idx_train = np.array([i for i in range(n) if fold_idx[i] != m])
        idx_valid = np.array([i for i in range(n) if fold_idx[i] == m])
        dat_train = dat[idx_train, :]
        dat_valid = dat[idx_valid, :]
        for l, lamk in enumerate(lam_list):
            theta_cv = solve_single(theta0, dat_train, lamk, k)
            v_cv[m, l] = valuek_func_ipw(theta_cv, dat_valid, k)
    v_cv_mean = np.mean(v_cv, 0)
    best_idx = v_cv_mean.argmax()
    return lam_list[best_idx]
        

## cross validation to find mu in FITR-IntL with lambda fixed
def cv_mu_intl(theta_init, dat, L, lam, ncv, mu_list, k, seed):
    rd.seed(seed)
    v_cv = np.zeros((ncv, len(mu_list)))
    fold_idx = rd.choice(ncv, n, replace=True)
    for m in range(ncv):
        idx_train = np.array([i for i in range(n) if fold_idx[i] != m])
        idx_valid = np.array([i for i in range(n) if fold_idx[i] == m])
        dat_train = dat[idx_train, :]
        dat_valid = dat[idx_valid, :]
        for l, mu in enumerate(mu_list):
            theta_cv = solve_intl(
                theta_init, dat_train, L, lam, mu, k
            )
            v_cv[m, l] = valuek_func_ipw(theta_cv, dat_valid, k)
    v_cv_mean = np.mean(v_cv, 0)
    best_idx = v_cv_mean.argmax()
    return mu_list[best_idx]
        

## cross validation to find mu in FITR-Ramp with lambda fixed
def cv_mu_ramp(theta_init, dat, L, lam, ncv, mu_list, eta_list, k, seed):
    rd.seed(seed)
    v_cv = np.zeros((ncv, len(mu_list), len(eta_list)))
    fold_idx = rd.choice(ncv, n, replace=True)
    for m in range(ncv):
        idx_train = np.array([i for i in range(n) if fold_idx[i] != m])
        idx_valid = np.array([i for i in range(n) if fold_idx[i] == m])
        dat_train = dat[idx_train, :]
        dat_valid = dat[idx_valid, :]
        for l, mu in enumerate(mu_list):
            for j, eta in enumerate(eta_list):
                theta_cv = solve_ramp(
                    theta_init, dat_train, L, lam, mu, eta, k
                )
                v_cv[m, l, j] = valuek_func_ipw(theta_cv, dat_valid, k)
    v_cv_mean = np.mean(v_cv, 0)
    best_idx = np.unravel_index(v_cv_mean.argmax(), v_cv_mean.shape)
    return [mu_list[best_idx[0]], eta_list[best_idx[1]]]
        

# %% data generation


## generate new data using the behavior policy
def new_data_b(n, d, seed):
    rd.seed(seed)
    X = rd.uniform(-1, 1, (n, d - 1))
    X[:, 2] = 0.8 * X[:, 2] + 0.2 * X[:, 0]
    X = np.column_stack([np.ones(n), X])
    P = np.ones(n) * 0.5
    ## treatments are taken w.p. 0.5
    A0 = 2 * rd.binomial(1, P) - 1
    A = np.tile(A0.reshape(-1, 1), (1, K))
    R = outcome(X, A, n, seed)
    return np.column_stack([X, P, A0, R])


## generate new data using the estimated ITR
def new_data_e(n, d, theta, seed):
    rd.seed(seed)
    X = rd.uniform(-1, 1, (n, d - 1))
    X[:, 2] = 0.8 * X[:, 2] + 0.2 * X[:, 0]
    X = np.column_stack([np.ones(n), X])
    ## treatments are taken based on the given ITR
    A = np.column_stack([
        2 * (X @ theta[:, k] > 0) - 1
        for k in range(K)
    ])
    R = outcome(X, A, n, seed)
    return np.column_stack([X, A, R])


## generate new data using the optimal ITR
def new_data_opt(n, d, seed):
    rd.seed(seed)
    X = rd.uniform(-1, 1, (n, d - 1))
    X[:, 2] = 0.8 * X[:, 2] + 0.2 * X[:, 0]
    X = np.column_stack([np.ones(n), X])
    ## optimal treatments
    A = opt_trt(X)
    R = outcome(X, A, n, seed)
    return np.column_stack([X, A, R])


# %% scenario settings


nlist = [100, 200]  ## sample size
nrlist = [1, 2, 4, 8, 16]  ## ratio between N and n
n = nlist[0]
nr = nrlist[0]

N = nr * n  ## sample size for learning secondary outcomes
N_test = 100000  ## test data sample size
d = 11  ## dimension of X, including the intercept
K = 2  ## number of outcomes
pX = np.arange(d)  ## index of X in the training data
pP = d  ## index of P in the training data
pA = d + 1  ## index of A in the training data
pR = np.arange(d + 2, d + 2 + K)  ## index of R in the training data
qX = np.arange(d)  ## index of X in the test data
qA = np.arange(d, d + K)  ## index of A in the test data
qR = np.arange(d + K, d + 2 * K)  ## index of R in the test data
theta0 = np.zeros((d, K))  ## initial point in optimization
lam_list = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
mu_list = [0.1, 0.2, 0.5, 1, 2]
eta_list = [0.05, 0.1, 0.2, 0.5, 1]
cov = [0.5, 0.2]  ## parameter in covariance matrix of noise term E
sig = [0.5, 0.8]  ## parameter in the interaction effect
ncv = 4  ## number of folds in cross validation
nitr = 400  ## number of replications
seed = 2022  ## random seed
n_jobs = 20  ## parallel computing


main_effect = [
    lambda X: 1 + 2 * X[:, 1] + X[:, 2]**2 + 1 * X[:, 1] * X[:, 2],
    lambda X: 1 + 2 * X[:, 1]**2 + 1.5 * X[:, 2] + 0.5 * X[:, 1] * X[:, 2],
]
inter_effect = [
    lambda X, A: sig[0] * (0.2 - 1 * X[:, 1] - 2 * X[:, 2]) * A,
    lambda X, A: sig[1] * (0.2 - 1 * X[:, 1] - 1.8 * X[:, 2]) * A,
]
true_theta = np.array([
    [0.2, -1, -2] + [0] * (d - 3),
    [0.2, -1, -1.8] + [0] * (d - 3),
]).T


## define the mean and covariance matrix of noise term E
Mu = np.zeros(K)
Sigma = cov[1] * np.ones((K, K))
np.fill_diagonal(Sigma, cov[0])

## outcome = main effect + interaction effect + noise term
def outcome(X, A, n, seed):
    E = rd.default_rng(seed).multivariate_normal(Mu, Sigma, n)
    R = []
    R = np.column_stack([
        main_effect[k](X) + inter_effect[k](X, A[:, k]) + E[:, k]
        for k in range(K)
    ])
    return R


## find the optimal treatment
def opt_trt(X):
    return np.column_stack([
        2 * (inter_effect[k](X, 1) > 0) - 1
        for k in range(K)
    ])


# %% simulation starts


filename = 'Linear_n' + str(n) + '_N' + str(N) + '.txt'
    
with open(filename, 'w') as file:
    file.write("")

rd.seed(seed)
## seed of each replication
seed_itr = rd.randint(low=0, high=5000, size=nitr)


## define the function for parallel computing
def policy_optimizer(itr):
    rd.seed(seed_itr[itr])
    seeds = rd.randint(low=0, high=5000, size=10)
    ## optimal value function
    dat_test = new_data_opt(N_test, d, seeds[-1])
    value_opt = value_func_test(dat_test)
    # trt_opt = opt_trt(dat_test[:, qX])
    # print(np.mean(trt_opt[:, 0] == trt_opt[:, 1]))
    # print(value_opt)
    # print(np.mean(trt_opt == 1, 0))

    ## generate main training data that contain primary outcome using the behavior policy
    dat = new_data_b(n, d, seeds[-2])

    ## SepL
    theta_init = theta0.copy()
    lam = np.zeros(K)
    ## primary outcome
    ## tune lambda for primary outcome
    lam[0] = cv_lam(theta0, dat, ncv, lam_list, 0, seeds[-3])
    ## estimate primary outcome ITRs using SepL
    theta_init[:, 0] = solve_single(theta0, dat, lam[0], 0)

    ## auxiliary outcome
    ## when nr < infinity
    if nr <= 10:
        ## auxiliary training data that contain secondary outcomes
        dat_aux = new_data_b(N, d, seeds[-4])
        for k in range(1, K):
            ## tune lambda for secondary outcomes
            lam[k] = cv_lam(theta0, dat_aux, ncv, lam_list, k, seeds[-5])
            ## estimate secondary outcome ITRs using SepL
            theta_init[:, k] = solve_single(theta0, dat_aux, lam[k], k)
    ## when nr = infinity
    else:
        ## optimal auxiliary outcome ITR is known
        for k in range(1, K):
            theta_init[:, k] = true_theta[:, k]

    ## metrics
    ## generate test data using ITRs from SepL
    dat_test = new_data_e(N_test, d, theta_init, seeds[0])
    value_hat0 = value_func_test(dat_test)  ## value function
    acc_hat0 = acc_func_test(dat_test)  ## accuracy
    agr_hat0 = agree_func_test(  ## agreement rate
        dat_test, theta_init[:, [0]], true_theta[:, [1]], 1
    )

    ## find the matrix Omega
    ## remove the conditional mean of outcomes
    phi = np.column_stack([dat[:, pX]])
    inter_eff_hat = np.zeros((n, K))
    for k in range(K):
        mod_lm = LinearRegression().fit(phi, dat[:, pR[k]])
        inter_eff_hat[:, k] = dat[:, pR[k]] - mod_lm.predict(phi)
    ## adjacency matrix
    Adj = np.corrcoef(inter_eff_hat.T)  ## Pearson's correlation
    # Adj, _ = spearmanr(inter_eff_hat)  ## Spearman's rank correlation
    D = np.diag(np.sum(abs(Adj), axis=1))
    ## Laplacian matrix
    L = D - Adj
    
    ## FITR-IntL
    theta_hat_intl = theta_init.copy()
    mu_intl = np.zeros(K)
    start = time.time()
    ## tune mu
    mu_intl[0] = cv_mu_intl(
        theta_init, dat, L, lam[0], ncv, mu_list, 0, seeds[1]
    )
    ## estimate FITR
    theta_hat_intl[:, 0] = solve_intl(
        theta_init, dat, L, lam[0], mu_intl[0], 0
    )
    elapse_intl = time.time() - start
    ## metrics
    ## generate test data using ITRs from FITR-IntL
    dat_test = new_data_e(N_test, d, theta_hat_intl, seeds[2])
    value_hat_intl = value_func_test(dat_test)
    acc_hat_intl = acc_func_test(dat_test)
    agr_hat_intl = agree_func_test(
        dat_test, theta_hat_intl[:, [0]], true_theta[:, [1]], 1
    )
    
    ## FITR-Ramp
    theta_hat_ramp = theta_init.copy()
    mu_ramp = np.zeros(K)
    eta = np.zeros(K)
    start = time.time()
    ## tune mu and eta
    mu_ramp[0], eta[0] = cv_mu_ramp(
        theta_init, dat, L, lam[0], ncv, mu_list, eta_list, 0, seeds[3]
    )
    ## estimate FITR
    theta_hat_ramp[:, 0] = solve_ramp(
        theta_init, dat, L, lam[0], mu_ramp[0], eta[0], 0
    )
    elapse_ramp = time.time() - start
    ## metrics
    ## generate test data using ITRs from FITR-Ramp
    dat_test = new_data_e(N_test, d, theta_hat_ramp, seeds[4])
    value_hat_ramp = value_func_test(dat_test)
    acc_hat_ramp = acc_func_test(dat_test)
    agr_hat_ramp = agree_func_test(
        dat_test, theta_hat_ramp[:, [0]], true_theta[:, [1]], 1
    )
    
    out = np.hstack([
        np.hstack([value_hat0, value_hat_intl, value_hat_ramp]),
        np.hstack([
            value_opt - value_hat0, 
            value_opt - value_hat_intl, value_opt - value_hat_ramp,
        ]),
        (np.maximum(0, np.hstack([
            value_opt - value_hat0, 
            value_opt - value_hat_intl, value_opt - value_hat_ramp,
        ])))**2,
        np.hstack([acc_hat0, acc_hat_intl, acc_hat_ramp]),
        np.hstack([agr_hat0, agr_hat_intl, agr_hat_ramp]),
        lam, mu_intl, mu_ramp, eta, 
        np.hstack([elapse_intl, elapse_ramp]),
    ])
    
    with open(filename, 'a') as file:
        np.savetxt(file, out.reshape((1, -1)), fmt='%.4f')


start = time.time()
results = Parallel(n_jobs=n_jobs)(
    delayed(policy_optimizer)(itr) for itr in range(nitr)
)
print(time.time() - start)    


res = np.loadtxt(filename)
with open(filename, 'a') as file:
    np.savetxt(
        file, 
        np.row_stack([np.mean(res, axis=0), np.std(res, axis=0)]), 
        fmt='%.4f'
    )

