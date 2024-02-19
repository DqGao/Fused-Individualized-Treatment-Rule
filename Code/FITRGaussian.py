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


# %% Gaussian kernel


## Euclidean distance
def euc_dist(x0, XX):
    x0 = x0.reshape(1, -1)
    return np.sqrt(np.sum((XX - x0)**2, 1))


## calculate the kernel between a vector x0 and a matrix XX
def kern_vec(x0, XX, sigma):
    x0 = x0.reshape(1, -1)
    return np.exp(- np.sum((XX - x0)**2, 1) / sigma**2)


## Gaussian kernel function
def f(x0, theta, XX, sigma):
    return theta[0] + theta[1:] @ kern_vec(x0, XX, sigma)


# %% value function for training set using IPW


## estimating the value function using IPW
def valuek_func_ipw(thetak, dat, KK, k):
    n = dat.shape[0]
    pred = np.column_stack([np.ones(n), KK])
    return sum(
        dat[:, pR[k]] / dat[:, pP] *
        (dat[:, pA] * (pred @ thetak) > 0)
    ) / n


# %% metrics for test set


## value function on the test set by taking the average outcome
def value_func_test(dat_test):
    return np.sum(dat_test[:, qR], 0) / dat_test.shape[0]


# %% functions for SepL


## objective function for SepL
def obj_func_single(thetak, dat, KK, lamk, k):
    n = dat.shape[0]
    pred = np.column_stack([np.ones(n), KK])
    val = sum(
        dat[:, pR[k]] / dat[:, pP] *
        surr_loss(dat[:, pA] * (pred @ thetak), 1)
    ) / n
    reg = lamk * thetak[1:] @ KK @ thetak[1:]
    return val + reg


## derivative of the objective function
def dobj_func_single(thetak, dat, KK, lamk, k):
    n = dat.shape[0]
    pred = np.column_stack([np.ones(n), KK])
    val = pred.T @ (
        dat[:, pR[k]] / dat[:, pP] *
        dsurr_loss(dat[:, pA] * (pred @ thetak), 1) * 
        dat[:, pA]
    ) / n
    reg = np.hstack([0, lamk * 2 * KK @ thetak[1:]])
    return val + reg


# %% functions for FITR_Ramp


## objective function for FITR-Ramp
def obj_func_ramp_iter(theta, fx_aux, dat, KK, L, lamk, mu, eta, k):
    n = dat.shape[0]
    pred = np.column_stack([np.ones(n), KK])
    val = sum(
        dat[:, pR[k]] / dat[:, pP] *
        surr_loss(dat[:, pA] * (pred @ theta), 1)
    ) / n
    reg = lamk * theta[1:] @ KK @ theta[1:]
    ## fusion penalty
    idx_aux = np.delete(np.arange(K), k)
    lap = mu * sum([
        2 * L[k, idx_aux[l]] * sum(
            ramp_loss((pred @ theta) * fx_aux[:, l], eta)
        )
        for l in range(fx_aux.shape[1])
    ]) / n
    return val + reg + lap


# %% optimization functions


## estimate ITR using SepL
def solve_single(theta0k, dat, KK, lam, k):
    ## remove the conditional mean of outcome
    dat_mod = dat.copy()
    mod_lm = LinearRegression().fit(dat[:, pX], dat[:, pR[k]])
    resid = dat[:, pR[k]] - mod_lm.predict(dat[:, pX])
    ## flip the sign of R and A if R < 0
    dat_mod[:, pR[k]] = abs(resid)
    dat_mod[:, pA] = dat[:, pA] * np.sign(resid)
    ## define the objective function and its derivative
    obj = lambda q: obj_func_single(q, dat_mod, KK, lam, k)
    dobj = lambda q: dobj_func_single(q, dat_mod, KK, lam, k)
    mod_opt = minimize(
        obj, jac=dobj, x0 = theta0k, method='BFGS'
    )
    return mod_opt.x
       

## estimate ITR using FITR-IntL
def solve_intl(theta_init, fx_aux, dat, KK, L, lam, mu, k):
    ## calculate the pseudo outcome
    idx_aux = np.delete(np.arange(K), k)
    consis = dat[:, pA].reshape(-1, 1) * np.sign(fx_aux)
    tmp_R = dat[:, pR[k]] - 2 * mu * dat[:, pP] * (consis @ L[idx_aux, k])
    ## remove the conditional mean of outcome
    dat_mod = dat.copy()
    mod_lm = LinearRegression().fit(dat[:, pX], tmp_R)
    resid = tmp_R - mod_lm.predict(dat[:, pX])
    ## flip the sign of R and A if R < 0
    dat_mod[:, pR[k]] = abs(resid)
    dat_mod[:, pA] = dat[:, pA] * np.sign(resid)
    ## define the objective function and its derivative
    obj = lambda q: obj_func_single(q, dat_mod, KK, lam, k)
    dobj = lambda q: dobj_func_single(q, dat_mod, KK, lam, k)
    mod_opt = minimize(
        obj, jac=dobj, x0 = theta_init, method='BFGS'
    )
    return mod_opt.x

            
## estimate ITR using FITR-Ramp
def solve_ramp(theta_init, fx_aux, dat, KK, L, lam, mu, eta, k):
    ## remove the conditional mean of outcome
    dat_mod = dat.copy()
    mod_lm = LinearRegression().fit(dat[:, pX], dat[:, pR[k]])
    resid = dat[:, pR[k]] - mod_lm.predict(dat[:, pX])
    ## flip the sign of R and A if R < 0
    dat_mod[:, pR[k]] = abs(resid)
    dat_mod[:, pA] = dat[:, pA] * np.sign(resid)
    ## define the objective function
    obj = lambda q: obj_func_ramp_iter(
        q, fx_aux, dat_mod, KK, L, lam, mu, eta, k
    )
    mod_opt = minimize(
        obj, x0 = theta_init, method='Powell'
    )
    return mod_opt.x

            
# %% cross validation


## cross validation to find lambda
def cv_lam(theta0, dat, KK, ncv, lam_list, k, seed):
    rd.seed(seed)
    v_cv = np.zeros((ncv, len(lam_list)))
    fold_idx = rd.choice(ncv, n, replace=True)
    for m in range(ncv):
        idx_train = np.array([i for i in range(n) if fold_idx[i] != m])
        idx_valid = np.array([i for i in range(n) if fold_idx[i] == m])
        dat_train = dat[idx_train, :]
        dat_valid = dat[idx_valid, :]
        KK_train = KK[np.ix_(idx_train, idx_train)]
        KK_valid = KK[np.ix_(idx_valid, idx_train)]
        for l, lamk in enumerate(lam_list):
            theta_cv = solve_single(
                theta0[np.hstack([0, idx_train + 1])], 
                dat_train, KK_train, lamk, k
            )
            v_cv[m, l] = valuek_func_ipw(
                theta_cv, dat_valid, KK_valid, k
            )
    v_cv_mean = np.mean(v_cv, 0)
    best_idx = v_cv_mean.argmax()
    return lam_list[best_idx]
        

## cross validation to find mu in FITR-IntL with lambda fixed
def cv_mu_intl(theta_init, fx_aux, dat, KK, L, lam, ncv, mu_list, k, seed):
    rd.seed(seed)
    v_cv = np.zeros((ncv, len(mu_list)))
    fold_idx = rd.choice(ncv, n, replace=True)
    for m in range(ncv):
        idx_train = np.array([i for i in range(n) if fold_idx[i] != m])
        idx_valid = np.array([i for i in range(n) if fold_idx[i] == m])
        dat_train = dat[idx_train, :]
        dat_valid = dat[idx_valid, :]
        KK_train = KK[np.ix_(idx_train, idx_train)]
        KK_valid = KK[np.ix_(idx_valid, idx_train)]
        fx_aux_train = fx_aux[idx_train, :]
        for l, mu in enumerate(mu_list):
            theta_cv = solve_intl(
                theta_init[np.hstack([0, idx_train + 1])], 
                fx_aux_train, dat_train, KK_train, L, lam, mu, k
            )
            v_cv[m, l] = valuek_func_ipw(
                theta_cv, dat_valid, KK_valid, k
            )
    v_cv_mean = np.mean(v_cv, 0)
    best_idx = v_cv_mean.argmax()
    return mu_list[best_idx]
        

## cross validation to find mu in FITR-Ramp with lambda fixed
def cv_mu_ramp(theta_init, fx_aux, dat, KK, L, lam, ncv, mu_list, eta_list, k, seed):
    rd.seed(seed)
    v_cv = np.zeros((ncv, len(mu_list), len(eta_list)))
    fold_idx = rd.choice(ncv, n, replace=True)
    for m in range(ncv):
        idx_train = np.array([i for i in range(n) if fold_idx[i] != m])
        idx_valid = np.array([i for i in range(n) if fold_idx[i] == m])
        dat_train = dat[idx_train, :]
        dat_valid = dat[idx_valid, :]
        KK_train = KK[np.ix_(idx_train, idx_train)]
        KK_valid = KK[np.ix_(idx_valid, idx_train)]
        fx_aux_train = fx_aux[idx_train, :]
        for l, mu in enumerate(mu_list):
            for j, eta in enumerate(eta_list):
                theta_cv = solve_ramp(
                    theta_init[np.hstack([0, idx_train + 1])], 
                    fx_aux_train, dat_train, KK_train, L, lam, mu, eta, k
                )
                v_cv[m, l, j] = valuek_func_ipw(
                    theta_cv, dat_valid, KK_valid, k
                )
    v_cv_mean = np.mean(v_cv, 0)
    best_idx = np.unravel_index(v_cv_mean.argmax(), v_cv_mean.shape)
    return [mu_list[best_idx[0]], eta_list[best_idx[1]]]
        

# %% data generation


## generate new data using the behavior policy
def new_data_b(n, d, seed):
    rd.seed(seed)
    X = rd.uniform(-1, 1, (n, d))
    X[:, 2] = 0.8 * X[:, 2] + 0.2 * X[:, 0]
    P = np.ones(n) * 0.5
    ## treatments are taken w.p. 0.5
    A0 = 2 * rd.binomial(1, P) - 1
    A = np.tile(A0.reshape(-1, 1), (1, K))
    R = outcome(X, A, n, seed)
    return np.column_stack([X, P, A0, R])


## generate X in new data
def new_data_X(n, d, seed):
    rd.seed(seed)
    X = rd.uniform(-1, 1, (n, d))
    X[:, 2] = 0.8 * X[:, 2] + 0.2 * X[:, 0]
    return X


## generate A in new data using the estimated ITR
def new_data_A(n, KK, theta):
    pred = np.column_stack([np.ones(n), KK])
    ## treatments are taken based on the given ITR
    return (2 * (pred @ theta.reshape(-1, 1) > 0) - 1).reshape(-1)


## generate new data using the optimal ITR
def new_data_opt(n, d, seed):
    rd.seed(seed)
    X = rd.uniform(-1, 1, (n, d))
    X[:, 2] = 0.8 * X[:, 2] + 0.2 * X[:, 0]
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
d = 10  ## dimension of X, including the intercept
K = 2  ## number of outcomes
pX = np.arange(d)  ## index of X in the training data
pP = d  ## index of P in the training data
pA = d + 1  ## index of A in the training data
pR = np.arange(d + 2, d + 2 + K)  ## index of R in the training data
qX = np.arange(d)  ## index of X in the test data
qA = np.arange(d, d + K)  ## index of A in the test data
qR = np.arange(d + K, d + 2 * K)  ## index of R in the test data
theta0 = np.zeros(n + 1)  ## initial point in optimization
theta0_aux = np.zeros(N + 1)  ## initial point in optimization
lam_list = [0.2, 0.1, 0.01, 0.001, 0.0001]
mu_list = [2, 1, 0.5, 0.2, 0.1]
eta_list = [1, 0.5, 0.2, 0.1, 0.05]
cov = [0.5, 0.2]  ## parameter in covariance matrix of noise term E
sig = [1, 1.5]  ## parameter in the interaction effect
ncv = 4  ## number of folds in cross validation
nitr = 400  ## number of replications
seed = 2022  ## random seed
n_jobs = 40  ## parallel computing


main_effect = [
    lambda X: 1 + 2 * X[:, 0] + X[:, 1]**2 + 1 * X[:, 0] * X[:, 1],
    lambda X: 1 + 2 * X[:, 0]**2 + 1.5 * X[:, 1] + 0.5 * X[:, 0] * X[:, 1],
]
inter_effect = [
    lambda X, A: sig[0] * (- 2.2 + 1 * np.exp(X[:, 0]) + 1 * np.exp(X[:, 1])) * A,
    lambda X, A: sig[1] * (- 2.3 + 1 * np.exp(X[:, 0]) + 1 * np.exp(X[:, 1])) * A,
]


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


filename = 'Gaussian_n' + str(n) + '_N' + str(N) + '.txt'

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
    ## generate X in the test data
    X_test = new_data_X(N_test, d, seeds[0])
    ## optimal treatment in the test data
    A_test_opt = opt_trt(X_test)
    
    ## parameter sigma in the Gaussian kernel using the median heuristic
    sigma = np.median(np.array(list(map(
        lambda x: euc_dist(x, dat[:, pX]), dat[:, pX]
    ))))
    ## kernel matrix within the main training data
    KK_pri = np.array(list(map(
        lambda x: kern_vec(x, dat[:, pX], sigma), dat[:, pX]
    )))
    ## kernel matrix between main data and test data
    KK_pri_new = np.array(list(map(
        lambda x: kern_vec(x, dat[:, pX], sigma), X_test
    )))

    ## SepL
    lam = np.zeros(K)
    ## primary outcome
    lam[0] = cv_lam(theta0, dat, KK_pri, ncv, lam_list, 0, seeds[-3]) 
    theta_init = solve_single(theta0, dat, KK_pri, lam[0], 0)
    A_test_init = np.zeros((N_test, K))

    ## auxiliary outcome
    ## when nr < infinity
    if nr <= 10:
        theta_aux = []
        ## auxiliary training data that contain secondary outcomes
        dat_aux = new_data_b(N, d, seeds[-4])
        ## kernel matrix within auxiliary training data
        KK_aux = np.array(list(map(
            lambda x: kern_vec(x, dat_aux[:, pX], sigma), dat_aux[:, pX]
        )))
        ## kernel matrix between auxiliary data and main data
        KK_aux_pri = np.array(list(map(
            lambda x: kern_vec(x, dat_aux[:, pX], sigma), dat[:, pX]
        )))
        ## kernel matrix between auxiliary data and test data
        KK_aux_new = np.array(list(map(
            lambda x: kern_vec(x, dat_aux[:, pX], sigma), X_test
        )))

        for k in range(1, K):
            ## tune lambda for secondary outcomes
            lam[k] = cv_lam(theta0_aux, dat_aux, KK_aux, ncv, lam_list, k, seeds[-5])
            ## estimate secondary outcome ITRs using SepL
            theta_aux.append(solve_single(theta0_aux, dat_aux, KK_aux, lam[k], k))
        for l in range(1, K):
            ## treatments suggested for secondary outcomes in the test set
            A_test_init[:, l] = new_data_A(N_test, KK_aux_new, theta_aux[l - 1])

        ## treatments suggested for secondary outcomes in the main dataset
        pred = np.column_stack([np.ones(n), KK_aux_pri])
        fx_aux = np.column_stack([
            pred @ theta
            for theta in theta_aux
        ])
    ## when nr = infinity
    else:
        ## treatments suggested for secondary outcomes in the main dataset
        fx_aux = opt_trt(dat[:, pX])[:, 1:]
        ## optimal auxiliary outcome ITR is known
        A_test_init[:, 1:] = A_test_opt[:, 1:]

    ## metrics
    ## treatments suggested for primary outcome in the test set
    A_test_init[:, 0] = new_data_A(N_test, KK_pri_new, theta_init)
    ## observed outcome in the test set
    R_test_init = outcome(X_test, A_test_init, N_test, seeds[0])
    value_hat0 = np.mean(R_test_init, axis=0)  ## value function
    acc_hat0 = np.mean(A_test_init == A_test_opt, axis=0)  ## accuracy
    agr_hat0 = np.mean(A_test_init[:, [0]] == A_test_opt[:, [1]], axis=0)  ## agreement rate


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
    mu_intl = np.zeros(K)
    start = time.time()
    ## tune mu
    mu_intl[0] = cv_mu_intl(
        theta_init, fx_aux, dat, KK_pri, L, lam[0], ncv, mu_list, 0, seeds[1]
    )
    ## estimate FITR
    theta_hat_intl = solve_intl(
        theta_init, fx_aux, dat, KK_pri, L, lam[0], mu_intl[0], 0
    )
    elapse_intl = time.time() - start
    ## metrics
    A_test_fitr_intl = A_test_init.copy()
    A_test_fitr_intl[:, 0] = new_data_A(N_test, KK_pri_new, theta_hat_intl)
    R_test_fitr_intl = outcome(X_test, A_test_fitr_intl, N_test, seeds[2])
    value_hat_intl = np.mean(R_test_fitr_intl, axis=0)
    acc_hat_intl = np.mean(A_test_fitr_intl == A_test_opt, axis=0)
    agr_hat_intl = np.mean(A_test_fitr_intl[:, [0]] == A_test_opt[:, [1]], axis=0)
    
    ## FITR-Ramp
    mu_ramp = np.zeros(K)
    eta = np.zeros(K)
    start = time.time()
    ## tune mu and eta
    mu_ramp[0], eta[0] = cv_mu_ramp(
        theta_init, fx_aux, dat, KK_pri, L, lam[0], ncv, mu_list, eta_list, 0, seeds[3]
    )
    ## estimate FITR
    theta_hat_ramp = solve_ramp(
        theta_init, fx_aux, dat, KK_pri, L, lam[0], mu_ramp[0], eta[0], 0
    )
    elapse_ramp = time.time() - start
    ## metrics
    A_test_fitr_ramp = A_test_init.copy()
    A_test_fitr_ramp[:, 0] = new_data_A(N_test, KK_pri_new, theta_hat_ramp)
    R_test_fitr_ramp = outcome(X_test, A_test_fitr_ramp, N_test, seeds[4])
    value_hat_ramp = np.mean(R_test_fitr_ramp, axis=0)
    acc_hat_ramp = np.mean(A_test_fitr_ramp == A_test_opt, axis=0)
    agr_hat_ramp = np.mean(A_test_fitr_ramp[:, [0]] == A_test_opt[:, [1]], axis=0)

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

