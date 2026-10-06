import numpy as np
from scipy.stats import rankdata, norm

def findRecolouringMatrix(samples: np.ndarray) -> np.ndarray:
    n, d = samples.shape
    #apply empirical cdf to each element of the samples to get uniform marginals
    u = np.column_stack(
        [rankdata(samples[:, j], method="max") / n for j in range(d)]
    )

    #apply rank based adjustement 0 to 1/n and 1 to n/n+1 to avoid 0 and 1 values
    u[u == 0.0] = 1.0 / (n + 1)
    u[u == 1.0] = n / (n + 1)

    #gaussianize the uniform marginals using the inverse CDF of the standard normal distribution
    z = norm.ppf(u)

    #compute the covariance matrix of the gaussianized samples
    sigma = np.cov(z, rowvar=False)
    sigma = sigma + 1e-8 * np.eye(sigma.shape[0])
    #find the recolouring matrix using Cholesky decomposition
    L = np.linalg.cholesky(sigma)
    return L

def simulate(samples: np.ndarray, recolouring_matrix: np.ndarray, M: int) -> np.ndarray:
    #get dimension of the samples
    n, d = samples.shape
    #simulate M samples from unit hypercube
    w = np.random.uniform(size=(M, d))
    w = np.clip(w, 1e-12, 1 - 1e-12)
    # gaussianize the simulated samples
    z = norm.ppf(w)
    #recolour the gaussianized samples using the recolouring matrix
    y = z @ recolouring_matrix.T
    #transform the recoloured samples back to uniform marginals using the CDF of the standard normal distribution
    u = norm.cdf(y)
    #apply inverse empirical distribution
    simulated_samples = np.zeros_like(u)
    for j in range(d):
        sorted_col = np.sort(samples[:, j])
        # F^-1(u) = ceil(n * u)-th order statistic (1-indexed)
        idx = np.ceil(n * u[:, j]).astype(int) - 1
        idx = np.clip(idx, 0, n - 1)
        simulated_samples[:, j] = sorted_col[idx]
        
    return simulated_samples
