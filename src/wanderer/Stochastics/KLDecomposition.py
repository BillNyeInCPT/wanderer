import numpy as np
from wanderer.Stochastics.StochasticProcess import StochasticProcess

def KLDecomposition(process: StochasticProcess, num_components: int) -> (np.ndarray, np.ndarray, np.ndarray):    

    #compute the covariance matrix of the process
    covariance_matrix = np.cov(process.paths, rowvar=False)

    #set up the trapezoidal weight matrix for numerical integration
    time_step = process.timeStep
    num_time_points = process.paths.shape[1]
    weights = np.ones(num_time_points) * time_step
    weights[0] /= 2
    weights[-1] /= 2
    weights_sqrt = np.sqrt(weights)

    #set up the weighted covariance matrix
    weighted_covariance_matrix  = (weights_sqrt[:, None] * covariance_matrix) * weights_sqrt[None, :]          

    #compute the eigenvalues and eigenvectors of the weighted covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_covariance_matrix)

    #find the untransformed eigenvectors 
    untransformed_eigenvectors = untransformed_eigenvectors = eigenvectors / weights_sqrt[:, None]

    #sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = untransformed_eigenvectors[:, sorted_indices]

    #select the top num_components eigenvalues and eigenvectors
    selected_eigenvalues = sorted_eigenvalues[:num_components]
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]

    #compute the KL coefficients for each path
    KL_coefficients  = (process.paths * weights[None, :]) @ selected_eigenvectors

    return KL_coefficients, selected_eigenvectors, selected_eigenvalues



def reconstructFromKL(KL_coefficients: np.ndarray, eigenvectors: np.ndarray, time_grid: np.ndarray, time_horizon: float) -> StochasticProcess:
    #reconstruct the paths from the KL coefficients and eigenvectors
    reconstructed_paths = KL_coefficients @ eigenvectors.T

    #return the reconstructed paths as a StochasticProcess object
    return StochasticProcess(paths=reconstructed_paths, timeHorizon=time_horizon, timeStep=time_grid[1]-time_grid[0], method="KL Reconstruction", num_paths=KL_coefficients.shape[0])

def outOfTimeSimulationKL(distributions: np.ndarray, eigenfunctions: np.ndarray, time_step: float, time_horizon: float, num_paths: int):
    #get number of components
    num_components = distributions.shape[1]

    #create time grid for simulation
    time_grid = np.arange(0, time_horizon + time_step, time_step)

    #initialize matrix to hold simulated paths
    simulated_paths = np.zeros((num_paths, len(time_grid))) 

    #simulate paths
    for i in range(num_paths):
        #sample a vector of coefficients from the distribution
        row = np.random.randint(0, distributions.shape[0])
        coeffs = distributions[row, :].reshape(1, -1)

        #reconstruct path from sampled coefficients
        simulated_paths[i, :] = reconstructFromKL(coeffs, eigenfunctions, time_grid, time_horizon).paths[0, :]

    #return the simulated paths as a StochasticProcess object
    return StochasticProcess(paths=simulated_paths, timeHorizon=time_horizon, timeStep=time_step, method="KL Simulation", num_paths=num_paths)