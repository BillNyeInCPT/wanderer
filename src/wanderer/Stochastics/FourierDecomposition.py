# method to run integration and return  matrix of coefficients, study the distribution of OUP coeffs under fourier across given paths and then use the paths to run prediction beyond the time horizon. 

import numpy as np
from wanderer.Stochastics.StochasticProcess import StochasticProcess



def fourierDecomposition(num_coefficients: int, process: "StochasticProcess") -> np.ndarray:
   
    #get time grid and paths
    time_grid = process.timeGrid
    paths = process.paths

    #initialize matrix to hold coefficients
    coefficients = np.zeros((paths.shape[0], 2*num_coefficients+1), dtype=np.complex128)

    #compute coefficients for each path
    for i in range(paths.shape[0]):
        #get current path
        path = paths[i, :]

        #compute coefficients 
        for n in range(-num_coefficients, num_coefficients+1):
            #compute the integrand for the stochastic integral
            integrand = path * np.exp(-1j * 2* np.pi * n * time_grid / process.timeHorizon)
            #compute the coefficient using the trapzium rule for numerical integration
            coefficients[i, n + num_coefficients] = np.trapezoid(integrand, time_grid)/ np.sqrt(process.timeHorizon)
    #return the matrix of coefficients
    return coefficients

def reconstructFromFourier(coefficients: np.ndarray, time_grid: np.ndarray, time_horizon: float)-> "StochasticProcess":
    #get number of coefficients
    num_coefficients = (coefficients.shape[1] - 1) // 2

    #initialize matrix to hold reconstructed paths
    reconstructed_paths = np.zeros((coefficients.shape[0], len(time_grid)), dtype=np.complex128)

    #compute reconstructed paths for each set of coefficients
    for i in range(coefficients.shape[0]):
        #get current set of coefficients
        coeffs = coefficients[i, :]

        #compute reconstructed path
        for n in range(-num_coefficients, num_coefficients+1):
            reconstructed_paths[i, :] += coeffs[n + num_coefficients] * np.exp(1j * 2 * np.pi * n * time_grid / time_horizon)/ np.sqrt(time_horizon)

    #return the reconstructed paths as a StochasticProcess object
    return StochasticProcess(paths=reconstructed_paths.real, timeHorizon=time_horizon, timeStep=time_grid[1]-time_grid[0], method="Fourier Reconstruction", num_paths=coefficients.shape[0])

def outOfTimeSimulation(distributions: np.ndarray, time_step: float, time_horizon: float, num_paths: int):
    #get number of coefficients
    num_coefficients = (distributions.shape[1] - 1) // 2

    #create time grid for simulation
    time_grid = np.arange(0, time_horizon + time_step, time_step)

    #initialize matrix to hold simulated paths
    simulated_paths = np.zeros((num_paths, len(time_grid)), dtype=np.complex128)

    #simulate paths
    for i in range(num_paths):
        #sample a vector of coefficients from the distribution
        #randomly pick a row number 
        row = np.random.randint(0, distributions.shape[0])
        
        coeffs = distributions[row, :].reshape(1, -1)

        #reconstruct path from sampled coefficients
        simulated_paths[i, :] = reconstructFromFourier(coeffs, time_grid, time_horizon).paths[0, :]
    #return the simulated paths as a StochasticProcess object
    return StochasticProcess(paths=simulated_paths.real, timeHorizon=time_horizon, timeStep=time_step, method="Fourier Simulation", num_paths=num_paths)

    