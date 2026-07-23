#Evaluate the stochastic integral of a function with respect to a Brownian motion path using right hand Riemann sums (this is a crude implementation of the Ito integral)

from wanderer.Stochastics.BrownianMotion import BrownianMotionGenerator
import numpy as np


def stochastic_integral(function, start, end, time_step, seed=123):
    np.random.seed(seed)
    #Generate a Brownian motion path on the interval [start, end] with the given time step
    bm_generator = BrownianMotionGenerator(mu=0, sigma=1, time_horizon=end, time_step=time_step, seed=seed)
    bm_path = bm_generator.generate_BrownianMotion(num_paths=1)
    bm_path_sliced = bm_path.time_slice(start=start, end=end)
    time_grid = bm_path_sliced.timeGrid
    #Evaluate the function at the time grid points
    function_values = function(time_grid)

    #Compute the Ito integral using right hand Riemann sums
    integral = 0.0
    for i in range(len(time_grid) - 1):
        dW = bm_path.paths[0, i + 1] - bm_path.paths[0, i]
        integral += function_values[i] * dW

    return integral