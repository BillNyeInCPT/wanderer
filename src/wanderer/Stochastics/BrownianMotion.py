# class to implement the generation of Brownian motion paths

from .StochasticProcess import StochasticProcess
import numpy as np

class BrownianMotionGenerator:
    def __init__(self, mu , sigma,  time_horizon, time_step, seed=123, num_paths=1):
        self.mu = mu
        self.sigma = sigma
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.seed = seed
        self.num_paths = num_paths
        

    def generate_BrownianMotion(self):
        np.random.seed(self.seed)  # Set the seed for reproducibility
        num_steps = int(self.time_horizon / self.time_step)
        paths = np.zeros((self.num_paths, num_steps + 1))
        paths[:, 0] = 0  # Initial value of Brownian motion is 0

        for i in range(num_steps):
            dt = self.time_step
            dW = np.random.normal(0, np.sqrt(dt), size=self.num_paths)
            paths[:, i + 1] = paths[:, i] + self.mu * dt + self.sigma * dW

        return StochasticProcess(paths=paths, timeHorizon=self.time_horizon, timeStep=self.time_step, method=f"BrownianMotion_{self.seed}", num_paths=self.num_paths)
