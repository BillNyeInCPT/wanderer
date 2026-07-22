# class to implement Euler-Marayuma method for approximating solutions to stochastic differential equations

import StochasticProcess


class EulerMarayumaGenerator:
    def __init__(self, drift_function, diffusion_function, initial_value, time_horizon, time_step):
        self.drift_function = drift_function
        self.diffusion_function = diffusion_function
        self.initial_value = initial_value
        self.time_horizon = time_horizon
        self.time_step = time_step

    def generate_EulerMarayuma(self, num_paths):
        num_steps = int(self.time_horizon / self.time_step)
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.initial_value

        for i in range(num_steps):
            dt = self.time_step
            dW = np.random.normal(0, np.sqrt(dt), size=num_paths)
            paths[:, i + 1] = paths[:, i] + self.drift_function(paths[:, i]) * dt + self.diffusion_function(paths[:, i]) * dW

        return StochasticProcess(paths=paths, timeHorizon=self.time_horizon, timeStep=self.time_step, method="Euler-Marayuma")

