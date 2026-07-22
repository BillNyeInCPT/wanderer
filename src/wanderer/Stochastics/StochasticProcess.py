#Discretized stochastic process class

from attr import dataclass

@dataclass
class StochasticProcess:
    paths: np.ndarray = attr.ib(eq=False)
    timeHorizon: float
    timeStep: float
    method: str

    @property
    def timeMean(self) -> np.ndarray:
        return np.mean(self.paths, axis=0)

    @property
    def timeVariance(self) -> np.ndarray:
        return np.var(self.paths, axis=0)

    def __len__(self):
        return int(self.timeHorizon / self.timeStep)

    @classmethod
    def from_csv(cls, csv, timeStep = 1,) -> "StochasticProcess":
        data = np.loadtxt(csv, delimiter=',')
        return cls(paths=data, timeHorizon=data.shape[1]*timeStep, timeStep=timeStep, method="from_csv")

    def time_slice(self, start: float = 0.0, end: float = None) -> "StochasticProcess":
        if end is None:
            end = self.timeHorizon

        if start < 0 or end > self.timeHorizon:
            raise ValueError(
                f"Slice range [{start}, {end}] outside process horizon "
                f"[0, {self.timeHorizon}]"
            )
        if start >= end:
            raise ValueError("start must be less than end")

        start_idx = int(round(start / self.timeStep))
        end_idx = int(round(end / self.timeStep))

        #Deep copy the sliced paths to avoid modifying the original paths
        sliced_paths = self.paths[:, start_idx:end_idx].copy()

        return attr.evolve(
            self,
            paths=sliced_paths,
            timeHorizon=sliced_paths.shape[1] * self.timeStep,
            method=f"{self.method}_sliced",
        )

    def step_slice(self, start_idx: int = 0, end_idx: int = None) -> "StochasticProcess":
        if end_idx is None:
            end_idx = len(self)

        n_steps = len(self)
        if not (0 <= start_idx < end_idx <= n_steps):
            raise ValueError(
                f"Index range [{start_idx}, {end_idx}) invalid for {n_steps} steps"
            )

        #Deep copy the sliced paths to avoid modifying the original paths
        sliced_paths = self.paths[:, start_idx:end_idx].copy()

        return attr.evolve(
            self,
            paths=sliced_paths,
            timeHorizon=sliced_paths.shape[1] * self.timeStep,
            method=f"{self.method}_sliced",
        )