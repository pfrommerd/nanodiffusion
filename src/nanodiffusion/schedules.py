from nanoconfig import config

import abc

from smalldiffusion import Schedule, ScheduleLogLinear

@config
class ScheduleConfig(abc.ABC):
    timesteps: int

    @abc.abstractmethod
    def create(self) -> Schedule:
        """
        Create the schedule.
        :return: The schedule.
        """
        pass

@config(variant="loglinear")
class LogLinearScheduleConfig(ScheduleConfig):
    sigma_min: float = 1e-3
    sigma_max: float = 1e0

    def create(self) -> ScheduleLogLinear:
        return ScheduleLogLinear(
            N=self.timesteps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max
        )