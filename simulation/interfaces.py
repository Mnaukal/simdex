#
# Base classes for important components and helper functions for dynamic loading and instantiation of their implementations.
#
import abc
from typing import TYPE_CHECKING

from workers import WorkerQueue

if TYPE_CHECKING:
    from simulation import Simulation
    from jobs import Job, RefJob


class AbstractMetricsCollector(abc.ABC):
    """Base class for all metrics collectors.

    The snapshot() is invoked before every MAPE-K loop invocation to capture workers state periodically.
    The job_finished() is invoked for every job after it is removed from the queue.
    """

    def snapshot(self, ts, workers):
        pass  # an empty placeholder that just declares the interface

    def job_finished(self, job):
        pass  # an empty placeholder that just declares the interface

    @abc.abstractmethod
    def print(self, machine=False, verbose=False):
        """Print the metrics to std. output.

        machine - if true, the output should be printed without any human-readable comments
        verbose - if true, more verbose output is given (if available)
        """
        pass


class AbstractDispatcher(abc.ABC):
    """Base class (interface declaration) for all dispatchers.

    Dispatcher is responsible for assigning jobs into worker queues.
    Dispatcher itself runs a fixed algorithm, but its behavior may be altered in MAPE-K loop.
    """

    def __init__(self, configuration: dict):
        super().__init__()

    def init(self, simulation: 'Simulation'):
        """Initialize the dispatcher before the first job."""
        pass

    @abc.abstractmethod
    def dispatch(self, job: 'Job', workers: list['WorkerQueue'], simulation: 'Simulation'):
        """Assign given job to one of the workers."""
        pass


class AbstractSystemMonitor:

    def periodic_monitoring(self, simulation: 'Simulation'):
        """The method is called periodically."""
        pass

    def job_dispatched(self, simulation: 'Simulation', job: 'Job'):
        """The method is called after a job is dispatched."""
        pass

    def job_done(self, simulation: 'Simulation', job: 'Job'):
        """The method is called after a job is finished."""
        pass

    def ref_job_done(self, simulation: 'Simulation', ref_job: 'RefJob'):
        """The method is called after a reference job is finished."""
        pass


class OnlineMLComponent:

    def __init__(self):
        self.system_monitor: AbstractSystemMonitor = self._init_system_monitor()

    def _init_system_monitor(self) -> AbstractSystemMonitor:
        return AbstractSystemMonitor()


class AbstractDurationPredictor(OnlineMLComponent, abc.ABC):

    def __init__(self, configuration: dict):
        super().__init__()

    def init(self, simulation: 'Simulation'):
        pass

    def end(self, simulation: 'Simulation'):
        pass

    @abc.abstractmethod
    def predict_duration(self, job) -> float:
        """Predict the duration of the given job."""
        return 0.0


class AbstractBatchedDurationPredictor(AbstractDurationPredictor, abc.ABC):
    """TODO: predicts in batches to speed up the simulation."""

    def __init__(self, configuration: dict):
        super().__init__(configuration)
        self.duration_prediction_cache = {}

    @abc.abstractmethod
    def _predict_batch(self, jobs) -> list:
        pass

    def precompute_batch(self, jobs):
        predictions = self._predict_batch(jobs)
        self.duration_prediction_cache = dict(zip(jobs, predictions))

    def predict_duration(self, job) -> float:
        if job in self.duration_prediction_cache:
            return float(self.duration_prediction_cache[job])
        else:
            return float(self._predict_batch([job])[0])


class AbstractWorkerSelector(OnlineMLComponent, abc.ABC):

    def __init__(self, configuration: dict):
        super().__init__()

    def init(self, simulation: 'Simulation'):
        pass

    def end(self, simulation: 'Simulation'):
        pass

    @abc.abstractmethod
    def select_worker(self, simulation: 'Simulation', job: 'Job') -> int:
        """Select the best worker for the given job and return its index."""
        return 0
