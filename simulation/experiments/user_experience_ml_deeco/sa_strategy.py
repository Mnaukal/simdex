import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf

from ml_deeco.estimators import NeuralNetworkEstimator
from ml_deeco.simulation import Experiment

from interfaces import AbstractSelfAdaptingStrategy

from ml_deeco.utils.verbose import setVerboseLevel
setVerboseLevel(0)


# TODO: split Experiment into ExperimentBase (with only the list of estimators)?
class CategorySelfAdaptingStrategy(AbstractSelfAdaptingStrategy, Experiment):
    """TODO: Uses machine-learning neural-network regression model to predict the job duration.

    The model is trained in SA and used by dispatcher (via estimation function interface).
    The model is implemented in TensorFlow.
    """

    def __init__(self, layers_widths=[64], batch_size=5000, batch_epochs=5, ref_jobs=None):
        super().__init__()
        self.jobsSinceLastTraining = 0

        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(8)

        if ref_jobs:
            self.ref_jobs = ref_jobs[:]
            self.ref_jobs.reverse()
        else:
            self.ref_jobs = None

        self.jobDurationEstimator = NeuralNetworkEstimator(
            experiment=self,
            hidden_layers=layers_widths,
            fit_params={
                "batch_size": batch_size,
                "epochs": batch_epochs,
            },
            accumulateData=False,
            testSplit=0,
            optimizer=tf.optimizers.Adam(learning_rate=tf.keras.experimental.CosineDecay(0.01, 10000000))
        )
        self.dispatcher = None
        self.batch_size = batch_size

    def _advance_ts(self, ts):
        while self.ref_jobs and self.ref_jobs[-1].spawn_ts + self.ref_jobs[-1].duration <= ts:
            job = self.ref_jobs.pop()
            if job.compilation_ok:
                self.dispatcher.collect_job_for_training(job)

    def _train(self):
        """Train the estimator."""
        for estimator in self.estimators:
            estimator.endIteration()
        self.useBaselines = False

    def init(self, ts, dispatcher, workers):
        self.dispatcher = dispatcher
        dispatcher.initEstimates(self)
        self.jobDurationEstimator.init()

    def do_adapt(self, ts, dispatcher, workers, job=None):
        self._advance_ts(ts)
        if job is None:
            if self.jobsSinceLastTraining >= self.batch_size:
                self._train()
                self.jobsSinceLastTraining = 0
        else:
            self.jobsSinceLastTraining += 1

    def prepareSimulation(self, _iteration, _run):
        # we don't use the Experiment.run or Experiment.runSimulation methods, so we don't need to implement this.
        pass
