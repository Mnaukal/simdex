import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf
import numpy as np

from constants import RUNTIME_ID_COUNT, EXERCISE_ID_COUNT
from interfaces import AbstractBatchedDurationPredictor, AbstractSystemMonitor

try:
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
except RuntimeError:  # "Inter op parallelism cannot be modified after initialization."
    pass


class MLModel:

    def __init__(self, layer_widths):
        self.model: tf.keras.Model = self._create_model(layer_widths)

    def _create_model(self, layer_widths):
        all_inputs, encoded_features = self._prepare_inputs()

        last_layer = tf.keras.layers.Concatenate()(encoded_features)
        for width in layer_widths:
            last_layer = tf.keras.layers.Dense(int(width), activation=tf.keras.activations.relu)(last_layer)
        output = tf.keras.layers.Dense(1, tf.keras.activations.exponential)(last_layer)

        model = tf.keras.Model(inputs=all_inputs, outputs=output)
        learning_rate = tf.keras.experimental.CosineDecay(0.01, 10000000)
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=tf.losses.Poisson())
        # model.summary()
        return model

    def _prepare_inputs(self):
        all_inputs = tf.keras.Input(shape=(2,), dtype='int32')
        encoded_features = []
        domain_sizes = [EXERCISE_ID_COUNT, RUNTIME_ID_COUNT]
        for idx in range(0, 2):
            encoding_layer = self._get_category_encoding_layer(domain_sizes[idx])
            encoded_col = encoding_layer(all_inputs[:, idx])
            encoded_features.append(encoded_col)

        return all_inputs, encoded_features

    @staticmethod
    def _get_category_encoding_layer(size):
        return lambda feature: tf.one_hot(feature, size + 1)  # +1 since classes are labeled from 1

    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)

    def predict(self, x, **kwargs) -> list:
        return self.model(x, **kwargs).numpy()


class Inference:

    def __init__(self):
        self.model: MLModel = ...

    def predict_batch(self, x) -> list:
        return self.model.predict(x, training=False)

    def set_model(self, model: MLModel):
        self.model = model


class DataStorage:

    def __init__(self):
        self.x_buffer = []
        self.y_buffer = []

    def add(self, x, y):
        self.x_buffer.append(x)
        self.y_buffer.append(y)

    @property
    def job_count(self) -> int:
        return len(self.x_buffer)

    def pop_batch(self) -> tuple:
        x = np.array(self.x_buffer)
        y = np.array(self.y_buffer)
        self.x_buffer = []
        self.y_buffer = []
        return x, y


class DataProcessor:

    @staticmethod
    def job_to_input(job):
        return [job.exercise_id, job.runtime_id]

    @staticmethod
    def job_to_target(job):
        return [job.duration]

    def jobs_to_inputs(self, jobs):
        return np.array([self.job_to_input(job) for job in jobs], dtype='int32')

    def process(self, job):
        return self.job_to_input(job), self.job_to_target(job)


class SystemMonitor(AbstractSystemMonitor):

    def __init__(self, parent: 'NNDurationPredictor'):
        self.parent = parent

    def job_done(self, simulation, job):
        self.parent.add_job(job)

    def ref_job_done(self, simulation, ref_job):
        self.parent.add_job(ref_job)


class MLMonitor:

    def __init__(self, parent: 'NNDurationPredictor', batch_size):
        self.parent = parent
        self.batch_size = batch_size

    def job_added(self, job):
        if self.parent.data_storage.job_count >= self.batch_size:
            self.parent.training.train()
            # TODO: update inference model


class Training:

    def __init__(self, parent: 'NNDurationPredictor', batch_size, batch_epochs):
        self.parent = parent
        self.batch_size = batch_size
        self.batch_epochs = batch_epochs

    def train(self, model=None):
        x, y = self.parent.data_storage.pop_batch()

        if model is None:
            model = self.parent.ml_model_storage.get_latest_model()
        # TODO: clone model

        model.fit(x, y, batch_size=self.batch_size, epochs=self.batch_epochs, verbose=False)
        # TODO: save model to model storage


class MLModelStorage:

    def __init__(self, **model_params):
        self.model_params = model_params
        self.models = [
            MLModel(**model_params)
        ]

    def get_latest_model(self):
        return self.models[-1]


class NNDurationPredictor(AbstractBatchedDurationPredictor):
    """Uses machine-learning neural-network regression model to predict the job duration.

    The model is trained in SA and used by dispatcher (via estimation function interface).
    The model is implemented in TensorFlow.
    """

    def __init__(self, layer_widths=[64], batch_size=5000, batch_epochs=5):
        super().__init__()

        self.system_monitor = SystemMonitor(self)
        self.ml_monitor = MLMonitor(self, batch_size)
        self.data_processor = DataProcessor()
        self.training = Training(self, batch_size, batch_epochs)
        self.ml_model_storage = MLModelStorage(layer_widths=layer_widths)
        self.data_storage = DataStorage()
        self.inference = Inference()

        self.inference.set_model(self.ml_model_storage.get_latest_model())

    def add_job(self, job):
        x, y = self.data_processor.process(job)
        self.data_storage.add(x, y)
        self.ml_monitor.job_added(job)

    def _predict_batch(self, jobs) -> list:
        x = self.data_processor.jobs_to_inputs(jobs)
        return self.inference.predict_batch(x)
