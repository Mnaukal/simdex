"""
Neural-network-based regression model to predict the job duration.
The software architecture of the predictor consists of several components.
More details are described in the paper.
"""

import os

from helpers import Timer, log_with_time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf
import numpy as np
from collections import deque

from constants import RUNTIME_ID_COUNT, EXERCISE_ID_COUNT
from interfaces import AbstractBatchedDurationPredictor, AbstractSystemMonitor

try:
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
except RuntimeError:  # "Inter op parallelism cannot be modified after initialization."
    pass


class MLModel:
    """
    Implementation of a feedforward neural network model (in TensorFlow).
    Inputs: [job.exercise_id, job.runtime_id]
    Output: job.duration
    """

    def __init__(self, copy_from: 'MLModel' = None, **model_params):
        if copy_from:
            self.model = tf.keras.models.clone_model(copy_from.model)
        else:
            self.model: tf.keras.Model = self._create_model(**model_params)

        self._compile()

        # run the model once with a dummy input to initialize it
        self.model(tf.zeros([1 if s is None else s for s in self.model.input_shape]))

    def _create_model(self, **model_params):
        layer_widths = model_params['layer_widths']
        all_inputs, encoded_features = self._prepare_inputs()

        last_layer = tf.keras.layers.Concatenate()(encoded_features)
        for width in layer_widths:
            last_layer = tf.keras.layers.Dense(int(width), activation=tf.keras.activations.relu)(last_layer)
        output = tf.keras.layers.Dense(1, tf.keras.activations.exponential)(last_layer)

        model = tf.keras.Model(inputs=all_inputs, outputs=output)
        return model

    def _compile(self):
        learning_rate = tf.keras.experimental.CosineDecay(0.01, 10_000_000)
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=tf.losses.Poisson(), metrics=['mse'])
        # model.summary()

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
        """Train the model on the given data. Both x and y are batches of training examples."""
        return self.model.fit(x, y, **kwargs)

    def predict(self, x, **kwargs) -> list:
        """Predict the job duration for a given batch of data."""
        return self.model(x, **kwargs).numpy()


class Inference:
    """Handles the inference (the prediction of the job duration)"""

    def __init__(self):
        self.model: MLModel = ...

    def predict_batch(self, x) -> list:
        return self.model.predict(x, training=False)

    def set_model(self, model: MLModel):
        self.model = model


class DataStorage:
    """Stores the training data."""

    def __init__(self):
        self.x_buffer = []
        self.y_buffer = []

    def add(self, x, y):
        self.x_buffer.append(x)
        self.y_buffer.append(y)

    @property
    def job_count(self) -> int:
        return len(self.x_buffer)

    def get_batch(self, batch_size) -> tuple:
        x = np.array(self.x_buffer[-batch_size:])
        y = np.array(self.y_buffer[-batch_size:])
        return x, y


class DataPreprocessor:
    """Preprocesses the data (both for training and inference) into a format suitable for the neural network model."""

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
    """Monitors the state of the system and handles the events such as job_finished."""

    def __init__(self, parent: 'NNDurationPredictor'):
        self.parent = parent

    def job_finished(self, simulation, job):
        self.parent.add_training_datum(job)

    def ref_job_finished(self, simulation, ref_job):
        self.parent.add_training_datum(ref_job)


class MLMonitor:
    """Monitors the ML inference, data storage, etc. and initiates training and updating the inference model."""

    def __init__(self, parent: 'NNDurationPredictor', training_interval, configuration):
        self.parent = parent
        self.training_interval = training_interval
        self.data_since_last_training = 0

        self.inference_timer = Timer("ML Duration Predictor inference time", configuration["output_folder"] / "duration_predictor_inference_times.csv")
        self.training_timer = Timer("ML Duration Predictor training time", configuration["output_folder"] / "duration_predictor_training_times.csv")

    def job_added(self, job):
        """Called when a new training example (job) is added to the DataStorage."""
        self.data_since_last_training += 1
        if self.data_since_last_training >= self.training_interval:
            self.parent.training.train()
            self.parent.update_inference_model()
            self.data_since_last_training = 0


class Training:
    """Performs the training of the neural network model."""

    def __init__(self, parent: 'NNDurationPredictor', batch_size, training_epochs):
        self.parent = parent
        self.batch_size = batch_size
        self.training_epochs = training_epochs

    def train(self, original_model=None):
        if self.parent.data_storage.job_count < self.batch_size:
            return

        self.parent.ml_monitor.training_timer.start()

        x, y = self.parent.data_storage.get_batch(self.batch_size)

        if original_model is None:
            original_model = self.parent.ml_model_storage.get_latest_model()

        model = MLModel(original_model)

        history = model.fit(x, y, batch_size=self.batch_size, epochs=self.training_epochs, verbose=False)
        # log_with_time(f"Training done. MSE: {[f'{e:.2f}' for e in history.history['mse']]}")

        self.parent.ml_model_storage.save_model(model)

        self.parent.ml_monitor.training_timer.stop()


class MLModelStorage:
    """The trained ML models are stored in this class."""

    def __init__(self):
        self.models = deque(maxlen=10)  # keep the last 10 models

    def get_latest_model(self):
        return self.models[-1]

    def save_model(self, model):
        self.models.append(model)


class NNDurationPredictor(AbstractBatchedDurationPredictor):
    """Uses neural network regression model to predict the job duration. The model is implemented in TensorFlow."""

    def __init__(self, layer_widths=[256], training_interval=1000, batch_size=1000, training_epochs=5, configuration=None):
        super().__init__(configuration)

        self.system_monitor = SystemMonitor(self)
        self.ml_monitor = MLMonitor(self, training_interval, configuration)
        self.data_processor = DataPreprocessor()
        self.training = Training(self, batch_size, training_epochs)
        self.ml_model_storage = MLModelStorage()
        self.data_storage = DataStorage()
        self.inference = Inference()

        self.model_params = {
            "layer_widths": layer_widths
        }

    def init(self, _simulation):
        initial_model = self._create_initial_model()
        self.ml_model_storage.save_model(initial_model)
        self.update_inference_model()

    def end(self, simulation):
        self.ml_monitor.inference_timer.print()
        self.ml_monitor.training_timer.print()
        self.ml_monitor.inference_timer.write()
        self.ml_monitor.training_timer.write()

    def _create_initial_model(self):
        return MLModel(None, **self.model_params)

    def update_inference_model(self):
        self.inference.set_model(self.ml_model_storage.get_latest_model())

    def add_training_datum(self, job):
        x, y = self.data_processor.process(job)
        self.data_storage.add(x, y)
        self.ml_monitor.job_added(job)

    def _predict_batch(self, jobs) -> list:
        self.ml_monitor.inference_timer.start()
        x = self.data_processor.jobs_to_inputs(jobs)
        predictions = self.inference.predict_batch(x)
        self.ml_monitor.inference_timer.stop()
        return predictions
