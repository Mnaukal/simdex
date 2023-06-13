from datetime import timedelta
import ruamel.yaml as yaml

from duration_predictors.nn_duration_predictor import NNDurationPredictor
from metrics.default import JobDelayMetricsCollector
from metrics.user_experience import UserExperienceMetricsCollector
from worker_selectors.q_network_worker_selector import QNetworkWorkerSelector


class CustomDumper(yaml.SafeDumper):
    # based on https://stackoverflow.com/a/65649481
    @staticmethod
    def float_to_str(f):
        return f"{f:.6f}"

    def represent_float(self, data):
        if data != data or (data == 0.0 and data == 1.0):
            value = '.nan'
        elif data == self.inf_value:
            value = '.inf'
        elif data == -self.inf_value:
            value = '-.inf'
        else:
            value = self.float_to_str(data).lower()
        return self.represent_scalar('tag:yaml.org,2002:float', value)


CustomDumper.add_representer(float, CustomDumper.represent_float)


def save_results(file_name, simulation, simulation_duration: timedelta):
    """
    Function to save the results for the June 2023 paper.

    It expects a certain structure of the simulation and the classes used for predictions. This probably won't work in the future.
    """
    try:
        data = {
            'simulation_duration': str(simulation_duration),
            'simulation_duration_s': simulation_duration.total_seconds()
        }

        for metric in simulation.metrics:
            if isinstance(metric, JobDelayMetricsCollector):
                data['jobs_total'] = metric.get_jobs()
                data['delay_avg'] = metric.get_avg_delay()
                data['delay_max'] = metric.get_max_delay()
            elif isinstance(metric, UserExperienceMetricsCollector):
                data['jobs_total'] = metric.get_total_jobs()
                data['jobs_on_time'] = metric.jobs_ontime
                data['jobs_delayed'] = metric.jobs_delayed
                data['jobs_late'] = metric.jobs_late
                data['jobs_on_time_percentage'] = metric.jobs_ontime / metric.get_total_jobs()
                data['jobs_delayed_percentage'] = metric.jobs_delayed / metric.get_total_jobs()
                data['jobs_late_percentage'] = metric.jobs_late / metric.get_total_jobs()

        def save_timer(timer, name):
            data[name] = timer.get_average_time()
            data[name + '_count'] = timer.count
            data[name + '_total'] = timer.total_time.total_seconds()

        save_timer(simulation.dispatch_timer, 'dispatch_time')

        if isinstance(simulation.duration_predictor, NNDurationPredictor):
            save_timer(simulation.duration_predictor.ml_monitor.inference_timer, 'duration_predictor_inference_time')
            save_timer(simulation.duration_predictor.ml_monitor.training_timer, 'duration_predictor_training_time')

        if isinstance(simulation.worker_selector, QNetworkWorkerSelector):
            save_timer(simulation.worker_selector.ml_monitor.inference_timer, 'worker_selector_inference_time')
            save_timer(simulation.worker_selector.ml_monitor.training_timer, 'worker_selector_training_time')

        with open(file_name, "w") as stream:
            yaml.dump(data, stream, default_flow_style=False, Dumper=CustomDumper)
    except:
        pass
