import pathlib
from datetime import datetime, timedelta
from typing import IO, Optional


def _create_instance(instance_config, configuration: dict):
    """Helper function that creates instance of a component from configuration."""
    if isinstance(instance_config, dict):
        # Basic type checks
        if "class" not in instance_config or "args" not in instance_config:
            raise RuntimeError("Component configuration descriptor must have 'class' and 'args' properties.")

        # argument "@@ref_jobs" is replaced with ref_jobs list (special injection)
        def inject(val):
            if val == "@@ref_jobs":
                return configuration["ref_jobs"]
            elif val == "@@hash_converters":
                return configuration["hash_converters"]
            else:
                return val
        if isinstance(instance_config["args"], dict):
            args = {key: inject(val) for key, val in instance_config["args"].items()}
        elif isinstance(instance_config["args"], list):
            args = [inject(arg) for arg in instance_config["args"]]
        else:
            raise RuntimeError("Invalid component constructor args given in configuration descriptor.")

        return create_component(instance_config["class"], args, configuration=configuration)
    else:
        return create_component(instance_config, configuration=configuration)  # instance_config is a string holding the class name


def create_component(class_name, constructor_args=None, configuration=None):
    """Create an instance of a component of given name.

    This is used in dynamic loading and component composition based on configuration file.
    class_name - fully qualified name of the class (module.submodule.classname)
    constructor_args - optional dict or list with args passed to constructor
    """
    if constructor_args is None:
        constructor_args = {}
    if configuration is None:
        configuration = {}

    components = class_name.split('.')
    module = __import__('.'.join(components[:-1]))
    for component in components[1:]:
        module = getattr(module, component)
    class_ = module
    if class_ is None:
        raise RuntimeError("Class {} not found.".format(class_name))

    # create instance
    if constructor_args and isinstance(constructor_args, dict):
        obj = class_(**constructor_args, configuration=configuration)
    elif constructor_args and isinstance(constructor_args, list):
        obj = class_(*constructor_args, configuration=configuration)
    else:
        obj = class_(configuration=configuration)

    return obj


class Timer:

    def __init__(self, name):
        self.name = name
        self.total_time = timedelta()
        self.count = 0
        self.__start_time: datetime = ...

    def start(self):
        self.__start_time = datetime.now()

    def stop(self):
        end_time = datetime.now()
        self.count += 1
        self.total_time += end_time - self.__start_time

    def get_average_time(self):
        return (self.total_time / self.count).total_seconds()

    def print(self):
        log(f"{self.name}: {self.get_average_time()} s (count: {self.count}, total: {self.total_time.total_seconds()} s)")


log_file: Optional[IO] = None


def init_log(output_folder: 'pathlib.Path'):
    global log_file
    log_file = open(output_folder / "log.txt", "w")


def close_log():
    if log_file:
        log_file.close()


def log(message):
    print(message)
    print(message, file=log_file)


def log_with_time(message):
    print(f"{datetime.now()}: {message}")
    print(f"{datetime.now()}: {message}", file=log_file)
