from datetime import datetime, timedelta


def _create_instance(config, ref_jobs, hash_converters):
    """Helper function that creates instance of a component from configuration."""
    if isinstance(config, dict):
        # Basic type checks
        if "class" not in config or "args" not in config:
            raise RuntimeError("Component configuration descriptor must have 'class' and 'args' properties.")

        # argument "@@ref_jobs" is replaced with ref_jobs list (special injection)
        def inject(val):
            if val == "@@ref_jobs":
                return ref_jobs
            elif val == "@@hash_converters":
                return hash_converters
            else:
                return val
        if isinstance(config["args"], dict):
            args = {key: inject(val) for key, val in config["args"].items()}
        elif isinstance(config["args"], list):
            args = [inject(arg) for arg in config["args"]]
        else:
            raise RuntimeError("Invalid component constructor args given in configuration descriptor.")

        return create_component(config["class"], args)
    else:
        return create_component(config)  # config is a string holding the class name


def create_component(class_name, constructor_args={}):
    """Create an instance of a component of given name.

    This is used in dynamic loading and component composition based on configuration file.
    class_name - fully qualified name of the class (module.submodule.classname)
    constructor_args - optional dict or list with args passed to constructor
    """
    components = class_name.split('.')
    module = __import__('.'.join(components[:-1]))
    for component in components[1:]:
        module = getattr(module, component)
    class_ = module
    if class_ is None:
        raise RuntimeError("Class {} not found.".format(class_name))

    # create instance
    if constructor_args and isinstance(constructor_args, dict):
        obj = class_(**constructor_args)
    elif constructor_args and isinstance(constructor_args, list):
        obj = class_(*constructor_args)
    else:
        obj = class_()

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
        print(f"{self.name}: {self.get_average_time()} s (count: {self.count}, total: {self.total_time.total_seconds()} s)")
