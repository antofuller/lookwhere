import yaml
from types import SimpleNamespace


class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            elif value == "None":
                self.__setattr__(key, None)
            else:
                self.__setattr__(key, value)

    def as_dict(self):
        return {
            key: value.as_dict() if isinstance(value, NestedNamespace) else value
            for key, value in self.items()
        }

    def get(self, key, default=None):
        return getattr(self, key, default)

    def items(self):
        return self.__dict__.items()

    def __setattr__(self, name, value):
        return super().__setattr__(name, value)


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return NestedNamespace(config)
