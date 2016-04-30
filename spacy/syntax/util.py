import json
import pathlib

class Config(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get(self, attr, default=None):
        return self.__dict__.get(attr, default)

    @classmethod
    def write(cls, model_dir, name, **kwargs):
        model_dir = pathlib.Path(model_dir)
        with (model_dir / ('%s.json' % name)).open('w') as file_:
            file_.write(json.dumps(kwargs))

    @classmethod
    def read(cls, model_dir, name):
        model_dir = pathlib.Path(model_dir)
        with (model_dir / ('%s.json' % name)).open() as file_:
            return cls(**json.load(file_))
