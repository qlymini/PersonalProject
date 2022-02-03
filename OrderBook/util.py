import os


def dir_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)
