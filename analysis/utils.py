from os.path import dirname


def get_repo_dir():
    return dirname(dirname(__file__))
