import os

def my_makedir(name: str):
    try:
        os.makedirs(name=name)
    except OSError:
        pass