import os

def my_makedir(name: str):
    try:
        os.mkdir(name)
    except OSError:
        pass