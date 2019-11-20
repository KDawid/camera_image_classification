import os
import shutil

def clean_folder(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)