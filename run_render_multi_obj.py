
import os
import subprocess

from render_helper import *
from settings import *

if __name__ == "__main__":
    
    #render rgb
    command = [g_blender_excutable_path, '--background', '--python', 'render_rgb_multi_obj.py']
    subprocess.run(command)