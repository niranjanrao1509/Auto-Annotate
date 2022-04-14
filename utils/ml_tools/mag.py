import os
os.chdir("/data/envs/tika-tools/tikapamtxt/models/mag")
cwd = os.getcwd() 
print("Current working directory is:", cwd) 
from magpie import Magpie

