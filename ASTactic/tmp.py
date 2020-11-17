import sys
import subprocess
import os

DETACHED_PROC = 0x00000008

for char in 'fghijklmnopqrstuvwxyz':
    # os.system("python extract_proof_steps.py --filter {}".format(char))
    proc = os.popen("python extract_proof_steps.py --filter {}".format(char))
    print("spawned filter {}: {}".format(char, proc.pid))
