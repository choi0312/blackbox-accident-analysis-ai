# SPDX-License-Identifier: GPL-3.0-only
"""Standalone GPL Stage 3 program. Usage: worker.py DATA MODEL OUTPUT.csv"""
import os,sys,socket,types
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True

def denied(*args,**kwargs):
    raise RuntimeError('Stage 3 inference has no network access')
for name in ('connect','connect_ex','sendto'):setattr(socket.socket,name,denied)
socket.create_connection=denied;socket.getaddrinfo=denied
# A private package contains only this standalone motion program, not Stage 1/2.
pkg=types.ModuleType('_motion_program');pkg.__path__=[str(Path(__file__).resolve().parent)];sys.modules['_motion_program']=pkg
import torch
torch.set_num_threads(4)
from _motion_program.gpl_program import predict
if __name__=='__main__':
    if len(sys.argv)!=4:raise SystemExit('Usage: worker.py DATA MODEL OUTPUT.csv')
    predict(sys.argv[1],sys.argv[2]).to_csv(sys.argv[3],index=False)
