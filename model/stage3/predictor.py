"""Invoke the separately licensed Stage 3 command-line program through CSV IPC.

No GPL Stage 3 modules are imported into the Stage 1/2 interpreter. Each program
retains its license and corresponding source; see THIRD_PARTY_NOTICES.md.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import pandas as pd


def predict(data_dir, model_dir):
    """Run GPL Stage 3 in an isolated process and read its schema-stable CSV."""
    stage=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='blackbox_stage3_') as temporary:
        output=Path(temporary)/'predictions.csv'
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        # A list argument (rather than a shell command) keeps all paths literal.
        result=subprocess.run([sys.executable,str(stage/'worker.py'),str(Path(data_dir).resolve()),str(Path(model_dir).resolve()),str(output)],env=env,capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError('Stage 3 local program failed: '+result.stderr[-3000:])
        return pd.read_csv(output,dtype={'ID':str})
