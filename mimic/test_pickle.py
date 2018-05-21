#! /usr/bin/env python
import sys
from pathlib import Path
import pickle

with Path(sys.argv[1]).open('rb') as f:
    print(pickle.load(f))
