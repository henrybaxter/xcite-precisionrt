import os
import pytoml as toml

from simulator.report import generate_summary


with open('reports/results.toml') as fp:
    sims = toml.load(fp)['simulations']

import shutil
if os.path.exists('summary'):
    shutil.rmtree('summary')
generate_summary({'simulations': sims})
