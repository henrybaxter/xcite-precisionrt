python collimator.py 1-row-1-cm.egsinp --target-width=1 --hole-radius=.2 && beamviz 1-row-1-cm.egsinp
python collimator.py 1-row-2-cm.egsinp --target-width=2 --hole-radius=.3 && beamviz 1-row-1-cm.egsinp
python collimator.py 1-row-4-cm.egsinp --target-width=4 --hole-radius=.51 && beamviz 1-row-1-cm.egsinp
python collimator_analyzer.py 1-row-*egsinp
