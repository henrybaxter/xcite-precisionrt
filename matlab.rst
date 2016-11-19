egsgeom = '/Users/henry/projects/EGSnrc/egs_home/BEAM_TUMOTRAK/input.egsgeom';
egsgph = '/Users/henry/projects/EGSnrc/egs_home/BEAM_TUMOTRAK/input.egsgph';
exist(egsgeom, 'file')
exist(egsgph, 'file')
EGS_WINDOWS(egsgeom, egsgph)
