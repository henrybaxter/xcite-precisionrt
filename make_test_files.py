from subprocess import run, PIPE, Popen
import os
import shutil

SIMULATION_FOLDER = '/Users/henry/projects/EGSnrc/egs_home/BEAM_TUMOTRAK/'
BEAMDPR_FOLDER = '/Users/henry/projects/beamdpr/test_data/'


def run_simulation(egsinp_path):
    result = run(['BEAM_TUMOTRAK', '-p', 'allkV', '-i', egsinp_path], stdout=PIPE, stderr=PIPE, cwd=SIMULATION_FOLDER)
    if result.returncode != 0:
        print(result.stderr.decode('utf-8'))
        print(result.stdout.decode('utf-8'))
    assert result.returncode == 0


def beamdp_combine(phsp1, phsp2, phspout):
    p = Popen(['beamdp'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    shutil.copy(phsp1, phspout)
    lines = [
        "y",  # show more detailed information
        "10",  # type of processing (10 = combine)
        phsp2,
        phspout,
        ""  # eof
    ]
    (stdout, stderr) = p.communicate("\n".join(lines).encode('utf-8'))

if __name__ == '__main__':
    egsinp = 'input.egsinp'
    base = os.path.splitext(egsinp)[0]
    try:
        path = os.path.join(SIMULATION_FOLDER, '{}.egsphsp1'.format(base))
        os.remove(path)
        print('Removed old phase space file', path)
    except IOError:
        print('Could not remove old phase space file', path)
    run_simulation(egsinp)
    print('Copying results to beamdpr folder')
    os.makedirs(BEAMDPR_FOLDER, exist_ok=True)
    for out_base in ['first', 'second']:
        for ext in ['egsinp', 'egsphsp1', 'egslst', 'egsgph']:
            src = os.path.join(SIMULATION_FOLDER, '{}.{}'.format(base, ext))
            dst = os.path.join(BEAMDPR_FOLDER, '{}.{}'.format(out_base, ext))
            shutil.copy(src, dst)
    # ok now we're going to run beamdp to combine the files into combined.egsphsp1
    phsp1 = os.path.join(BEAMDPR_FOLDER, 'first.egsphsp1')
    phsp2 = os.path.join(BEAMDPR_FOLDER, 'second.egsphsp1')
    phspout = os.path.join(BEAMDPR_FOLDER, 'combined.egsphsp1')
    beamdp_combine(phsp1, phsp2, phspout)
