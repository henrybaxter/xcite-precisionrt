import logging
import os
import asyncio

from .utils import remove, read_3ddose, run_command, XCITE_DIR

logger = logging.getLogger(__name__)


async def simulate(sim, template, beamlet):
    context = {
        'egsphant_path': os.path.join(XCITE_DIR, sim['phantom']),
        'phsp_path': beamlet['phsp'],
        'ncase': beamlet['stats']['total_photons'] * (sim['dose-recycle'] + 1),
        'nrcycl': sim['dose-recycle'],
        'n_split': sim['dose-photon-splitting'],
        # careful now. this is the distance from the end of the collimator to the isocenter
        # the isocenter may not be where the lesion is, depending on how we're trying to do things
        # so we want the bore diameter, nothing to do with the lesion
        'dsource': sim['bore-diameter'] / 2,
        'phicol': 90,
        'x': sim['isocenter'][0],
        'y': sim['isocenter'][1],
        'z': sim['isocenter'][2],
        'idat': 1,
        'phi': 90,
        'theta': 180
    }

    doses = {
        'arc': []
    }

    # stationary
    egsinp_str = template.format(**context)
    doses['stationary'] = simulate_dose(sim, beamlet, egsinp_str)

    # arc
    for i, theta in enumerate(dose_angles(sim)):
        context['theta'] = theta
        egsinp_str = template.format(**context)
        dose = simulate_dose(sim, beamlet, egsinp_str, theta)
        doses['arc'].append(dose)
        if i == sim['operations']:
            break
    futures = [doses['stationary']] + doses['arc']
    doses['stationary'], *doses['arc'] = await asyncio.gather(*futures)
    return doses


async def simulate_dose(sim, beamlet, egsinp_str, theta=180):
    folder = os.path.join(sim['egs-home'], 'dosxyznrc')
    # hash
    md5 = beamlet['hash'].copy()
    md5.update(egsinp_str.encode('utf-8'))
    base = md5.hexdigest()

    # dose properties
    dose = {
        'egsinp': os.path.join(folder, '{}.egsinp'.format(base)),
        '3ddose': os.path.join(folder, '{}.3ddose'.format(base)),
        'npz': os.path.join(folder, '{}.3ddose.npz'.format(base)),
        'egslst': os.path.join(folder, '{}.egslst'.format(base)),
        'theta': theta
    }

    if not os.path.exists(dose['npz']):
        # simulate
        remove(dose['3ddose'])
        with open(dose['egsinp'], 'w') as f:
            f.write(egsinp_str)
        command = ['dosxyznrc', '-p', sim['pegs4'], '-i', os.path.basename(dose['egsinp'])]
        out = await run_command(command, cwd=folder)
        if 'Warning' in out:
            logger.info('Warning in {}'.format(dose['egslst']))
        with open(dose['egslst'], 'w') as f:
            f.write(out)
        # use side effect to generate npz
        dose['dose'] = await read_3ddose(dose['3ddose'])
        remove(dose['3ddose'])  # save some space now we have npz

    return dose


def dose_angles(sim):
    angular_increment = 5  # degrees
    angular_sweep = 120  # degrees
    start_angle = 120
    n_angles = angular_sweep // angular_increment + 1
    angles = []
    for i in range(n_angles):
        angles.append(start_angle + i * angular_increment)
    return angles


if __name__ == '__main__':
    print(dose_angles('stuff'))
