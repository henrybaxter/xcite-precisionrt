import logging
import os
import asyncio

from .utils import remove, read_3ddose, run_command, XCITE_DIR

logger = logging.getLogger(__name__)


async def simulate(sim, templates, beamlet):
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
        'phicol': 0,
        'x': sim['isocenter'][0],
        'y': sim['isocenter'][1],
        'z': sim['isocenter'][2],
        'idat': 1,
        'theta': 180,
    }

    doses = {
        'arc': []
    }

    # stationary
    context['phi'] = 0
    egsinp_str = templates['stationary_dose'].format(**context)
    doses['stationary'] = simulate_dose(sim, beamlet, egsinp_str)

    # arc
    for i, (phimin, phimax) in enumerate(dose_angles(sim)):
        context['nang'] = 1
        context['phimin'] = phimin
        context['phimax'] = phimax
        egsinp_str = templates['arc_dose'].format(**context)
        dose = simulate_dose(sim, beamlet, egsinp_str, phimin, phimax)
        doses['arc'].append(dose)
        if i == sim['operations']:
            break
    futures = [doses['stationary']] + doses['arc']
    doses['stationary'], *doses['arc'] = await asyncio.gather(*futures)
    return doses


async def simulate_dose(sim, beamlet, egsinp_str, phimin=0, phimax=0):
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
        'phimin': phimin,
        'phimax': phimax
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
    # recall that 180 theta is center
    # and we do (theta, phi)
    # we just want pairs of phimin and phimax
    angular_increment = 5  # degrees
    angular_sweep = 120  # degrees
    n_angles = angular_sweep // angular_increment
    angles = []
    for i in range(n_angles):
        angles.append((120 + i * angular_increment, 120 + (i + 1) * angular_increment))
    return angles
