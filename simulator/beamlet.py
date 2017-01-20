import asyncio
import math
import os
import logging
import hashlib
import json

from . import egsinp
from .utils import run_command, read_3ddose, remove, XCITE_DIR, regroup

logger = logging.getLogger(__name__)


async def simulate(sim, templates, index, y):
    source_beamlet = await simulate_source(sim, templates['source'], y)
    logger.info('{} - simulated source'.format(index))
    filtered_beamlet = await filter_source(sim, templates['filter'], source_beamlet)
    logger.info('{} - simulated filter'.format(index))
    collimated_beamlet = await collimate(sim, templates['collimator'], filtered_beamlet)
    logger.info('{} - simulated collimator'.format(index))
    if sim['reflect']:
        reflected_beamlet = await reflect(collimated_beamlet)
        dose = regroup(await asyncio.gather(*[
            simulate_doses(sim, templates, reflected_beamlet, index[0]),
            simulate_doses(sim, templates, collimated_beamlet, index[1])
        ]))
    else:
        dose = await simulate_doses(sim, templates, collimated_beamlet, index)
    result = {
        'source': source_beamlet,
        'filter': filtered_beamlet,
        'collimator': collimated_beamlet,
        'stationary': dose['stationary'],
        'arc': dose['arc']
    }
    logger.info('{} - simulated doses'.format(index))
    return result


async def reflect(original):
    folder = os.path.dirname(original['phsp'])

    md5 = original['hash'].copy()
    md5.update('reflect=y'.encode('utf-8'))
    base = md5.hexdigest()

    beamlet = {
        'egsinp': original['egsinp'],  # note that this is NOT reflected
        'phsp': os.path.join(folder, '{}.egsphsp'.format(base)),
        'hash': md5,
        'stats': original['stats']
    }
    temp_phsp = os.path.join(folder, '{}.egsphsp1'.format(base))

    if not os.path.exists(beamlet['phsp']):
        remove(temp_phsp)
        await run_command(['beamdpr', 'reflect', '-y', '1', original['phsp'], temp_phsp])
        os.rename(temp_phsp, beamlet['phsp'])

    return beamlet


async def simulate_source(sim, template, y):
    folder = os.path.join(sim['egs-home'], 'BEAM_RFLCT')
    # calculate
    theta = math.atan(y / sim['target-distance'])
    cos_x = -math.cos(theta)
    cos_y = math.copysign(math.sqrt(1 - cos_x * cos_x), y)

    # prepare template
    template['uinc'] = cos_x
    template['vinc'] = cos_y

    # hash
    egsinp_str = egsinp.unparse_egsinp(template)
    md5 = hashlib.md5(egsinp_str.encode('utf-8'))
    base = md5.hexdigest()

    # beamlet properties
    beamlet = {
        'egsinp': os.path.join(folder, '{}.egsinp'.format(base)),
        'phsp': os.path.join(folder, '{}.egsphsp'.format(base)),
        'hash': md5
    }
    temp_phsp = os.path.join(folder, '{}.egsphsp1'.format(base))

    if not os.path.exists(beamlet['phsp']):
        # simulate
        remove(temp_phsp)
        with open(beamlet['egsinp'], 'w') as f:
            f.write(egsinp_str)
        command = ['BEAM_RFLCT', '-p', sim['pegs4'], '-i', os.path.basename(beamlet['egsinp'])]
        await run_command(command, cwd=folder)
        # translate
        await run_command(['beamdpr', 'translate', '-i', temp_phsp, '-y', '({})'.format(y)])
        # rotate
        angle = str(math.pi / 2)
        await run_command(['beamdpr', 'rotate', '-i', temp_phsp, '-a', angle])
        os.rename(temp_phsp, beamlet['phsp'])

    # stats
    command = ['beamdpr', 'stats', '--format=json', beamlet['phsp']]
    beamlet['stats'] = json.loads(await run_command(command))

    return beamlet


async def filter_source(sim, template, source_beamlet):
    folder = os.path.join(sim['egs-home'], 'BEAM_FILTR')
    # prepare template
    template['ncase'] = source_beamlet['stats']['total_particles']
    template['spcnam'] = os.path.join('../', 'BEAM_RFLCT', os.path.basename(source_beamlet['phsp']))

    # hash
    egsinp_str = egsinp.unparse_egsinp(template)
    md5 = source_beamlet['hash'].copy()
    md5.update(egsinp_str.encode('utf-8'))
    base = md5.hexdigest()

    # beamlet properties
    beamlet = {
        'egsinp': os.path.join(folder, '{}.egsinp'.format(base)),
        'phsp': os.path.join(folder, '{}.egsphsp'.format(base)),
        'hash': md5
    }
    temp_phsp = os.path.join(folder, '{}.egsphsp1'.format(base))

    if not os.path.exists(beamlet['phsp']):
        # simulate
        remove(temp_phsp)
        with open(beamlet['egsinp'], 'w') as f:
            f.write(egsinp_str)
        command = ['BEAM_FILTR', '-p', sim['pegs4'], '-i', os.path.basename(beamlet['egsinp'])]
        await run_command(command, cwd=folder)
        os.rename(temp_phsp, beamlet['phsp'])

    # stats
    command = ['beamdpr', 'stats', '--format=json', beamlet['phsp']]
    beamlet['stats'] = json.loads(await run_command(command))

    return beamlet


async def collimate(sim, template, source_beamlet):
    name = 'BEAM_{}'.format(template['title'])
    folder = os.path.join(sim['egs-home'], name)

    # prepare template
    template['ncase'] = source_beamlet['stats']['total_particles']
    template['spcnam'] = '../BEAM_FILTR/' + os.path.basename(source_beamlet['phsp'])

    # hash
    egsinp_str = egsinp.unparse_egsinp(template)
    md5 = source_beamlet['hash'].copy()
    md5.update(egsinp_str.encode('utf-8'))
    base = md5.hexdigest()

    # beamlet properties
    beamlet = {
        'egsinp': os.path.join(folder, '{}.egsinp'.format(base)),
        'phsp': os.path.join(folder, '{}.egsphsp'.format(base)),
        'hash': md5
    }
    temp_phsp = os.path.join(folder, '{}.egsphsp1'.format(base))

    if not os.path.exists(beamlet['phsp']):
        # simulate
        remove(temp_phsp)
        with open(beamlet['egsinp'], 'w') as f:
            f.write(egsinp_str)
        command = [name, '-p', sim['pegs4'], '-i', os.path.basename(beamlet['egsinp'])]
        await run_command(command, cwd=folder)
        os.rename(temp_phsp, beamlet['phsp'])

    # stats
    command = ['beamdpr', 'stats', '--format=json', beamlet['phsp']]
    beamlet['stats'] = json.loads(await run_command(command))

    return beamlet


async def simulate_doses(sim, templates, beamlet, index):
    context = {
        'egsphant_path': os.path.join(XCITE_DIR, sim['phantom']),
        'phsp_path': beamlet['phsp'],
        'ncase': beamlet['stats']['total_photons'] * (sim['dose-recycle'] + 1),
        'nrcycl': sim['dose-recycle'],
        'n_split': sim['dose-photon-splitting'],
        'dsource': sim['collimator']['lesion-distance'],
        'phicol': 0,
        'x': sim['phantom-isocenter'][0],
        'y': sim['phantom-isocenter'][1],
        'z': sim['phantom-isocenter'][2],
        'idat': 1,
        'theta': 180,
    }

    doses = {
        'arc': []
    }

    # stationary
    context['phi'] = 0
    egsinp_str = templates['stationary_dose'].format(**context)
    doses['stationary'] = simulate_dose(sim, beamlet, egsinp_str, index)

    # arc
    for i, (phimin, phimax) in enumerate(dose_angles(sim)):
        context['nang'] = 1
        context['phimin'] = phimin
        context['phimax'] = phimax
        egsinp_str = templates['arc_dose'].format(**context)
        dose = simulate_dose(sim, beamlet, egsinp_str, index, phimin, phimax)
        doses['arc'].append(dose)
        if i == sim['operations']:
            break
    futures = [doses['stationary']] + doses['arc']
    doses['stationary'], *doses['arc'] = await asyncio.gather(*futures)
    return doses


async def simulate_dose(sim, beamlet, egsinp_str, index, phimin=0, phimax=0):
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
        'index': index,
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


def dose_angles(args):
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
