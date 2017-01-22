import math
import os
import logging
import hashlib
import json

from . import egsinp

from .utils import run_command, remove

logger = logging.getLogger(__name__)


async def simulate(sim, templates, y):
    source_beamlet = await simulate_source(sim, templates['source'], y)
    filtered_beamlet = await filter_source(sim, templates['filter'], source_beamlet)
    collimated_beamlet = await collimate(sim, templates['collimator'], filtered_beamlet)
    result = {
        'source': source_beamlet,
        'filter': filtered_beamlet,
        'collimator': collimated_beamlet
    }
    return result

"""
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
"""


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
