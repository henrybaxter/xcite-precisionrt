import math
import os
import logging
import hashlib
import json

import egsinp
from utils import run_command

logger = logging.getLogger(__name__)

def remove_phsp(phsp):
    try:
        os.remove(phsp)
        logger.info('Removed old phase space file {}'.format(phsp))
    except IOError:
        pass

def executable(name):
    return 'BEAM_{}'.format(name)

async def simulate_source(args, template, y):
    folder = os.path.join(args.egs_home, 'BEAM_RFLCT')
    # calculate
    theta = math.atan(y / args.beam_distance)
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
        'phsp': os.path.join(folder, '{}.egsphsp1'.format(base)),
        'hash': md5
    }

    if args.overwrite or not os.path.exists(beamlet['phsp']):
        # simulate
        remove_phsp(beamlet['phsp'])
        open(beamlet['egsinp'], 'w').write(egsinp_str)
        command = ['BEAM_RFLCT', '-p', args.pegs4, '-i', os.path.basename(beamlet['egsinp'])]
        await run_command(command, cwd=folder)
        # translate
        await run_command(['beamdpr', 'translate', '-i', beamlet['phsp'], '-y', '({})'.format(y)])
        # rotate
        angle = str(math.pi / 2)
        await run_command(['beamdpr', 'rotate', '-i', beamlet['phsp'], '-a', angle])

    # stats
    command = ['beamdpr', 'stats', '--format=json', beamlet['phsp']]
    beamlet['stats'] = json.loads(await run_command(command))
    return beamlet


async def filter_source(args, template, source_beamlet):
    folder = os.path.join(args.egs_home, 'BEAM_FILTR')
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
        'phsp': os.path.join(folder, '{}.egsphsp1'.format(base)),
        'hash': md5
    }

    if args.overwrite or not os.path.exists(beamlet['phsp']):
        # simulate
        remove_phsp(beamlet['phsp'])
        open(beamlet['egsinp'], 'w').write(egsinp_str)
        command = ['BEAM_FILTR', '-p', args.pegs4, '-i', os.path.basename(beamlet['egsinp'])]
        await run_command(command, cwd=folder)

    # stats
    command = ['beamdpr', 'stats', '--format=json', beamlet['phsp']]
    beamlet['stats'] = json.loads(await run_command(command))
    return beamlet


async def collimate(args, template, source_beamlet):
    name = 'BEAM_{}'.format(template['title'])
    folder = os.path.join(args.egs_home, name)

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
        'phsp': os.path.join(folder, '{}.egsphsp1'.format(base)),
        'hash': md5
    }

    if args.overwrite or not os.path.exists(beamlet['phsp']):
        # simulate
        remove_phsp(beamlet['phsp'])
        open(beamlet['egsinp'], 'w').write(egsinp_str)
        command = [name, '-p', args.pegs4, '-i', os.path.basename(beamlet['egsinp'])]
        await run_command(command, cwd=folder)

    # stats
    command = ['beamdpr', 'stats', '--format=json', beamlet['phsp']]
    beamlet['stats'] = json.loads(await run_command(command))
    return beamlet

async def dose_simulations(folder, pegs4, simulations):
    to_simulate = []
    for simulation in simulations:
        # we only check for the compressed version, since we delete the 3ddose file to save space
        if not os.path.exists(simulation['dose'] + '.npz'):
            to_simulate.append(simulation)
    logger.info('Reusing {} and running {} dose calculations'.format(
        len(simulations) - len(to_simulate), len(to_simulate)))
    dose = functools.partial(dose_simulation, folder, pegs4)
    start = time.time()
    for i, result in enumerate(pool.imap(dose, to_simulate)):
        elapsed = time.time() - start
        portion_complete = (i + 1) / len(to_simulate)
        estimated_remaining = elapsed / portion_complete
        logger.info('{} of {} simulations complete, {:.2f} mimnutes remaining'.format(
            i + 1, len(to_simulate), estimated_remaining / 60))


async def dose_simulation(folder, pegs4, simulation):
    try:
        os.remove(simulation['dose'])
    except IOError:
        pass
    command = ['dosxyznrc', '-p', pegs4, '-i', simulation['egsinp']]
    await run_command(command, cwd=folder)
    egslst = os.path.join(folder, simulation['egsinp'].replace('.egsinp', '.egslst'))
    logger.info('Writing to {}'.format(egslst))
    open(egslst, 'w').write(out)
    if 'Warning' in out:
        logger.info('Warning in {}'.format(egslst))
    py3ddose.read_3ddose(simulation['dose'])
    os.remove(simulation['dose'])


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


def fast_dose(beamlets, args):
    logger.info('Fast dosing')
    templates = {
        'stationary': open(args.dose_egsinp).read(),
        'arc': open(args.arc_dose_egsinp).read()
    }
    folder = os.path.join(args.egs_home, 'dosxyznrc')
    for stage in ['stationary', 'arc']:
        for i, beamlet in enumerate(beamlets):
            # run two simulations, normal and arced
            context = {
                'egsphant_path': os.path.join(SCRIPT_DIR, args.phantom),
                'phsp_path': beamlet['phsp'],
                'ncase': beamlet['stats']['total_photons'] * (args.dose_recycle + 1),
                'nrcycl': args.dose_recycle,
                'n_split': args.dose_photon_splitting,
                'dsource': args.target_distance,
                'phicol': 90,
                'x': args.target_x,
                'y': args.target_y,
                'z': args.target_z,
                'idat': 1,
                # only for stationary
                'theta': 180,
                'phi': 0
            }
            egsinp_str = templates[stage].format(**context)
            md5 = beamlet['hash'].copy()
            md5.update(egsinp_str.encode('utf-8'))
            base = md5.hexdigest()
            inp = '{}.egsinp'.format(base)
            inp_path = os.path.join(folder, inp)
            open(inp_path, 'w').write(egsinp_str)
            dose_filename = '{}.3ddose'.format(base)
            dose_path = os.path.join(folder, dose_filename)
            simulations.setdefault(stage, []).append({
                'egsinp': inp,
                'dose': dose_path
            })
            doses.setdefault(stage, []).append({
                'dose': dose_path,
                'hash': md5
            })
        dose_simulations(folder, args.pegs4, simulations[stage])
        index = len(simulations) // 2
        egslst = os.path.join(folder, simulations[stage][index]['egsinp'].replace('.egsinp', '.egslst'))
        shutil.copy(egslst, os.path.join(args.output_dir, '{}_dose{}.egslst'.format(stage, index)))
        _egsinp = os.path.join(folder, simulations[stage][index]['egsinp'])
        shutil.copy(_egsinp, os.path.join(args.output_dir, '{}_dose{}.egsinp'.format(stage, index)))
        for i, dose in enumerate(doses[stage]):
            ipath = dose['dose'] + '.npz'
            opath = os.path.join(args.output_dir, '{}_dose/{}_dose{}.3ddose.npz'.format(stage, stage, i))
            shutil.copy(ipath, opath)
    return doses


def slow_dose(beamlets, args):
    logger.info('Slow dosing')
    template = open(args.dos_egsinp).read()
    folder = os.path.join(args.egs_home, 'dosxyznrc')
    dose_contributions = []
    simulations = []
    for i, beamlet in enumerate(beamlets):
        angled_dose_contributions = {}
        for j, (theta, phi) in enumerate(dose_angles(args)):
            kwargs = {
                'egsphant_path': os.path.join(SCRIPT_DIR, args.phantom),
                'phsp_path': beamlet['phsp'],
                'ncase': beamlet['stats']['total_photons'] * (args.dose_recycle + 1),
                'nrcycl': args.dose_recycle,
                'n_split': args.dose_photon_splitting,
                'dsource': args.target_distance,
                'theta': theta,
                'phi': phi,
                'phicol': 90
            }
            logger.info('Dose using each particle {} times so {} histories'.format(
                kwargs['nrcycl'] + 1, kwargs['ncase']))
            egsinp_str = template.format(**kwargs)
            md5 = beamlet['hash'].copy()
            md5.update(egsinp_str.encode('utf-8'))
            base = md5.hexdigest()
            inp = '{}.egsinp'.format(base)
            inp_path = os.path.join(folder, inp)
            open(inp_path, 'w').write(egsinp_str)
            dose_filename = '{}.3ddose'.format(base)
            dose_path = os.path.join(folder, dose_filename)
            simulations.append({
                'egsinp': inp,
                'dose': dose_path
            })
            angled_dose_contributions[(theta, phi)] = {
                'dose': dose_path,
                'hash': md5
            }
        dose_contributions.append(angled_dose_contributions)
    shutil.copy(inp_path, os.path.join(args.output_dir, 'last_dose.egsinp'))
    dose_simulations(folder, args.pegs4, simulations)
    egslst = inp_path.replace('.egsinp', '.egslst')
    shutil.copy(egslst, os.path.join(args.output_dir, 'last_dose.egslst'))
    for i, beamlet_contribution in enumerate(dose_contributions):
        for (theta, phi), contribution in beamlet_contribution.items():
            slug = '{}_{}_{}'.format(i, theta, phi)
            ipath = contribution['dose'] + '.npz'
            opath = os.path.join(args.output_dir, 'dose/dose{}.3ddose.npz'.format(slug))
            shutil.copy(ipath, opath)
    return dose_contributions


async def simulate(args, templates, i, y):
    source_beamlet = await simulate_source(args, templates['source'], y)
    filtered_beamlet = await filter_source(args, templates['filter'], source_beamlet)
    collimated_beamlet = await collimate(args, templates['collimator'], filtered_beamlet)
    # dosing
    return {
        'source': source_beamlet,
        'filter': filtered_beamlet,
        'collimator': collimated_beamlet
    }
