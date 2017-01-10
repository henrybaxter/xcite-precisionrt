"""
Builds the egsinp for the source (reflection target), filter, and collimator,
and compiles the appropriate modules.
"""
import sys
import logging
import os
import time
import platform

from . import egsinp
from .utils import run_command


logger = logging.getLogger(__name__)


def write_specmodule(egs_home, name, cms):
    logger.info('Writing spec module for {}'.format(name))
    types = []
    identifiers = []
    for cm in cms:
        types.append(cm['type'])
        identifiers.append(cm['identifier'])
    types_line = ' CM names:  {}'.format(' '.join(types))
    identifiers_line = ' Identifiers:  {}'.format(' '.join(identifiers))
    path = os.path.join(egs_home,
                        'beamnrc/spec_modules',
                        '{}.module'.format(name))
    with open(path, 'w') as f:
        f.write('\n'.join([types_line, identifiers_line]))
    logger.info('Spec module written')


def get_egsinp(path):
    logger.info('Reading template {}'.format(path))
    try:
        with open(path) as f:
            text = f.read()
    except IOError:
        logger.error('Could not open template {}'.format(path))
        sys.exit(1)
    try:
        template = egsinp.parse_egsinp(text)
    except egsinp.ParseError as e:
        logger.error('Could not parse template {}: {}'.format(path, e))
        sys.exit(1)
    return template


async def build_filter(args):
    logger.info('Building filter')
    template = get_egsinp(args.egsinp_template)
    template['cms'] = [
        {
            'type': 'SLABS',
            'identifier': 'FLTR',
            'rmax_cm': args.rmax,
            'title': 'FLTR',
            'zmin_slabs': 0.01,
            'slabs': [
                {
                    'zthick': 0.1,
                    'ecut': 0.521,
                    'pcut': 0.001,
                    'dose_zone': 0,
                    'iregion_to_bit': 0,
                    'esavein': 0,
                    'medium': 'Al_516kV'
                },
                {
                    'zthick': 0.3,
                    'ecut': 0.521,
                    'pcut': 0.001,
                    'dose_zone': 0,
                    'iregion_to_bit': 0,
                    'esavein': 0,
                    'medium': 'H2O_516kV'
                },
                {
                    'zthick': 0.05,
                    'ecut': 0.521,
                    'pcut': 0.001,
                    'dose_zone': 0,
                    'iregion_to_bit': 0,
                    'esavein': 0,
                    'medium': 'steel304L_521kV'
                },

            ]
        }
    ]
    template['isourc'] = '21'
    template['iqin'] = '0'
    for k in ['nrcycl', 'iparallel', 'parnum', 'isrc_dbs', 'rsrc_dbs', 'ssdrc_dbs', 'zsrc_dbs']:
        template[k] = '0'
    template['init_icm'] = 1
    name = 'FILTR'
    template['title'] = name
    folder = os.path.join(args.egs_home, 'BEAM_{}'.format(name))
    if not os.path.exists(folder):
        await beam_build(args.egs_home, name, template['cms'])
    return template


async def build_source(args, histories):
    logger.info('Building source')
    template = get_egsinp(args.egsinp_template)
    template['ncase'] = histories
    template['ybeam'] = args.beam_width / 2
    template['zbeam'] = args.beam_height / 2
    xtube = template['cms'][0]
    xtube['rmax_cm'] = args.rmax
    xtube['anglei'] = args.target_angle
    name = 'RFLCT'
    template['title'] = name
    folder = os.path.join(args.egs_home, 'BEAM_{}'.format(name))
    if not os.path.exists(folder):
        await beam_build(args.egs_home, name, template['cms'])
    return template


async def beam_build(egs_home, name, cms):
    logger.info('Building {} with cms {}'.format(name, ", ".join([cm['type'] for cm in cms])))
    start = time.time()
    write_specmodule(egs_home, name, cms)
    await run_command(['beam_build.exe', name])
    await run_command(['make'], cwd=os.path.join(egs_home, 'BEAM_{}'.format(name)))
    if platform.system() == 'Darwin':
        await run_command([
            'install_name_tool',
            '-change',
            '../egs++/dso/osx/libiaea_phsp.dylib',
            os.path.expanduser('~/projects/EGSnrc/HEN_HOUSE/egs++/dso/osx/libiaea_phsp.dylib'),
            os.path.expanduser('~/projects/EGSnrc/egs_home/bin/osx/BEAM_{}'.format(name))]
        )
    elapsed = time.time() - start
    logger.info('{} built in {} seconds'.format(name, elapsed))


async def build_collimator(args):
    logger.info('Building collimator')
    template = get_egsinp(args.egsinp_template)
    template['cms'] = []
    collimator = get_egsinp(args.collimator)
    template['cms'] = [cm for cm in collimator['cms'] if cm['type'] == 'BLOCK']
    if not template['cms']:
        raise ValueError('No BLOCK CMs found in collimator')
    # for collimators that are part of a larger egsinp simulation file
    zoffset = template['cms'][0]['zmin']
    if not args.target_distance:
        # target distance is measured from the end of the collimator
        args.target_distance = template['cms'][0]['zfocus'] - zoffset - template['cms'][-1]['zmax']
        logger.info('Inferring target distance of {} cm'.format(args.target_distance))
    for block in template['cms']:
        block['zmin'] -= zoffset
        block['zmax'] -= zoffset
        block['zfocus'] -= zoffset
        block['rmax_cm'] = args.rmax
    template['isourc'] = '21'
    template['iqin'] = '0'
    template['default_medium'] = 'Air_516kV'
    template['nsc_planes'] = '1'
    template['init_icm'] = 1
    template['nrcycl'] = 0
    template['iparallel'] = 0
    template['parnum'] = 0
    template['isrc_dbs'] = 0
    template['rsrc_dbs'] = 0
    template['ssdrc_dbs'] = 0
    template['zsrc_dbs'] = 0
    template['scoring_planes'] = [{
        'cm': len(template['cms']),  # the LAST block of the collimator
        'mzone_type': 1,
        'nsc_zones': 1,
        'zones': tuple([args.rmax])
    }]
    name = 'CLMT{}'.format(len(template['cms']))
    template['title'] = name
    folder = os.path.join(args.egs_home, 'BEAM_{}'.format(name))
    if not os.path.exists(folder):
        await beam_build(args.egs_home, name, template['cms'])
    return template
