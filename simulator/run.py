import io
import time
import argparse
import os

import pytoml as toml
import logging

import boto3
from botocore.exceptions import ClientError as BotoClientError

from . import simulate


logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


async def main():
    configure_logging()
    args = parse_args()
    local = read_local()
    simulations = read_simulations(args)
    for sim in simulations:
        sim['server'] = local['server']
        if sim['beam-height'] != local['beam-height']:
            logger.warning('Skipping (beam height != {}) {}'.format(
                local['beam-height'], sim['name']))
            continue
        if args.directory and os.path.basename(args.directory) == os.path.basename(sim['directory']):
            await go(sim)
            continue
        if not claim(sim) and not args.force:
            logger.warning('Skipping (claimed) {}'.format(sim['name']))
            continue
        await go(sim)


async def go(sim):
    start = time.time()
    logger.warning('Starting {}'.format(sim['name']))
    try:
        await simulate.run_simulation(sim)
    except Exception as e:
        logger.exception(e)
    else:
        logger.info('Finished in {:.2f} seconds'.format(time.time() - start))
        logger.info('Output in {}'.format(sim['directory']))
        upload_report(sim)


def read_local():
    with open('local.toml') as f:
        local = toml.load(f)
    assert local['server']
    assert local['beam-height'] in [0.2, 1.0]
    return local


def configure_logging():
    formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s', '%H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler('debug.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.DEBUG)
    for name in ['boto3', 'botocore', 'nose', 's3transfer']:
        logging.getLogger(name).setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--default-simulation', default='simulation.defaults.toml')
    parser.add_argument('--simulations', default='simulations.toml')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--directory')
    parser.add_argument('--histories', type=float)
    parser.add_argument('--operations', type=int, default=None)
    return parser.parse_args()


def read_simulations(args):
    with open(args.simulations) as f:
        overrides = toml.load(f)['simulations']
    with open(args.default_simulation) as f:
        defaults = toml.load(f)
    simulations = []
    for override in overrides:
        simulation = defaults.copy()
        simulation.update(override)
        if args.histories:
            simulation['desired-histories'] = args.histories
        simulation['operations'] = args.operations
        simulations.append(simulation)
    return [verify_sim(sim) for sim in simulations]


def claim(simulation):
    """Clearly not thread safe, but it'll do the trick."""
    s3 = boto3.resource('s3')
    key = os.path.join(os.path.basename(simulation['directory']), 'claimed.toml')
    try:
        remote = toml.loads(s3.Object('xcite-simulations', key).get()['Body'].read())
    except BotoClientError:
        body = io.BytesIO(toml.dumps(simulation).encode('utf-8'))
        s3.Object('xcite-simulations', key).put(Body=body)
        return True
    else:
        return simulation['server'] == remote['server']


def upload_report(sim):
    s3 = boto3.client('s3')
    report_path = os.path.join(sim['directory'], 'report.pdf')
    slug = os.path.basename(sim['directory'])
    report_key = os.path.join(slug, slug + '.pdf')
    with open(report_path, 'rb') as f:
        s3.put_object(
            Bucket='xcite-simulations',
            Key=report_key,
            Body=f,
            ACL='public-read'
        )
    url = 'https://s3-us-west-2.amazonaws.com/xcite-simulations/' + report_key
    logger.info('Report uploaded to {}'.format(url))
    for subfolder in ['dose', 'doselets/arc', 'doselets/stationary']:
        for filename in os.listdir(os.path.join(sim['directory'], subfolder)):
            path = os.path.join(sim['directory'], 'dose', filename)
            logger.info('Uploading {}'.format(path))
            dose_key = os.path.join(os.path.basename(sim['directory']), subfolder, os.path.basename(path))
            with open(path, 'rb') as f:
                s3.put_object(
                    Bucket='xcite-simulations',
                    Key=dose_key,
                    Body=f,
                    ACL='public-read'
                )


def verify_sim(sim):
    sim['directory'] = os.path.abspath(sim['name'].replace(' - ', '-').replace(' ', '-'))
    assert 'collimator' in sim, "{} does not defined collimator".format(sim['name'])
    sim['egs-home'] = os.path.abspath(os.path.join(os.environ['HEN_HOUSE'], '../egs_home/'))
    paths = [
        sim['beamnrc-template'], sim['stationary-dose-template'],
        sim['arc-dose-template'], sim['grace'],
        os.path.join(os.environ['HEN_HOUSE'], 'pegs4', 'data', sim['pegs4'] + '.pegs4dat'),
    ]
    for path in paths:
        assert os.path.exists(path), "Could not find {}".format(path)
    floats = [
        'rmax', 'beam-width', 'beam-height', 'beam-gap',
        'target-distance', 'target-length', 'target-angle'
    ]
    ints = [
        'desired-histories', 'dose-recycle', 'dose-photon-splitting'
    ]
    sim['phantom-isocenter'] = list(map(float, sim['phantom-isocenter']))
    for key in floats:
        sim[key] = float(sim[key])
    for key in ints:
        sim[key] = int(sim[key])
    reserved = ['beamlet-count', 'beamlet-histories']
    for key in reserved:
        if key in sim:
            raise KeyError('{} is a reserved keyword'.format(key))
    sim['collimator']['lesion-distance'] = sim['lesion-distance']
    sim['collimator']['lesion-diameter'] = sim['lesion-diameter']
    return sim
