import asyncio
import hashlib
import logging
import os
import platform
import json
from subprocess import Popen
from collections import OrderedDict

import pytoml as toml

from .utils import run_command

logger = logging.getLogger(__name__)

GRACE = 'grace'
if platform.system() == 'Darwin':
    GRACE = 'xmgrace'

"""
BEAMDP = {
    'beam_characterization': 0,
    'fluence_vs_position': 1,
    'energy_fluence_vs_position': 2,
    'spectral_distribution': 3,
    'energy_fluence_distribution': 4,
    'mean_energy_distribution': 5,
    'angular_distribution': 6,
    'zlast_distribution': 7,
    'particle_weight_distribution': 8,
    'xy_scatter_plot': 9,
    'combine': 10,
    'list_parameters': 11,
    'output_xyz': 12
}

GRACE = ['fluence_vs_position', 'energy_fluence_vs_position', 'spectral_distribution',
         'mean_energy_distribution', 'angular_distribution', 'zlast_distribution',
         'particle_weight_distribution', 'xy_scatter_plot']
"""

"""
Grace operates on a toml file that describes all of the options.
It can run totally independently of the report. It needs the phase space
files, and json configuration, and then it's all set.

"""


async def make_plots(phsp_paths, plots):
    # each config is a dictionary with one element, plots
    # plots is a list of plots, we just merge them together
    os.makedirs('grace', exist_ok=True)
    plots = await asyncio.gather(*[
        make_plot(plot, phsp_paths[plot['phsp']]) for plot in plots
    ])
    # they are in order, now group them by plot['type']
    grouped = OrderedDict()
    for plot in plots:
        grouped.setdefault(plot['type'], []).append(plot)
    return grouped


async def make_plot(plot, phsp):
    # phase space, plot options
    base = hashlib.md5((json.dumps(plot) + phsp).encode('utf-8')).hexdigest()
    # we need to generate a grace file and then an eps file, and we want to keep both
    # unless the eps file exists, we assume something went wrong
    grace_path = os.path.join('grace', base + '.grace')
    eps_path = os.path.join('grace', base + '.eps')
    if plot['type'] == 'scatter':
        plotter = scatter
    elif plot['type'] == 'energy_fluence':
        plotter = energy_fluence_vs_position
    elif plot['type'] == 'spectral':
        plotter = spectral_distribution
    elif plot['type'] == 'angular':
        plotter = angular_distribution
    else:
        raise ValueError('Unknown plot type {}'.format(plot['type']))
    logger.info("Processing {}".format(plot['slug']))
    temp_path = grace_path + '.temp'
    plot, lines = plotter(phsp, temp_path, **plot)
    if not os.path.exists(grace_path):
        await generate(lines, temp_path)
        await run_command([GRACE, '-hardcopy', '-nosafe', '-printfile', eps_path, temp_path])
        os.rename(temp_path, grace_path)
    plot['path'] = grace_path
    return plot


async def generate(arguments, output_path):
    # logger.info('Generating grace plot:\n{}'.format('\n'.join(arguments)))
    await run_command(['beamdp'], stdin="\n".join(arguments).encode('utf-8'))
    result = []
    to_delete = ['legend', 'subtitle']
    prefix = '@    '
    for line in open(output_path):
        if any(line.startswith(prefix + key) for key in to_delete):
            continue
        result.append(line.strip())
        if line.startswith(prefix + 's0 symbol color'):
            result.append(prefix + 's0 symbol size 0.040000')
    with open(output_path, 'w') as f:
        f.write("\n".join(result))


def view(path):
    Popen([GRACE, path])


def scatter(input_path, output_path, **kwargs):
    args = {
        'charge': 0,
        'field_shape': 'rectangular',
        'extents': {
            'xmin': -1,
            'xmax': 1,
            'ymin': -35,
            'ymax': 35
        },
        'min_ke': 0,
        'max_ke': 100,
        'max_particles': 10000
    }
    args.update(kwargs)
    field_shape_requirements = {
        'rectangular': ['xmin', 'xmax', 'ymin', 'ymax'],
        'annular': ['rmin', 'rmax']
    }
    field_shape_requirement = field_shape_requirements[args['field_shape']]
    extents = ', '.join(str(args['extents'][key]) for key in field_shape_requirement)
    lines = [
        "y",  # show more detailed information
        "9",  # type of processing (9 = scatter plot)
        "{}, {}".format(args['charge'], extents),  # charge, (xmin, xmax, ymin, ymax | rmin, rmax)
        "{}, {}".format(args['min_ke'], args['max_ke']),  # emin, emax in MeV
        "",  # I_IN_EX, Nbit1, Nbit2
        input_path,  # egsphsp input
        output_path,  # grace output
        # maximum number of particles to output (defaults to all apparently)
        str(args['max_particles']),
        "",  # create another?
        "",  # grace?
        ""  # eof
    ]
    return (args, lines)


def angular_distribution(input_path, output_path, **kwargs):
    args = {
        'charge': 0,
        'field_shape': 'rectangular',
        'extents': {
            'xmin': -100,
            'xmax': 100,
            'ymin': -100,
            'ymax': 100
        },
        'bins': 1000,
        'min_angle': 0,
        'max_angle': 180,
        'min_ke': 0,
        'max_ke': 100,
        'graph_type': 'normal',
        'grouping': 'angular_bin'
    }
    args.update(kwargs)
    field_shape_requirements = {
        'rectangular': ['xmin', 'xmax', 'ymin', 'ymax'],
        'annular': ['rmin', 'rmax']
    }
    graph_types = {
        'normal': 0,
        'histogram': 1
    }
    groupings = {
        'angular_bin': 0,
        'solid_angle': 1
    }
    field_shape_requirement = field_shape_requirements[args['field_shape']]
    assert args['bins'] <= 2000
    assert 0 <= args['min_angle'] < args['max_angle'] <= 180
    extents = ', '.join(str(args['extents'][key]) for key in field_shape_requirement)
    lines = [
        'y',
        '6',
        '{}, {}'.format(args['charge'], extents),
        '{}, {}, {}'.format(args['bins'], args['min_angle'], args['max_angle']),
        '{}, {}'.format(args['min_ke'], args['max_ke']),
        '',  # latch etc
        input_path,
        output_path,
        str(graph_types[args['graph_type']]),  # 0-normal, 1-histogram
        str(groupings[args['grouping']]),  # 0-angular bin, 1-solid angle
        '0',  # same phsp?
        '0',  # plot?
        '',
    ]
    return args, lines


def energy_fluence_vs_position(input_path, output_path, **kwargs):
    args = {
        'field_shape': 'rectangular',
        'bins': 200,
        'processing_type': 'energy_fluence',
        'charge': 0,
        'extents': {
            'xmin': -100,
            'xmax': 100,
            'ymin': -100,
            'ymax': 100
        },
        'graph_type': 'normal',
        'fluence_type': 'estimate_real'
    }
    args.update(kwargs)
    axes = {
        'x': 0,
        'y': 1
    }
    processing_types = {
        'fluence': 1,
        'energy_fluence': 2
    }
    field_shapes = {
        'circular': 0,
        'square': 1,
        'rectangular': 2
    }
    field_shape_requirements = {
        'circular': 'radius',
        'square': 'half_width',
        'rectangular': ['xmin', 'xmax', 'ymin', 'ymax']
    }
    graph_types = {
        'normal': 0,
        'histogram': 1
    }
    fluence_types = {
        'estimate_real': 0,
        'planar': 1
    }
    assert args['bins'] <= 2000
    field_shape_requirement = field_shape_requirements[args['field_shape']]
    extents = ", ".join('{:.4f}'.format(args['extents'][key]) for key in field_shape_requirement)
    lines = [
        "y",  # show more detailed information
        str(processing_types[args['processing_type']]),
        str(field_shapes[args['field_shape']]),
        "{}, {}, {}, {}".format(args['bins'], axes[args['axis']], args['charge'], extents),
        "",  # I_IN_EX, Nbit1, Nbit2
        input_path,  # input file
        output_path,   # place to save grace file
        str(graph_types[args['graph_type']]),
        str(fluence_types[args['fluence_type']]),
        "",  # create another?
        "",  # open grace?
        ""  # eof
    ]
    return args, lines


def spectral_distribution(input_path, output_path, **kwargs):
    args = {
        'charge': 0,
        'field_shape': 'rectangular',
        'extents': {
            'xmin': -100,
            'xmax': 100,
            'ymin': -100,
            'ymax': 100
        },
        'bins': 1000,
        'min_ke': 0,
        'max_ke': .5,
        'graph_type': 'normal',
        'fluence_type': 'estimate_real'
    }
    args.update(kwargs)
    field_shape_requirements = {
        'rectangular': ['xmin', 'xmax', 'ymin', 'ymax'],
        'annular': ['rmin', 'rmax']
    }
    graph_types = {
        'normal': 0,
        'histogram': 1
    }
    fluence_types = {
        'estimate_real': 0,
        'planar': 1
    }
    assert args['bins'] <= 2000
    field_shape_requirement = field_shape_requirements[args['field_shape']]
    extents = ", ".join(str(args['extents'][key]) for key in field_shape_requirement)
    lines = [
        "y",  # show more detailed information
        "3",  # type of processing (3 = spectral distribution)
        # iq (-1 = electron, 0 = photon, 1 = positron, ...), xmin, xmax, ymin, ymax (or rmin, rmax)
        "{}, {}".format(args['charge'], extents),
        # nbins < 200, emin, emax
        "{}, {}, {}".format(args['bins'], args['min_ke'], args['max_ke']),
        "",  # I_IN_EX, Nbit1, Nbit2
        input_path,  # input file
        output_path,   # place to save grace file
        str(graph_types[args['graph_type']]),  # 0=normal 1=histogram
        str(fluence_types[args['fluence_type']]),  # 1 = planar fluence, 0 = estimate of real
        "",  # create another?
        "",  # grace?
        ""  # eof
    ]
    return args, lines


def eps_convert(directory):
    loop = asyncio.get_event_loop()
    operations = []
    for path in os.listdir(directory):
        if not path.endswith('.grace'):
            continue
        path = os.path.join(args.eps, path)
        print('generating eps for {}'.format(path))
        output = path.replace('.grace', '.eps')
        operations.append(run_command([GRACE, '-hardcopy', '-nosafe', '-printfile', path, output]))
    loop.run_until_complete(asyncio.gather(*operations))
    loop.close()

async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('grace')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--eps')
    parser.add_argument('--config', default='grace.toml')
    args = parser.parse_args()
    with open(args.config) as f:
        config = toml.load(f)['plots']
    if args.eps:
        eps_convert(args.eps)
    else:
        phsp_paths = {
            'source': os.path.join(args.input, 'sampled_source.egsphsp1'),
            'filter': os.path.join(args.input, 'sampled_filter.egsphsp1'),
            'collimator': os.path.join(args.input, 'sampled_collimator.egsphsp1')
        }
        plots = await make_plots(phsp_paths, config)
        print(plots)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

