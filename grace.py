import logging
from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)

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


def generate(arguments, output_path):
    logger.info('Generating grace plot:\n{}'.format('\n'.join(arguments)))
    p = Popen(['beamdp'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    (stdout, stderr) = p.communicate("\n".join(arguments).encode('utf-8'))
    if p.returncode != 0:
        print('Could not generate grace with arguments:')
        print(arguments)
        print(output_path)
        print()
        print(stdout.decode('utf-8'))
        print(stderr.decode('utf-8'))
    result = []
    for line in open(output_path):
        if line.startswith('@    legend') or line.startswith('@    subtitle'):
            continue
        if line.startswith('@    s0 symbol color'):
            result.append(line.strip())
            result.append('@    s0 symbol size 0.040000')
        else:
            result.append(line.strip())
    open(output_path, 'w').write("\n".join(result))


def view(path):
    Popen(['xmgrace', path])


def eps(path):
    output = path.replace('.grace', '.eps')
    Popen(['xmgrace', '-hardcopy', '-nosafe', '-printfile', output, path])


def xy(input_path, output_path, **kwargs):
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
    arguments = [
        "y",  # show more detailed information
        "9",  # type of processing (9 = scatter plot)
        "{}, {}".format(args['charge'], extents),  # charge, (xmin, xmax, ymin, ymax | rmin, rmax)
        "{}, {}".format(args['min_ke'], args['max_ke']),  # emin, emax in MeV
        "",  # I_IN_EX, Nbit1, Nbit2
        input_path,  # egsphsp input
        output_path,  # grace output
        str(args['max_particles']),  # maximum number of particles to output (defaults to all apparently)
        "",  # create another?
        "",  # grace?
        ""  # eof
    ]
    generate(arguments, output_path)
    return args


def angular(input_path, output_path, **kwargs):
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
    arguments = [
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
    generate(arguments, output_path)
    return args


def energy_fluence_vs_position(input_path, output_path, **kwargs):
    args = {
        'field_shape': 'rectangular',
        'bins': 200,
        'axis': 'y',
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
    arguments = [
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
    generate(arguments, output_path)
    return args


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
    arguments = [
        "y",  # show more detailed information
        "3",  # type of processing (3 = spectral distribution)
        "{}, {}".format(args['charge'], extents),  # iq (-1 = electron, 0 = photon, 1 = positron, ...), xmin, xmax, ymin, ymax (or rmin, rmax)
        "{}, {}, {}".format(args['bins'], args['min_ke'], args['max_ke']),  # nbins < 200, emin, emax
        "",  # I_IN_EX, Nbit1, Nbit2
        input_path,  # input file
        output_path,   # place to save grace file
        str(graph_types[args['graph_type']]),  # 0=normal 1=histogram
        str(fluence_types[args['fluence_type']]),  # 1 = planar fluence, 0 = estimate of real
        "",  # create another?
        "",  # grace?
        ""  # eof
    ]
    generate(arguments, output_path)
    return args

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps')
    parser.add_argument('--xy')
    parser.add_argument('--fluencey')
    parser.add_argument('--xmin', type=float, default=-100.0)
    parser.add_argument('--ymin', type=float, default=-100.0)
    parser.add_argument('--xmax', type=float, default=100.0)
    parser.add_argument('--ymax', type=float, default=100.0)
    args = parser.parse_args()
    extents = {
        'xmin': args.xmin,
        'xmax': args.xmax,
        'ymin': args.ymin,
        'ymax': args.ymax
    }
    import os
    if args.eps:
        for path in os.listdir(args.eps):
            if path.endswith('.grace'):
                path = os.path.join(args.eps, path)
                print('generating eps for {}'.format(path))
                eps(path)
    elif args.xy:
        output = args.xy.replace('.egsphsp1', '.grace')
        xy(args.xy, output, extents=extents)
        view(output)
    elif args.fluencey:
        output = args.fluencey.replace('.egsphsp1', '.fluencey.grace')
        energy_fluence_vs_position(args.fluencey, output, axis='y', extents=extents)
        view(output)
