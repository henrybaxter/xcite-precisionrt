import re
import os

from jinja2 import Environment, FileSystemLoader, Markup

import latexmake

LATEX_SUBS = (
    (re.compile(r'\\'), r'\\textbackslash'),
    (re.compile(r'([{}_#%&$])'), r'\\\1'),
    (re.compile(r'~'), r'\~{}'),
    (re.compile(r'\^'), r'\^{}'),
    (re.compile(r'"'), r"''"),
    (re.compile(r'\.\.\.+'), r'\\ldots'),
)


def escape_tex(value):
    newval = value
    for pattern, replacement in LATEX_SUBS:
        newval = pattern.sub(replacement, newval)
    return newval


def translate_medium(medium):
    return Markup({
        'Air_516kV': 'Air',
        'H2O_516kV': '\ce{H2O}',
        'VACUUM': 'Vacuum',
        'steel304L_521kV': 'Steel (304)',
        'Al_516kV': 'Aluminum'
    }[medium])


def strip_extension(path):
    return os.path.splitext(path)[0]


def get_env():
    env = Environment(loader=FileSystemLoader('.'))
    env.block_start_string = '((*'
    env.block_end_string = '*))'
    env.variable_start_string = '((('
    env.variable_end_string = ')))'
    env.comment_start_string = '((='
    env.comment_end_string = '=))'
    env.filters['escape'] = escape_tex
    env.filters['medium'] = translate_medium
    env.filters['strip_extension'] = strip_extension
    return env


"""



def itemize_photons(beamlets):
    lines = []
    previous = args.simulation_properties['source']['histories']
    lines.append('\t\item {}: {}'.format('Incident electrons', str(previous)))
    for stage in ['source', 'filter', 'collimator']:
        photons = sum([beamlet['stats']['total_photons'] for beamlet in beamlets[stage]])
        if previous:
            rate = previous / photons
            text = '{} photons (reduced by a factor of {:.2f})'.format(photons, rate)
        else:
            text = '{} photons'.format(photons)
        lines.append('\t\item {}: {}'.format(stage.capitalize(), text))
        previous = photons
    overall = args.simulation_properties['source']['histories'] / previous
    lines.append('\t\item Efficiency: {:.2f} electrons generates one photon'.format(overall))
    return '\n'.join(lines)

"""


def generate(data, args):
    context = data.copy()
    context['args'] = args
    report = get_env().get_template('template.tex').render(context)
    open(os.path.join(args.output_dir, 'report.tex'), 'w').write(report)
    os.chdir(args.output_dir)
    latex_args, rest = latexmake.arg_parser().parse_known_args()
    latexmake.LatexMaker('report', latex_args).run()
    os.chdir('..')


