import re
import os

from jinja2 import Environment, FileSystemLoader, Markup

from . import latexmake

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


def format_float(f):
    return '{:.1f}'.format(f)


def percent(f):
    return '{}\%'.format(int(f * 100))


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
    env.filters['f'] = format_float
    env.filters['percent'] = percent
    return env


def generate_summary(summary):
    report = get_env().get_template('templates/summary.tex').render(summary)
    os.makedirs('summary', exist_ok=True)
    slug = 'Precisionrt-RT-Simulation-Summary_Henry-Baxter_University-of-Victoria'
    with open('summary/{}.tex'.format(slug), 'w') as f:
        f.write(report)
    os.chdir('summary')
    latex_args, rest = latexmake.arg_parser().parse_known_args()
    latexmake.LatexMaker(slug, latex_args).run()
    os.chdir('..')


def generate(sim, context):
    report = get_env().get_template('templates/report.tex').render(context)
    with open(os.path.join(sim['directory'], 'report.tex'), 'w') as f:
        f.write(report)
    os.chdir(sim['directory'])
    latex_args, rest = latexmake.arg_parser().parse_known_args()
    latexmake.LatexMaker('report', latex_args).run()
    os.chdir('..')
