import json
import sys
import argparse
import time

import yaml

from export import cgs
from egsinp import parse_egsinp, unparse_egsinp, verify, ParseError

import logging
logger = logging.getLogger(__name__)


def get_type(fname, valid):
    logger.debug('Getting type for %s', fname)
    try:
        _, ext = fname.rsplit('.', 1)
    except ValueError as e:
        raise ValueError('No file extension found: {}'.format(str(e)))
    if ext not in valid:
        raise ValueError('Invalid output type {}'.format(ext))
    return ext


# args
def parse_args():
    logger.debug('Parsing arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='file to read (and watch)')
    parser.add_argument('output', nargs='+', help='output filename (will be overwritten')
    args = parser.parse_args()
    try:
        valid_input = ['yaml', 'json', 'egsinp']
        args.input_type = get_type(args.input, valid_input)
        args.output_types = []
        for fname in args.output:
            valid_output = ['yaml', 'json', 'egsinp', 'scad', 'collada', 'occ']
            args.output_types.append(get_type(fname, valid_output))
    except ValueError as e:
        print(str(e))
        parser.print_help()
        sys.exit(1)
    logger.debug('Arguments are: %s', args)
    return args


def read_file(fname):
    # logger.debug('Reading %s', fname)
    try:
        f = open(fname)
    except OSError as e:
        print("Could not open {}: {}".format(fname, str(e)))
        sys.exit(1)
    return f.read()


def unparse_json(data):
    options = {
        'sort_keys': True,
        'indent': 4,
        'separators': (', ', ': ')
    }
    return json.dumps(data, **options)


def unparse_yaml(data):
    return yaml.dump(data)
    # return yaml.dump(data, default_flow_style=False)


def get_by_string(data, s):
    def integer(tokens):
        return int(tokens[0])
    import pyparsing as pp
    index = pp.Literal('[').suppress() + pp.Word(pp.nums).setParseAction(integer) + pp.Literal(']').suppress()
    key = pp.Word(pp.alphanums)
    dotKey = pp.Literal('.').suppress() + key
    grammar = key + pp.ZeroOrMore(dotKey | index)
    accessors = grammar.parseString(s)
    for accessor in accessors:
        data = data[accessor]
    return accessors, data


def variants(data):
    if 'vary' not in data:
        return [data]
    for key, values in data['vary'].items():
        accessors, data = get_by_string(data, key)
        # name = accessors[-1]
    # NOT DONE


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    parsers = {
        'egsinp': parse_egsinp,
        'json': json.loads,
        'yaml': yaml.load
    }
    parse = parsers[args.input_type]
    unparsers = {
        'egsinp': unparse_egsinp,
        'json': unparse_json,
        'yaml': unparse_yaml,
        'scad': cgs
    }
    previous_result = None
    error_last_time = False
    while True:
        text = read_file(args.input)
        try:
            result = parse(text)
        except ParseError as e:
            if not error_last_time:
                print(str(e))
            error_last_time = True
        else:
            error_last_time = False
            if result != previous_result:
                result = verify(result)
                for out_fname, out_type in zip(args.output, args.output_types):
                    unparse = unparsers[out_type]
                    logger.info('Writing to %s', out_fname)
                    open(out_fname, 'w').write(unparse(result))
        time.sleep(1)
        previous_result = result
