import logging
import re
from itertools import zip_longest

logger = logging.getLogger(__name__)


class ParseError(Exception):
    pass


def text(line):
    """Removes comments"""
    return line.split('#')[0].strip()


def values(line):
    """Splits comma separated fields, missing fields become None"""
    return [v.strip() or None for v in text(line).split(',')]


class LineIterator(object):
    """Iterator that counts lines seen, and allows peeking ahead"""
    def __init__(self, lines):
        self.lines = lines
        self.line_number = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.line_number += 1
        return self.lines[self.line_number - 1]

    def peek(self):
        return self.lines[self.line_number]


def validate(line_number, position, identifier, validator, value):
    try:
        return validator.validate(value)
    except (TypeError, ValueError) as e:
        print('Could not read `{}` in position {} on line {}:'.format(identifier, position, line_number))
        print('\tFound {}, but {} {}'.format(value, identifier, str(e)))
        raise ParseError()


def pick(lines, fields, peek=False):
    if peek:
        line = lines.peek()
        line_number = lines.line_number + 1
    else:
        line = next(lines)
        line_number = lines.line_number
    result = []
    for i, ((keyword, validator), value) in enumerate(zip_longest(fields, values(line)[:len(fields)])):
        value = validate(line_number, i + 1, keyword, validator, value)
        result.append((keyword, value))
    return result


def pickone(lines, identifier, validator):
    return pick(lines, [(identifier, validator)])


def pickvalue(lines, identifier, validator):
    return pickone(lines, identifier, validator)[0][1]


def pickcounted(lines, id1, id2, validator2):
    line = next(lines)
    count, *rest = values(line)
    count = validate(lines.line_number, 1, id1, NonNegativeInteger(), count)
    rest = rest[:count]
    result = []
    for position, value in enumerate(rest, 2):
        value = validate(lines.line_number, position, id2, validator2, value)
        result.append(value)
    return result


class Boolean(object):
    def validate(self, token):
        token = token.strip().lower()
        if token.isdigit():
            return bool(int(token))
        elif token in ['on', 'true', 'yes']:
            return True
        elif token in ['off', 'false', 'no']:
            return False
        else:
            raise ValueError('must be boolean indicator')


class Any(object):
    def validate(self, token):
        return token


class Word(object):
    def validate(self, token):
        if re.match('\w+', token) is None:
            raise ValueError('must be a word')
        return token


class Medium(Word):
    pass


class Words(object):
    def __init__(self, values):
        assert values
        self.values = values

    def validate(self, token):
        if token not in self.values:
            raise ValueError('must be one of {}'.format(', '.join(self.values)))
        return token


class Integers(object):
    def __init__(self, values):
        assert values
        self.values = values

    def validate(self, token):
        value = int(token or 0)
        if value not in self.values:
            raise ValueError('must be one of {}'.format(', '.join(map(str, self.values))))
        return value


class NonNegativeInteger(object):
    def validate(self, token):
        value = int(token or 0)
        if value < 0:
            raise ValueError('must be non-negative')
        return value


class NonNegativeFloat(object):
    def validate(self, token):
        value = float(token or 0)
        if value < 0:
            raise ValueError('must be non-negative')
        return value


class Float(object):
    def validate(self, token):
        value = float(token or 0)
        return value


class PositiveFloat(object):
    def validate(self, token):
        value = float(token or 0)
        if value <= 0:
            raise ValueError('must be positive')
        return value


class Integer(object):
    def validate(self, token):
        value = int(token or 0)
        return value


class PositiveInteger(object):
    def validate(self, token):
        value = int(token or 0)
        if value <= 0:
            raise ValueError('must be positive')
        return value


def commalist(d, l):
    def stringify(v):
        if isinstance(v, float):
            return '{:.5f}'.format(v)
        return str(v)
    return ', '.join([stringify(d[k]) for k in l])


def unparse_xtube(d):
    lines = [
        commalist(d, ['rmax_cm']),
        commalist(d, ['title']),
        commalist(d, ('zmin', 'zthick')) + ', ZMIN, ZTHICK',
        commalist(d, ['anglei']) + ', ANGLE',
        str(len(d['layers'])) + ', # LAYERS'
    ]
    for layer in d['layers']:
        lines.extend([
            commalist(layer, ('dthick', 'extra')),
            commalist(layer, ('ecut', 'pcut', 'dose_zone', 'iregion_to_bit')) + ', ',
            layer['medium']
        ])
    lines.extend([
        commalist(d['front'], ('ecut', 'pcut', 'dose_zone', 'iregion_to_bit')) + ', ',
        d['front']['medium'],
        commalist(d['holder'], ('ecut', 'pcut', 'dose_zone', 'iregion_to_bit')) + ', ',
        d['holder']['medium'],
    ])
    return lines


def parse_xtube(lines):
    d = {}
    d.update(pickone(lines, 'rmax_cm', PositiveFloat()))
    d.update(pickone(lines, 'title', Word()))
    d.update(pick(lines, [
        ('zmin', NonNegativeFloat()),
        ('zthick', NonNegativeFloat())
    ]))
    d.update(pickone(lines, 'anglei', Float()))
    n_layers = pickvalue(lines, 'n_layers', NonNegativeInteger())
    d['layers'] = []
    for i in range(n_layers):
        layer = dict(pick(lines, [
            ('dthick', NonNegativeFloat()),
            ('extra', Integers([0, 1]))
        ]))
        if layer['extra']:
            raise NotImplementedError('The extra is not supported yet')
        layer.update(pick(lines, [
            ('ecut', NonNegativeFloat()),
            ('pcut', NonNegativeFloat()),
            ('dose_zone', NonNegativeInteger()),
            ('iregion_to_bit', NonNegativeInteger())
        ]))
        layer.update(pickone(lines, 'medium', Medium()))
        d['layers'].append(layer)
    pattern = [
        ('ecut', NonNegativeFloat()),
        ('pcut', NonNegativeFloat()),
        ('dose_zone', NonNegativeInteger()),
        ('iregion_to_bit', NonNegativeInteger())
    ]
    d['front'] = dict(pick(lines, pattern))
    d['front'].update(pickone(lines, 'medium', Medium()))
    d['holder'] = dict(pick(lines, pattern))
    d['holder'].update(pickone(lines, 'medium', Medium()))
    return d


def unparse_block(d):
    lines = [
        commalist(d, ['rmax_cm']),
        commalist(d, ['title']),
        commalist(d, ('zmin', 'zmax', 'zfocus')) + ', # ZMIN, ZMAX, ZFOCUS',
        str(len(d['regions']))
    ]
    for region in d['regions']:
        lines.append(len(region['points']))
        for point in region['points']:
            # if these are too long BEAM starts losing digits off the *front*
            lines.append('{:.5f}, {:.5f}'.format(point['x'], point['y']))
    lines.extend([
        commalist(d, ('xpmax', 'ypmax', 'xnmax', 'ynmax')),
        commalist(d['air_gap'], ('ecut', 'pcut', 'dose_zone', 'iregion_to_bit')),
        commalist(d['opening'], ('ecut', 'pcut', 'dose_zone', 'iregion_to_bit')),
        d['opening']['medium'],
        commalist(d['block'], ('ecut', 'pcut', 'dose_zone', 'iregion_to_bit')),
        d['block']['medium']
    ])
    return lines


def parse_block(lines):
    d = {}
    d.update(pickone(lines, 'rmax_cm', PositiveFloat()))
    d.update(pickone(lines, 'title', Word()))
    d.update(pick(lines, [
        ('zmin', NonNegativeFloat()),
        ('zmax', NonNegativeFloat()),
        ('zfocus', Float())
    ]))
    n_regions = pickvalue(lines, 'n_regions', NonNegativeInteger())
    d['regions'] = []
    for i in range(n_regions):
        n_points = pickvalue(lines, 'n_points', NonNegativeInteger())
        region = {
            'points': []
        }
        for j in range(n_points):
            region['points'].append(dict(pick(lines, [
                ('x', Float()),
                ('y', Float())
            ])))
        d['regions'].append(region)
    d.update(pick(lines, [
        ('xpmax', Float()),
        ('ypmax', Float()),
        ('xnmax', Float()),
        ('ynmax', Float())
    ]))
    pattern = [
        ('ecut', NonNegativeFloat()),
        ('pcut', NonNegativeFloat()),
        ('dose_zone', NonNegativeInteger()),
        ('iregion_to_bit', NonNegativeInteger())
    ]
    d['air_gap'] = dict(pick(lines, pattern))
    d['opening'] = dict(pick(lines, pattern))
    d['opening']['medium'] = pickvalue(lines, 'medium', Medium())
    d['block'] = dict(pick(lines, pattern))
    d['block']['medium'] = pickvalue(lines, 'medium', Medium())
    return d


def unparse_slabs(d):
    lines = [
        commalist(d, ['rmax_cm']),
        commalist(d, ['title']),
        str(len(d['slabs'])),
        commalist(d, ['zmin_slabs']),
    ]
    for slab in d['slabs']:
        lines.append(commalist(slab, ('zthick', 'ecut', 'pcut', 'dose_zone', 'iregion_to_bit', 'esavein')))
        lines.append(slab['medium'])
    return lines


def parse_slabs(lines):
    d = {}
    d.update(pickone(lines, 'rmax_cm', PositiveFloat()))
    d.update(pickone(lines, 'title', Word()))
    n_slabs = pickvalue(lines, 'n_slabs', PositiveInteger())
    d.update(pickone(lines, 'zmin_slabs', NonNegativeFloat()))
    d['slabs'] = []
    for i in range(n_slabs):
        slab = dict(pick(lines, [
            ('zthick', NonNegativeFloat()),
            ('ecut', NonNegativeFloat()),
            ('pcut', NonNegativeFloat()),
            ('dose_zone', NonNegativeInteger()),
            ('iregion_to_bit', NonNegativeInteger()),
            ('esavein', NonNegativeInteger())
        ]))
        slab['medium'] = pickvalue(lines, 'medium', Medium())
        d['slabs'].append(slab)
    return d

mc_transport_parameters = [
    ('global_ecut', 'Global ECUT', NonNegativeFloat()),
    ('global_pcut', 'Global PCUT', NonNegativeFloat()),
    ('global_smax', 'Global SMAX', NonNegativeInteger()),
    ('estepe', 'ESTEPE', NonNegativeFloat()),
    ('ximax', 'XIMAX', NonNegativeFloat()),
    ('boundary_crossing_algorithm', 'Boundary crossing algorithm', Words(['EXACT'])),
    ('skin_depth_for_bca', 'Skin depth for BCA', NonNegativeInteger()),
    ('electron_step', 'Electron-step algorithm', Words(['PRESTA-II'])),
    ('spin_effects', 'Spin effects', Boolean()),
    ('brems_angular_sampling', 'Brems angular sampling', Words(['Simple'])),
    ('brems_cross_sections', 'Brems cross sections', Words(['BH'])),
    ('bound_compton_scattering', 'Bound Compton scattering', Boolean()),
    ('compton_cross_sections', 'Compton cross sections', Words(['default'])),
    ('pair_angular_sampling', 'Pair angular sampling', Words(['Simple'])),
    ('pair_cross_sections', 'Pair cross sections', Words(['BH'])),
    ('photoelectron_angular_sampling', 'Photoelectron angular sampling', Boolean()),
    ('rayleigh_scattering', 'Rayleigh scattering', Boolean()),
    ('atomic_relaxations', 'Atomic relaxations', Boolean()),
    ('electron_impact_ionization', 'Electron impact ionization', Boolean()),
    ('photon_cross_sections', 'Photon cross sections', Words(['xcom'])),
    ('photon_cross_sections_output', 'Photon cross-sections output', Boolean())
]


def unparse_mc_transport(mc_transport):
    lines = []
    lines.append(' :Start MC Transport Parameter:')
    lines.append('')
    for key, description, validator in mc_transport_parameters:
        value = mc_transport[key]
        if isinstance(validator, Boolean):
            if value:
                value = 'On'
            else:
                value = 'Off'
        if value is None:
            value = ''
        lines.append(' {}= {}'.format(description, value))
    lines.append('')
    lines.append(' :Stop MC Transport Parameter:')
    lines.append(' #########################')
    return lines


def parse_mc_transport(lines):
    results = []
    for line in lines:
        line = line.strip()
        if line.startswith(':Stop MC Transport Parameter:'):
            break
        for key, description, validator in mc_transport_parameters:
            if line.lower().startswith(description.lower()):
                value = line.split('=')[1].strip()
                value = validate(lines.line_number, 1, description, validator, value)
                results.append((key, value))
    return results


bcse_parameters = [
    ('use_bcse', 'Use BCSE', Boolean()),
    ('media_to_enhance', 'Media to enhance', Any()),
    ('enhancement_constant', 'Enhancement constant', Any()),
    ('enhancement_power', 'Enhancement power', Any()),
    ('russian_roulette', 'Russian Roulette', Boolean())
]


def parse_bcse(lines):
    results = []
    for line in lines:
        line = line.strip()
        if line.startswith(':Stop BCSE:'):
            break
        for key, description, validator in bcse_parameters:
            if line.lower().startswith(description.lower()):
                value = line.split('=')[1].strip()
                value = validate(lines.line_number, 1, description, validator, value)
                results.append((key, value))
    return results


def unparse_bcse(bcse):
    lines = []
    lines.append(' :Start BCSE:')
    lines.append('')
    for key, description, validator in bcse_parameters:
        value = bcse[key]
        if isinstance(validator, Boolean):
            if value:
                value = ' On'
            else:
                value = ' Off'
        if value is None:
            value = ''
        if key == 'russian_roulette':
            value = value.lower()
        lines.append(' {}={}'.format(description, value))
    lines.append('')
    lines.append(' :Stop BCSE:')
    lines.append(' #########################')
    return lines


def parse_egsinp(text):
    lines = LineIterator(text.split('\n'))
    d = {}
    d.update(pickone(lines, 'title', Word()))
    d.update(pickone(lines, 'default_medium', Medium()))
    d.update(pick(lines, [
        ('iwatch', Integers([0, 1, 2, 4])),
        ('istore', Integers([0, 1])),
        ('irestart', Integers([0, 1, 2, 3, 4])),
        ('io_opt', Integers([0, 1, 2, 3, 4])),
        ('idat', Integers([0, 1])),
        ('latch_option', Integers([0, 1, 2, 3])),
        ('izlast', Integers([0, 1, 2]))
    ]))
    d.update(pick(lines, [
        ('ncase', PositiveInteger()),
        ('ixxn', NonNegativeInteger()),
        ('jxxn', NonNegativeInteger()),
        ('timmax', PositiveFloat()),
        ('ibrspl', Integers([0, 1, 2, 29])),
        ('nbrspl', NonNegativeInteger()),
        ('irrltt', Integers([0, 1, 2])),
        ('icm_split', NonNegativeInteger())
    ]))
    if d['ibrspl'] >= 2:
        d.update(pick(lines, [
            ('fs', PositiveFloat()),
            ('ssd', PositiveFloat()),
            ('nmin', PositiveInteger()),
            ('icm_dbs', NonNegativeInteger()),
            ('zplane_dbs', NonNegativeInteger()),
            ('irad_dbs', NonNegativeFloat()),
            ('zrr_dbs', NonNegativeFloat())
        ]))
    if d['icm_split'] > 0:
        d.update(pick(lines, [
            ('nsplit_phot', NonNegativeInteger()),
            ('nsplit_elec', NonNegativeFloat())
        ]))
    d.update(pick(lines, [
        ('iqin', Integer()),
        ('isourc', Words(['1', '6', '13', '21']))  # in future can be 13a (!)
    ], peek=True))
    if d['isourc'] == '1':
        d.update(pick(lines, [
            ('iqin', Integer()),
            ('isourc', Word()),
            ('rbeam', PositiveFloat()),
            ('uinc', Float()),
            ('vinc', Float()),
            ('winc', Float())
        ]))
    elif d['isourc'] == '6':
        d.update(pick(lines, [
            ('iqin', Integer()),
            ('isourc', Word()),
            ('xbeam0', NonNegativeFloat()),
            ('ybeam0', NonNegativeFloat()),
            ('xbeam', NonNegativeFloat()),
            ('ybeam', NonNegativeFloat())
        ]))
    elif d['isourc'] == '13':
        d.update(pick(lines, [
            ('iqin', Integer()),
            ('isourc', Word()),
            ('ybeam', PositiveFloat()),
            ('zbeam', Float()),
            ('uinc', Float()),
            ('vinc', Float())
        ]))
    elif d['isourc'] == '21':
        d.update(pick(lines, [
            ('iqin', Integer()),
            ('isourc', Word()),
            ('init_icm', PositiveInteger()),
            ('nrcycl', NonNegativeInteger()),
            ('iparallel', NonNegativeInteger()),
            ('parnum', NonNegativeInteger()),
            ('isrc_dbs', Integers([0, 1])),
            ('rsrc_dbs', NonNegativeFloat()),
            ('ssdrc_dbs', NonNegativeFloat()),
            ('zsrc_dbs', NonNegativeFloat())
        ]))
        d.update(pickone(lines, 'spcnam', Any()))
    else:
        raise ValueError('Unsupported isourc {}'.format(d['isourc']))
    if d['isourc'] == '13a' or int(d['isourc']) < 21:
        d.update(pickone(lines, 'monoen', Integers([0, 1])))
        if d['monoen'] == 0:
            # is monoenergetic
            d.update(pickone(lines, 'ein', Float()))
        else:
            d.update(pickone(lines, 'filnam', Word()))
            d.update(pickone(lines, 'ioutsp', Word()))
    d.update(pick(lines, [
        ('estepin', NonNegativeInteger()),  # dummy
        ('smax', NonNegativeInteger()),  # dummy
        ('ecutin', NonNegativeFloat()),
        ('pcutin', NonNegativeFloat()),
        ('idoray', NonNegativeInteger()),  # dummy
        ('ireject_global', Integers([0, 1, 2, -1, -2])),
        ('esave_global', NonNegativeFloat()),
        ('iflour', NonNegativeInteger())  # dummy
    ]))
    d.update(pick(lines, [
        ('iforce', NonNegativeInteger()),
        ('nfmin', NonNegativeInteger()),
        ('nfmax', NonNegativeInteger()),
        ('nfcmin', NonNegativeInteger()),
        ('nfcmax', NonNegativeInteger())
    ]))
    # list of component modules to score
    iplane_to_cm = pickcounted(lines, 'nsc_planes', 'iplane_to_cm', NonNegativeInteger())
    d['scoring_planes'] = []
    for cm in iplane_to_cm:
        plane = {
            'cm': cm,
            'zones': []
        }
        plane.update(pick(lines, [
            ('nsc_zones', PositiveInteger()),
            ('mzone_type', PositiveInteger())
        ]))
        if plane['mzone_type'] in [0, 1]:
            # for now we just grab one
            for i in range(plane['nsc_zones']):
                # grab up to 10 values
                for position, value in enumerate([v for v in values(next(lines)) if v], 2):
                    plane['zones'].append(validate(lines.line_number, position, 'rscore_zone', PositiveFloat(), value))
        else:
            plane['zones'].append(pick(lines, [
                ('xmin_zone', Float()),
                ('xmax_zone', Float()),
                ('ymin_zone', Float()),
                ('ymax_zone', Float()),
                ('nx_zone', NonNegativeInteger()),
                ('ny_zone', NonNegativeInteger())
            ]))
        d['scoring_planes'].append(plane)
    d.update(pickone(lines, 'itdose_on', Integers([0, 1])))
    if d['itdose_on']:
        raise NotImplementedError("itdose_on not yet implemented")
        d.update(pick(lines, [
            'icm_contam',
            'iq_contam'
        ]))
        d.update(pickone(lines, 'lnexc', NonNegativeInteger()))
        for i in range(d['lnexc']):
            raise NotImplementedError()
            next(lines)
        d.update(pickone(lines, 'lninc', NonNegativeInteger()))
        for i in range(d['lninc']):
            raise NotImplementedError()
            next(lines)
    d.update(pickone(lines, 'z_min_cm', NonNegativeFloat()))
    d['cms'] = []
    while True:
        line = next(lines)
        mo = re.search('CM (\w+) with identifier (\w+)', line)
        if mo is None:
            break
        cm = mo.group(1)
        identifier = mo.group(2)
        if cm == 'SLABS':
            dcm = parse_slabs(lines)
        elif cm == 'XTUBE':
            dcm = parse_xtube(lines)
        elif cm == 'BLOCK':
            dcm = parse_block(lines)
        dcm['type'] = cm
        dcm['identifier'] = identifier
        d['cms'].append(dcm)
    d.update(parse_mc_transport(lines))
    d.update(parse_bcse(lines))
    return d


def unparse_egsinp(d):
    lines = [
        d['title'].ljust(81) + '#!GUI1.0',  # so the GUI does not complain
        d['default_medium'],
        commalist(d, ('iwatch', 'istore', 'irestart', 'io_opt', 'idat', 'latch_option', 'izlast')) + ', IWATCH ETC.',
        commalist(d, ('ncase', 'ixxn', 'jxxn', 'timmax', 'ibrspl', 'nbrspl', 'irrltt', 'icm_split'))
    ]
    if d['ibrspl'] >= 2:
        lines.append(commalist(d, ('fs', 'ssd', 'nmin', 'icm_dbs', 'zplane_dbs', 'irad_dbs', 'zrr_dbs')))
    if d['icm_split'] > 0:
        lines.append(commalist(d, ('nsplit_phot', 'nsplit_elec')))
    if d['isourc'] == '1':
        lines.append(commalist(d, ('iqin', 'isourc', 'rbeam', 'uinc', 'vinc', 'winc')))
    elif d['isourc'] == '6':
        lines.append(commalist(d, ('iqin', 'isourc', 'xbeam0', 'ybeam0', 'xbeam', 'ybeam')))
    elif d['isourc'] == '13':
        lines.append(commalist(d, ('iqin', 'isourc', 'ybeam', 'zbeam', 'uinc', 'vinc')))
    elif d['isourc'] == '21':
        lines.append(commalist(d, ('iqin', 'isourc', 'init_icm', 'nrcycl', 'iparallel', 'parnum', 'isrc_dbs', 'rsrc_dbs', 'ssdrc_dbs', 'zsrc_dbs')))
        lines.append(commalist(d, ('spcnam', )))
    else:
        raise NotImplementedError('Unsupported isourc = {}'.format(d['isourc']))
    if d['isourc'] not in ['21', '24']:
        lines.append(d['monoen'])
    if d['isourc'] == '13a' or int(d['isourc']) < 21:
        if d['monoen'] == 0:
            lines.append(commalist(d, ['ein']))
        else:
            lines.append(d['filnam'])
            lines.append(d['ioutsp'])
    lines.append(commalist(d, ('estepin', 'smax', 'ecutin', 'pcutin', 'idoray', 'ireject_global', 'esave_global', 'iflour')))
    lines.append(commalist(d, ('iforce', 'nfmin', 'nfmax', 'nfcmin', 'nfcmax')))
    lines.append(', '.join(
        [str(len(d['scoring_planes']))] +
        [str(p['cm']) for p in d['scoring_planes']]
    ))
    for plane in d['scoring_planes']:
        lines.append(commalist(plane, ('nsc_zones', 'mzone_type')))
        if plane['mzone_type'] in [0, 1]:
            for zone in plane['zones']:
                lines.append('{:.5f}'.format(zone))
        else:
            for zone in plane['zones']:
                lines.append(commalist(zone, ('xmin_zone', 'xmax_zone', 'ymin_zone', 'ymax_zone', 'nx_zone', 'ny_zone')))
    lines.append(d['itdose_on'])
    if d['itdose_on']:
        raise NotImplementedError('itdose_on not yet implemented')
    lines.append(d['z_min_cm'])
    for cm in d['cms']:
        lines.append('*********** start of CM {} with identifier {}  ***********'.format(cm['type'], cm['identifier']))
        if cm['type'] == 'XTUBE':
            lines.extend(unparse_xtube(cm))
        elif cm['type'] == 'SLABS':
            lines.extend(unparse_slabs(cm))
        elif cm['type'] == 'BLOCK':
            lines.extend(unparse_block(cm))
        else:
            raise NotImplementedError('Unsupported cm type {}'.format(cm))
    lines.append('*********************end of all CMs*****************************')
    lines.append(' #########################')
    lines.extend(unparse_mc_transport(d))
    lines.extend(unparse_bcse(d))
    lines.append('')
    return '\n'.join(map(str, lines))


EPSILON = 0.000000000001


def verify(d):
    d = d.copy()
    if d['isourc'] == '13':
        d['uinc'] = -abs(d['uinc'])
        length = d['uinc'] * d['uinc'] + d['vinc'] * d['vinc']
        if length < EPSILON:
            d['uinc'] = -1
            d['vinc'] = 0
        else:
            d['uinc'] /= length
            d['vinc'] /= length
    return d


def polygon_area(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


if __name__ == '__main__':
    import argparse
    import json
    import statistics
    parser = argparse.ArgumentParser()
    parser.add_argument('--print', help='Print the JSON representation of this egsinp')
    parser.add_argument('--regurg', help='Read in an egsinp and write it out again (for testing)')
    parser.add_argument('--collimator', help='Output some stats on a collimator')
    args = parser.parse_args()
    if args.print:
        egsinp = parse_egsinp(open(args.print).read())
        print(json.dumps(egsinp, indent='\t'))
    elif args.regurg:
        egsinp = parse_egsinp(open(args.regurg).read())
        open(args.regurg.replace('.egsinp', '.egsregurg'), 'w').write(unparse_egsinp(egsinp))
    elif args.collimator:
        egsinp = parse_egsinp(open(args.collimator).read())
        blocks = [cm for cm in egsinp['cms'] if cm['type'] == 'BLOCK']
        max_y = 0
        max_x = 0
        areas = []
        for region in blocks[0]['regions']:
            area = polygon_area([(p['x'], p['y']) for p in region['points']])
            max_y = max(max_y, max(p['y'] for p in region['points']))
            max_x = max(max_x, max(p['x'] for p in region['points']))
            areas.append(area)
        print('first block')
        print('\tnumber of regions', len(areas))
        print('\taverage area', statistics.mean(areas))
        print('\ttotal area', sum(areas))
        print('\tmax x', max_x)
        print('\tmax y', max_y)
        print('\tzfocus', blocks[0]['zfocus'])
        max_y = 0
        max_x = 0
        areas = []
        for region in blocks[-1]['regions']:
            area = polygon_area([(p['x'], p['y']) for p in region['points']])
            max_y = max(max_y, max(p['y'] for p in region['points']))
            max_x = max(max_x, max(p['x'] for p in region['points']))
            areas.append(area)
        print('final block')
        print('\tnumber of regions', len(areas))
        print('\taverage area', statistics.mean(areas))
        print('\ttotal area', sum(areas))
        print('\tmax x', max_x)
        print('\tmax y', max_y)
        print('\tzfocus', blocks[-1]['zfocus'])
        print('total blocks', len(blocks))
