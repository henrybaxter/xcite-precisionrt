import logging
import sys
import pprint
import argparse
import pyparsing as pp

# from collada import Collada, source, geometry, material, scene


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def commaSeparatedLine(fields):
    line = fields[0]
    for field in fields[1:]:
        line += ',' + field
    line += pp.restOfLine.suppress()
    return line


def integer(toks):
    return int(toks[0])

floatNumber = pp.Regex(r'-?\d+(\.\d*)?([eE]\d+)?').setParseAction(lambda t: float(t[0]))
positiveFloat = pp.Regex(r'\d+(\.\d*)?([eE]\d+)?').setParseAction(lambda t: float(t[0]))
integerNumber = pp.Regex(r'-?\d+').setParseAction(integer)
positiveInteger = pp.Regex(r'[1-9]\d*').setParseAction(integer)
nonNegativeInteger = pp.Regex(r'\d+').setParseAction(integer)


def nline_grammar():
    fields = [
        nonNegativeInteger('ncase'),
        nonNegativeInteger('ixxn'),
        nonNegativeInteger('jxxn'),
        floatNumber('timmax'),
        pp.oneOf(['0', '1'])('ibrspl'),
        nonNegativeInteger('nbrspl'),
        pp.oneOf(['0', '1', '2'])('irrltt'),
        None,
    ]

    IBRSPL = 4
    ICM_SPLIT = 7
    # no brem splitting, no icm
    fields[IBRSPL] = pp.oneOf(['0', '1'])('ibrspl')
    fields[ICM_SPLIT] = pp.Keyword('0')('icm_split')
    nline_no_brem_no_icm = commaSeparatedLine(fields)
    # no brem splitting, with icm
    fields[ICM_SPLIT] = positiveInteger('icm_split')
    nline_no_brem_with_icm = commaSeparatedLine(fields)
    # with brem, with icm
    fields[IBRSPL] = pp.oneOf(['2', '29'])('ibrspl')
    nline_with_brem_with_icm = commaSeparatedLine(fields)
    # with brem, no icm
    fields[ICM_SPLIT] = pp.Keyword('0')('ICM_SPLIT')
    nline_with_brem_no_icm = commaSeparatedLine(fields)

    # if ibrspl = 2 or ibrspl = 29
    ibrspl_line = commaSeparatedLine([
        floatNumber('fs'),
        floatNumber('ssd'),
        pp.Optional(nonNegativeInteger('nmin')),
        pp.Optional(nonNegativeInteger('icm_dbs')),
        pp.Optional(nonNegativeInteger)('zplane_dbs'),
        pp.Optional(pp.oneOf(['0', '1']))('irad_dbs'),
        pp.Optional(floatNumber)('zrr_dbs')
    ])

    # if ICM_SPLIT > 0
    icm_split_line = commaSeparatedLine([
        nonNegativeInteger('nsplit_phot'),
        nonNegativeInteger('nsplit_elec')
    ])

    return nline_no_brem_no_icm | \
        (nline_no_brem_with_icm + icm_split_line) | \
        (nline_with_brem_no_icm + ibrspl_line) | \
        (nline_with_brem_with_icm + ibrspl_line + icm_split_line)


def source_grammar():
    iqin = integerNumber('iqin')
    rbeam = floatNumber('rbream')
    uinc = floatNumber('uinc')
    vinc = floatNumber('vinc')
    winc = floatNumber('winc')
    ybeam = floatNumber('ybeam')
    zbeam = floatNumber('zbeam')
    gamma = floatNumber('gamma')
    zfocus = floatNumber('zfocus')
    rthetain = floatNumber('rthetain')
    thetain = floatNumber('thetain')
    cmsou = nonNegativeInteger('cmsou')
    spcnam = pp.Word(pp.printables)

    sourc1 = commaSeparatedLine([iqin, pp.Keyword('1'), rbeam, uinc, vinc, winc])
    sourc13 = commaSeparatedLine([iqin, pp.Keyword('13'), ybeam, zbeam, uinc, vinc])
    sourc15 = commaSeparatedLine([iqin, pp.Keyword('15'), gamma, zfocus, rthetain, thetain])
    sourc31 = commaSeparatedLine([iqin, pp.Keyword('31'), cmsou]) + commaSeparatedLine([spcnam])

    mono_energy = pp.Keyword('0')('monoen') + pp.restOfLine.suppress() + \
        floatNumber('ein') + pp.restOfLine.suppress()
    energy_spectrum = pp.Keyword('1')('monoen') + pp.restOfLine.suppress() + \
        pp.Word(pp.printables)('filnam') + pp.restOfLine.suppress() + \
        pp.oneOf(['0', '1'])('ioutsp') + pp.restOfLine.suppress()

    energy = mono_energy | energy_spectrum

    sourcWithEnergy = (sourc1 | sourc13 | sourc15) + energy
    sourcWithoutEnergy = sourc31
    return sourcWithEnergy | sourcWithoutEnergy

cm_scoring = commaSeparatedLine([
    positiveFloat('ecut'),
    positiveFloat('pcut'),
    nonNegativeInteger('dose_zone'),
    nonNegativeInteger('iregion_to_bit')
])

medium = commaSeparatedLine([pp.Word(pp.alphanums)])


def xtube_cm_grammar():
    rmax_cm = commaSeparatedLine([floatNumber('rmax_cm')])
    title = commaSeparatedLine([pp.Word(pp.alphanums)('title')])
    z = commaSeparatedLine([
        positiveFloat('zmin'),
        positiveFloat('zthick')
    ])
    anglei = commaSeparatedLine([positiveFloat('anglei')])
    use_extra_region = pp.Optional(pp.oneOf(['0', '1']).setParseAction(integer))
    extra_region_definition = commaSeparatedLine([
        positiveFloat('width'),
        positiveFloat('height')]) + \
        cm_scoring + \
        medium
    extra_region = pp.Forward()

    def checkExtraRegion(toks):
        if toks[0]:
            extra_region << extra_region_definition
        else:
            extra_region << pp.Empty()
    use_extra_region.addParseAction(checkExtraRegion)
    n_layers = commaSeparatedLine([nonNegativeInteger('n_layers')])
    layer_definitions = pp.Forward()
    thickness = commaSeparatedLine([positiveFloat('zthickness'), use_extra_region])
    layer = thickness + cm_scoring + medium

    def countedLayers(toks):
        n = toks[0]
        layer_definitions << pp.And([layer] * n)
    layers = n_layers + layer_definitions
    front_of_target = cm_scoring + medium
    target_holder = cm_scoring + medium

    return pp.Group(rmax_cm + title + z + anglei + layers + front_of_target + target_holder)


def slabs_cm_grammar():
    rmax_cm = commaSeparatedLine([floatNumber('rmax_cm')])
    title = commaSeparatedLine([pp.Word(pp.alphanums)('title')])
    n_slabs = commaSeparatedLine([nonNegativeInteger('n_slabs')])
    zmin = commaSeparatedLine([positiveFloat('zmin')])
    slab = commaSeparatedLine([
        positiveFloat('zthickness'),
        positiveFloat('ecut'),
        positiveFloat('pcut'),
        nonNegativeInteger('dose_zone'),
        nonNegativeInteger('iregion_to_bit'),
        positiveFloat('esavein')
    ]) + medium
    slabs = pp.Forward()

    def countedSlabs(toks):
        n = toks[0]
        slabs << [slab] * n
    n_slabs.addParseAction(countedSlabs)
    return pp.Group(rmax_cm + title + n_slabs + zmin + slabs)


def block_cm_grammar():
    rmax_cm = commaSeparatedLine([floatNumber('rmax_cm')])
    title = commaSeparatedLine([pp.Word(pp.alphanums)('title')])
    z = commaSeparatedLine([
        positiveFloat('zmin'),
        positiveFloat('zmax'),
        positiveFloat('zfocus')
    ])

    point = pp.Group(commaSeparatedLine([floatNumber('x'), floatNumber('y')]))
    n_points = commaSeparatedLine([nonNegativeInteger('n_points')])
    points = pp.Forward()

    def countedPoints(toks):
        n = toks[0]
        points << pp.And([point] * n)
    n_points.addParseAction(countedPoints)
    region = pp.Group(n_points + points)

    n_regions = commaSeparatedLine([nonNegativeInteger('n_regions')])
    region_definitions = pp.Forward()

    def countedRegions(toks):
        n = toks[0]
        region_definitions << pp.And([region] * n)
    n_regions.addParseAction(countedRegions)
    regions = n_regions + region_definitions

    block = commaSeparatedLine([
        floatNumber('ecut'),
        floatNumber('pcut'),
        nonNegativeInteger('dose_zone'),
        nonNegativeInteger('iregion_to_bit')
    ])
    medium = commaSeparatedLine([pp.Word(pp.alphanums)('medium')])
    airgap = block + medium
    openings = block + medium

    return pp.Group(rmax_cm + title + z + regions + airgap + openings + block + medium)


def build_grammar():
    title = pp.Word(pp.alphanums).setResultsName('title') + pp.restOfLine.suppress()
    medium = pp.Word(pp.alphanums).setResultsName('default_medium') + pp.restOfLine.suppress()
    iline = commaSeparatedLine([
        pp.oneOf(['0', '1', '2', '4'])('iwatch'),
        pp.oneOf(['0', '1'])('istore'),
        pp.oneOf(['0', '1', '2', '3', '4'])('irestart'),
        pp.oneOf(['0', '1', '2', '3', '4'])('io_opt'),
        pp.oneOf(['0', '1'])('idat'),
        pp.oneOf(['0', '1', '2', '3'])('latch_option'),
        pp.oneOf(['0', '1', '2'])('izlast')
    ])
    header = title + medium + iline
    nline = nline_grammar()
    source = source_grammar()

    eline = commaSeparatedLine([
        pp.Optional(floatNumber('estepin')),
        pp.Optional(nonNegativeInteger('smax')),
        floatNumber('ecutin'),
        floatNumber('pcutin'),
        pp.Optional(pp.oneOf(['0', '1'])('idoray')),
        pp.oneOf(['-2', '-1', '0', '1', '2'])('ireject_global'),
        pp.Optional(floatNumber('esave_global')),
        pp.Optional(pp.oneOf(['0', '1'])('iflour')),
    ])

    # photon forcing
    pline = commaSeparatedLine([
        pp.oneOf(['0', '1'])('iforce').setParseAction(integer),
        nonNegativeInteger('nfmin'),
        nonNegativeInteger('nfmax'),
        nonNegativeInteger('nfcmin'),
        nonNegativeInteger('nfcmax')
    ])

    # scoring plane NOT DONE
    nsc_planes = nonNegativeInteger('nsc_planes').setParseAction(integer)
    iplane_to_cm = pp.Forward()

    def countedPlanes(toks):
        n_planes = toks[0]
        iplane_to_cm << pp.And([pp.Literal(',').suppress(), positiveInteger] * n_planes)('iplane_to_cm')
    nsc_planes.addParseAction(countedPlanes)
    scoring_planes = nsc_planes + iplane_to_cm


    """
    iplane_to_cm = ZeroOrMore(positiveInteger + sep.suppress())('iplane_to_cm')
    scoring_planes = nsc_planes + sep + iplane_to_cm + restOfLine.suppress()
    zone_line = commaSeparatedLine([positiveInteger('nsc_zones'), oneOf(['0', '1', '2'])('mzone_type')])
    square_scoring = commaSeparatedLine([positiveInteger('nsc_zones'), Keyword('0')('mzone_type')]) + restOfLine.suppress() + \

    scoring_plane = square_scoring | annular_scoring | grid_scoring

    # dose components calculation input NOT DONE
    dose_on = commaSeparatedLine([Keyword('1')('itdose_on')])
    dose_off = commaSeparatedLine([Keyword('0')('itdose_off')])
    contam = commaSeparatedLine([Word(nums)('icm_contam'), oneOf(['0', '1'])('iq_contam')])
    dose_components = dose_off # | (dose_on + contam + lnexc_group + lninc_group)

    # input front surface z for cm 1
    front_surface = nonNegativeInteger('z_min_cm')
    """
    cms = pp.OneOrMore(block_cm_grammar() | xtube_cm_grammar() | slabs_cm_grammar())
    return header + nline + source + eline + pline + scoring_planes + cms # + dose_components + front_surface


# args
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('egsinp')
    parser.add_argument('dae', nargs='?', default='output.dae')
    return parser.parse_args()


def read_egs(fname):
    try:
        f = open(fname)
    except OSError as e:
        print("Could not open {}: {}".format(fname, str(e)))
        sys.exit(1)
    return f.read()


if __name__ == '__main__':
    args = parse_args()
    egsinp = read_egs(args.egsinp)
    grammar = build_grammar()
    result = grammar.parseString(egsinp)
    pprint.pprint(dict(result.items()))


    #vertices, indices = calculate_verticies(result)
    #export_dae(vertices, indices, args.dae)
