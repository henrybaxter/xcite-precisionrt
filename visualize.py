import argparse
import sys
import time
import hashlib

import numpy as np

import egsinp
import cgs


def main():
    args = parse_args()
    prev_hsh = None
    while True:
        # watch file hash to avoid causing OpenSCAD to re-render continuously
        hsh = hashlib.sha256(open(args.egsinp).read().encode('utf-8')).hexdigest()
        if hsh == prev_hsh:
            time.sleep(.2)
            continue
        prev_hsh = hsh

        scene = cgs.CGSElement()
        blocks = get_blocks(args.egsinp)

        # assume phantom is centered at zfocus of first block
        phantom = cgs.CGSSphere(2)
        phantom.translate((0, 0, blocks[0]['zfocus'] * 10))
        scene.add(phantom)

        collimator = build_collimator(blocks)
        scene.add(collimator)

        output_path = args.egsinp.replace('.egsinp', '.scad')
        print('Rendering to {}'.format(output_path))
        open(output_path, 'w').write(scene.render())

        if not args.watch:
            sys.exit()


def build_collimator(blocks):
    element = cgs.CGSElement()
    for block in blocks:
        for region in block['regions']:
            points = [(p['x'] * 10, p['y'] * 10) for p in region['points']]
            points, faces = solid_region(points, block['zmin'] * 10, block['zmax'] * 10, block['zfocus'] * 10)
            element.primitives.append(('polyhedron', points, faces))
    return element


def solid_region(planar_points, zmin, zmax, zfocus):
    far_points = []
    near_points = []
    # i specify points CCW in my egsinp
    # and OpenSCAD expects them CW (when looking at the face)
    # so reverse them
    for x, y in reversed(planar_points):
        v0 = np.array([x, y, zmin])
        vz = np.array([0, 0, zfocus])
        v = vz - v0
        t = (zmax - zmin) / v[2]
        v1 = v0 + t * v
        near_points.append(v0.tolist())
        far_points.append(v1.tolist())
    points = near_points + far_points
    faces = [
        list(range(len(near_points))),
        list(range(len(near_points), len(near_points) + len(far_points))),
    ]
    for i in range(len(far_points)):
        i, (i + 1) % len(far_points)
        faces.append([
            len(near_points) + i,
            len(near_points) + (i + 1) % len(far_points),
            (i + 1) % len(near_points),
            i
        ])
    return (points, faces)


def get_blocks(path):
    cms = egsinp.parse_egsinp(open(path).read())['cms']
    # in case they have other geometry before the collimator, strip it out
    blocks = [cm for cm in cms if cm['type'] == 'BLOCK']
    if not cms:
        print('Could not find any BLOCK CMs in {}'.format(path))
        sys.exit(1)
    return blocks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('egsinp')
    parser.add_argument('--watch', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
