
#!/usr/bin/env python3
import argparse, os, re, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mjcf', required=True, help='Path to MJCF XML file')
    ap.add_argument('--meshdir', default=None, help='Override meshdir (defaults to sibling ./meshes)')
    args = ap.parse_args()

    xml = Path(args.mjcf).read_text(errors='ignore')
    # report meshdir
    mdir = args.meshdir
    if mdir is None:
        mdir = str(Path(args.mjcf).parent / 'meshes')
    print(f'[info] meshdir assumed: {mdir}')

    # extract mesh file references
    meshes = re.findall(r'file="([^"]+\.stl)"', xml, flags=re.I)
    uniq = sorted(set(meshes))
    print(f'[info] referenced STL files: {len(uniq)}')
    missing = []
    present = []
    for f in uniq:
        # consider meshdir base
        candidate = Path(mdir) / Path(f).name if not Path(f).is_absolute() else Path(f)
        if candidate.exists():
            present.append(str(candidate))
        else:
            missing.append(f)
    if missing:
        print('[warn] Missing STL files ({}):'.format(len(missing)))
        for f in missing:
            print('  -', f)
    else:
        print('[ok] All STL files are present under', mdir)

    # extract actuator names
    actuators = re.findall(r'<motor[^>]*name="([^"]+)"', xml)
    print(f'[info] actuators found ({len(actuators)}):', ', '.join(actuators))
    # extract joint names
    joints = re.findall(r'<joint[^>]*name="([^"]+)"', xml)
    print(f'[info] joints found ({len(joints)}):', ', '.join(joints[:20]) + (' ...' if len(joints)>20 else ''))

    # generate mapping skeleton for 4-legs (fl,fr,bl,br) grouping by common prefixes
    groups = {'fl': [], 'fr': [], 'bl': [], 'br': []}
    for a in actuators:
        low = a.lower()
        if 'front' in low and 'left' in low:
            groups['fl'].append(a)
        elif 'front' in low and 'right' in low:
            groups['fr'].append(a)
        elif 'back' in low and 'left' in low:
            groups['bl'].append(a)
        elif 'back' in low and 'right' in low:
            groups['br'].append(a)
    print('[info] actuator mapping skeleton (edit as needed):')
    for k,v in groups.items():
        print(f'  {k}: {v}')

if __name__ == '__main__':
    main()
