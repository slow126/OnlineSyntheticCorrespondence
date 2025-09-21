# Combines all the annotation json files into a single json file each for train/val/test.
# It is much faster to load a single json file than several tens of thousands of them,
# and the disk is much less cluttered (which may be important if there are file quota
# limits on the system).
import json
from pathlib import Path
import shutil
import subprocess

import tqdm


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root_data_dir', type=str, help='Path to root of dataset')
    parser.add_argument('-r', '--remove', action=argparse.BooleanOptionalAction,
        help='Force removal of individual json files after merging')
    args = parser.parse_args()

    root_dir = Path(args.root_data_dir)

    for split in ['trn', 'val', 'test']:
        spt_path = root_dir.joinpath(f'Layout/large/{split}.txt')
        anno_path = root_dir.joinpath('PairAnnotation/', split)
        data = spt_path.open().read().strip().split('\n')
        annos = {}
        for p in tqdm.tqdm(data):
            ap = anno_path.joinpath(p).with_suffix('.json')
            d = json.load(ap.open())
            annos[p] = d
        temp_dest = root_dir.joinpath(f'pair_annotations_{split}.json')
        json.dump(annos, temp_dest.open('w'))

        print('Merging complete for "{split}"')

        remove = args.remove
        if not remove:
            answer = ''
            while answer.lower() not in ('y', 'n'):
                answer = input('Would you like to delete the original json files (cannot be undone)? [n|y]')
            remove = answer.lower() == 'y'
        if remove:
            print('Removing individual files')
            subprocess.run(['rm', '-r', str(anno_path)])

        Path.mkdir(anno_path)
        shutil.move(temp_dest, anno_path.joinpath('pair_annotations.json'))