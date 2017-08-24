import os
import argparse

parser = argparse.ArgumentParser(description='make-cell-file')
parser.add_argument('--image-path', required=True)
options = parser.parse_args()

counts = {}
for directory, _, filenames in os.walk(options.image_path):
    directory = os.path.relpath(directory, start=options.image_path)
    for filename in filenames:
        if not filename.endswith('.png') and not filename.endswith('.tif'):
            continue
        key_stop = filename.rfind('-')
        key = os.path.join(directory, filename[:key_stop])
        last_count = counts.get(key, 0)
        counts[key] = last_count + 1

print('key,number_of_cells')
for key, count in counts.items():
    print('{0},{1}'.format(key, count))
