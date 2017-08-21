import pandas as pd
import os
import re
import glob


def image_key_for_path(path, root_path):
    # We use the path relative to the root_path, without the file extension, as
    # the image key.
    return os.path.relpath(path, startdir=root_path).splitext()[0]


def get_single_cell_names(root_path, plate_names, file_names, patterns):
    assert os.path.isabs(root_path)
    single_cell_names = []
    for plate_name, file_name in zip(plate_names, file_names):
        image_path = os.path.join(plate_name, file_name)
        if not any(pattern.search(image_path) for pattern in patterns):
            continue
        full_path = os.path.join(root_path, image_path)
        assert os.path.isabs(full_path)
        # We assume single-cell images are stored wit the original image name
        # as prefix and then '-{digit}' suffixes, where {digit} is the id/number
        # of the cell within the image.
        glob_result = glob.glob('{0}-*'.format(full_path))
        image_keys = [image_key_for_path(p, root_path) for p in glob_result]
        single_cell_names.extend(image_keys)

    return single_cell_names


# Takes all metadata as a dataframe and returns a new dataframe with only the
# relevant information, which has columns:
# - key (Image_Metadata_Plate_DAPI/Image_FileName_DAPI-0),
# - compound
# - concentration
# Note that for a particular image path in the original dataframe, we will not
# actually use the path of that image, but of the single cell images, assumed to
# have the original image name as a prefix.
def preprocess_metadata(metadata, patterns, root_path):
    plate_names = list(metadata['Image_Metadata_Plate_DAPI'])
    full_file_names = list(metadata['Image_FileName_DAPI'])
    file_names = full_file_names.apply(lambda f: os.path.splitext(f)[0])

    patterns = [re.compile(pattern) for pattern in patterns]
    image_names = get_single_cell_names(root_path, plate_names, file_names,
                                        patterns)

    compounds = metadata['Image_Metadata_Compound']
    concentrations = metadata['Image_Metadata_Concentration']

    data = dict(compound=list(compounds), concentration=list(concentrations))
    processed = pd.DataFrame(data=data, index=image_names)
    processed.index.name = 'name'

    return processed


class CellData(object):

    def __init__(self,
                 metadata_file_path,
                 labels_file_path,
                 image_root,
                 patterns=None):
        self.image_root = os.path.realpath(image_root)
        self.labels = pd.read_csv(labels_file_path)

        all_metadata = pd.read_csv(metadata_file_path)
        self.metadata = preprocess_metadata(all_metadata, patterns,
                                            self.image_root)
