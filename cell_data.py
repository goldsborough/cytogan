import glob
import os
import re

import pandas as pd
import scipy.misc


def image_key_for_path(path, root_path):
    # We use the path relative to the root_path, without the file extension, as
    # the image key.
    relative_path = os.path.relpath(path, start=root_path)
    return os.path.splitext(relative_path)[0]


def get_single_cell_names(root_path, plate_names, file_names, patterns):
    assert os.path.isabs(root_path)
    original_indices = []
    single_cell_names = []
    for index, components in enumerate(zip(plate_names, file_names)):
        image_path = os.path.join(*components)
        if patterns and not any(p.search(image_path) for p in patterns):
            continue
        full_path = os.path.join(root_path, image_path)
        assert os.path.isabs(full_path)
        # We assume single-cell images are stored wit the original image name
        # as prefix and then '-{digit}' suffixes, where {digit} is the id/number
        # of the cell within the image.
        glob_paths = glob.glob('{0}-*'.format(full_path))
        image_keys = [image_key_for_path(p, root_path) for p in glob_paths]
        single_cell_names.extend(image_keys)
        original_indices.extend([index] * len(image_keys))

    return original_indices, single_cell_names


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
    full_file_names = metadata['Image_FileName_DAPI']
    file_names = [os.path.splitext(name)[0] for name in full_file_names]

    if patterns:
        assert not isinstance(patterns, str)
        patterns = [re.compile(pattern) for pattern in patterns]
    indices, image_keys = get_single_cell_names(root_path, plate_names,
                                                file_names, patterns)

    compounds = metadata['Image_Metadata_Compound'].iloc[indices]
    concentrations = metadata['Image_Metadata_Concentration'].iloc[indices]

    data = dict(compound=list(compounds), concentration=list(concentrations))
    processed = pd.DataFrame(data=data, index=image_keys)
    processed.index.name = 'key'

    return processed


def load_image(root_path, image_key, extension):
    full_path = os.path.join(root_path, '{0}.{1}'.format(image_key, extension))
    return scipy.misc.imread(full_path)


class LazyImageLoader(object):
    def __init__(self, root_path, extension='png'):
        self.root_path = root_path
        self.extension = extension
        self.loaded_images = {}

    def __getitem__(self, image_key):
        image = self.loaded_images.get(image_key)
        if image is None:
            image = load_image(self.root_path, image_key, self.extension)
            self.loaded_images[image_key] = image
        return image


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

        self.images = LazyImageLoader(self.image_root)

        self.batch_index = 0

    @property
    def number_of_images(self):
        return self.metadata.shape[0]

    def next_batch_of_images(self, number_of_images):
        if self.batch_index >= self.number_of_images:
            self.reset_batching_state()

        last_index = self.batch_index + number_of_images
        keys = self.metadata.iloc[self.batch_index:last_index].index
        self.batch_index = last_index

        return {key: self.images[key] for key in keys}

    def reset_batching_state(self):
        self.batch_index = 0
        # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        self.metadata = self.metadata.sample(frac=1)

    def all_images(self):
        return {key: self.images[key] for key in self.metadata.index}

    def create_dataset_from_profiles(self, profiles):
        return pd.DataFrame(
            index=self.metadata.index,
            data=dict(
                compound=self.metadata['compound'],
                concentration=self.metadata['concentration'],
                profile=profiles,
                label=self.labels))
