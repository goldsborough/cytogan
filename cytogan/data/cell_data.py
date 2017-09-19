import glob
import os.path
import re

import pandas as pd
import tqdm

from cytogan.data.image_loader import AsyncImageLoader
from cytogan.extra import logs

log = logs.get_logger(__name__)


def _normalize_luminance(images):
    normalized = []
    for image in images:
        maxima = image.max(axis=(0, 1))
        if maxima.sum() > 0:
            image /= maxima.reshape(1, 1, -1)
        normalized.append(image)
    return normalized


def _image_key_for_path(path, root_path):
    # We use the path relative to the root_path, without the file extension, as
    # the image key.
    relative_path = os.path.relpath(path, start=root_path)
    return os.path.splitext(relative_path)[0]


def _get_single_cell_names(root_path, plate_names, file_names, patterns):
    assert os.path.isabs(root_path)
    assert os.path.exists(root_path)
    original_indices = []
    single_cell_names = []
    file_range = tqdm.tqdm(enumerate(zip(plate_names, file_names)))
    file_range.unit = ' files'
    for index, components in file_range:
        image_path = os.path.join(*components)
        if patterns and not any(p.search(image_path) for p in patterns):
            continue
        full_path = os.path.join(root_path, image_path)
        assert os.path.isabs(full_path)
        # We assume single-cell images are stored wit the original image name
        # as prefix and then '-{digit}' suffixes, where {digit} is the id/number
        # of the cell within the image.
        glob_paths = glob.glob('{0}-*'.format(full_path))
        image_keys = [_image_key_for_path(p, root_path) for p in glob_paths]
        single_cell_names.extend(image_keys)
        original_indices.extend([index] * len(image_keys))

    return original_indices, single_cell_names


def _load_single_cell_names_from_cell_count_file(metadata, cell_count_path):
    log.info('Using cell count file %s', cell_count_path)
    indices = []
    single_cell_names = []
    with open(cell_count_path) as cell_count_file:
        next(cell_count_file)  # skip header
        for line in tqdm.tqdm(cell_count_file, unit=' files'):
            key, count = line.split(',')
            plate, file_name = key.split('/')
            file_name += '.tif'
            plate_index = metadata['Image_Metadata_Plate_DAPI'] == plate
            file_index = metadata['Image_FileName_DAPI'] == file_name
            index = metadata[plate_index & file_index].index[0]
            for cell_index in range(int(count)):
                indices.append(index)
                single_cell_names.append('{0}-{1}'.format(key, cell_index))

    return indices, single_cell_names


# Takes all metadata as a dataframe and returns a new dataframe with only the
# relevant information, which has columns:
# - key (Image_Metadata_Plate_DAPI/Image_FileName_DAPI-0),
# - compound
# - concentration
# Note that for a particular image path in the original dataframe, we will not
# actually use the path of that image, but of the single cell images, assumed to
# have the original image name as a prefix.
def _preprocess_metadata(metadata, patterns, root_path, cell_count_path):
    plate_names = list(metadata['Image_Metadata_Plate_DAPI'])
    full_file_names = metadata['Image_FileName_DAPI']
    file_names = [os.path.splitext(name)[0] for name in full_file_names]

    log.info('Reading single-cell names')
    if cell_count_path is None:
        if patterns:
            assert not isinstance(patterns, str)
            patterns = [re.compile(pattern) for pattern in patterns]
        indices, image_keys = _get_single_cell_names(root_path, plate_names,
                                                     file_names, patterns)
    else:
        indices, image_keys = _load_single_cell_names_from_cell_count_file(
            metadata, cell_count_path)

    compounds = metadata['Image_Metadata_Compound'].iloc[indices]
    concentrations = metadata['Image_Metadata_Concentration'].iloc[indices]

    data = dict(compound=list(compounds), concentration=list(concentrations))
    processed = pd.DataFrame(data=data, index=image_keys)
    processed.index.name = 'key'

    return processed


class CellData(object):
    def __init__(self,
                 metadata_file_path,
                 labels_file_path,
                 image_root,
                 cell_count_path=None,
                 patterns=None,
                 normalize_luminance=False):
        self.image_root = os.path.realpath(image_root)

        self.labels = pd.read_csv(labels_file_path)
        self.labels.set_index(['compound', 'concentration'], inplace=True)

        all_metadata = pd.read_csv(metadata_file_path)
        self.metadata = _preprocess_metadata(all_metadata, patterns,
                                             self.image_root, cell_count_path)

        unique_compounds = set(map(tuple, self.metadata.values))
        log.info('Have {0:,} single-cell images for {1} unique '
                 '(compound, concentration) pairs with {2} MOA labels'.format(
                     len(self.metadata),
                     len(unique_compounds), len(self.labels)))

        self.images = AsyncImageLoader(self.image_root)
        self.normalize_luminance = normalize_luminance
        self.batch_index = 0

    @property
    def number_of_images(self):
        return self.metadata.shape[0]

    def next_batch(self, number_of_images, with_keys=False):
        last_index = self.batch_index + number_of_images
        keys = self.metadata.iloc[self.batch_index:last_index].index
        ok_keys, ok_images = self.images[keys]

        self.batch_index = last_index
        if self.batch_index >= self.number_of_images:
            self.reset_batching_state()

        next_keys = self.metadata.iloc[
            self.batch_index:self.batch_index + number_of_images].index
        self.images.fetch_async(next_keys)

        if self.normalize_luminance:
            ok_images = _normalize_luminance(ok_images)

        if with_keys:
            return ok_keys, ok_images
        return ok_images

    def reset_batching_state(self):
        self.batch_index = 0
        # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        self.metadata = self.metadata.sample(frac=1)

    def batches_of_size(self, batch_size):
        self.reset_batching_state()
        for start in range(0, self.number_of_images, batch_size):
            end = start + batch_size
            keys = self.metadata.iloc[start:end].index
            keys, images = self.images[keys]
            next_keys = self.metadata.iloc[end:end + batch_size].index
            self.images.fetch_async(next_keys)
            if self.normalize_luminance:
                images = _normalize_luminance(images)
            yield keys, images

    def create_dataset_from_profiles(self, keys, profiles):
        # First filter out metadata for irrelevant keys.
        relevant_metadata = self.metadata.loc[keys]
        compounds = relevant_metadata['compound']
        concentrations = relevant_metadata['concentration']
        # The keys to the labels dataframe are (compound, concentration) pairs.
        labels = self.labels.loc[list(zip(compounds, concentrations))]

        dataset = pd.DataFrame(
            index=keys,
            data=dict(
                compound=compounds,
                concentration=concentrations,
                moa=list(labels.moa),
                profile=list(profiles)))

        # Ignore (compound, concentration) pairs for which we don't have labels.
        dataset.dropna(inplace=True)

        return dataset

    def get_treatment_indices(self, keys):
        filtered = self.metadata.loc[keys][['compound', 'concentration']]
        compounds = list(set(map(tuple, filtered.values.tolist())))
        index_map = {(com, con): i for i, (com, con) in enumerate(compounds)}
        indices = filtered.apply(lambda row: index_map[tuple(row)], axis=1)
        compound_strings = ['{}/{}'.format(com, con) for com, con in compounds]

        return compound_strings, indices

    def get_compound_indices(self, dataset):
        compounds = list(dataset['compound'].unique())
        indices = dataset['compound'].apply(compounds.index)
        return compounds, indices

    def get_moa_indices(self, dataset):
        moas = list(dataset['moa'].unique())
        indices = dataset['moa'].apply(moas.index)
        return moas, indices
