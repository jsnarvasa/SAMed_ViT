import os
import numpy as np
import pickle
import json


PASTIS24_DIR = '/home/narvjes/data/PASTIS24/pickle24x24'
SAVE_DIR = '/home/narvjes/data/PASTIS24/SAMed_timeseries_full_channels_normalised'
METADATA_PATH = '/home/narvjes/data/PASTIS/metadata.geojson'
FILE_LISTS_PATH = '/home/narvjes/repos/SAMed-jnar/lists/lists_PASTIS24_timeseries_full_channels_normalised'
FILE_LISTS_TRAIN_NAME = 'train.txt'
FILE_LISTS_TEST_NAME = 'pastis_test_vol.txt'
NUMBER_CHANNELS = 10

S2_files = os.listdir(PASTIS24_DIR)
S2_pickle_files = [file for file in S2_files if file.endswith('.pickle')]

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
if not os.path.exists(FILE_LISTS_PATH):
    os.makedirs(FILE_LISTS_PATH)

# Load the metadata into a dictionary
with open(f'{METADATA_PATH}', 'r') as f:
    metadata_dict = json.load(f)

# Create the train and test files if they don't exist yet
with open(os.path.join(FILE_LISTS_PATH, FILE_LISTS_TRAIN_NAME), 'w') as f:
    pass
with open(os.path.join(FILE_LISTS_PATH, FILE_LISTS_TEST_NAME), 'w') as f:
    pass


def normalise_channels(img: np.array, num_channels: int) -> np.array:
    '''
    Normalise channels of the input data
    and return normalised image
    '''

    CHANNEL_MAX = [943, 1266, 1535, 1786, 3023, 3596, 3831, 3918, 2938, 2114]
    CHANNEL_MIN = [276, 500, 382, 863, 1810, 2050, 2175, 2298, 1612, 874]

    # Perform 0 to 1 normalisation for each channel
    for i in range(num_channels):
        img[:,i,:,:] = np.divide(
            (img[:, i, :, :] - CHANNEL_MIN[i]).astype('float64'),
            (np.array(CHANNEL_MAX[i]) - np.array(CHANNEL_MIN[i])).astype('float64'),
            out=np.zeros_like(img[:, i, :, :]).astype('float64'),
            where=(CHANNEL_MAX[i] - CHANNEL_MIN[i]) != 0
        )
    
    # Perform clipping to ensure values are between 0 and 1
    img = np.clip(img, 0, 1)

    return img

for pickle_file in S2_pickle_files:
    print(f'Processing file {pickle_file}')

    patch_id = pickle_file.replace('.pickle', '')

    with open(os.path.join(PASTIS24_DIR, pickle_file), 'rb') as f:
        data = pickle.load(f)

    # Send the image section of the data,
    # Label and doy should not be normalised
    img_normalised = normalise_channels(data['img'], NUMBER_CHANNELS)

    np.savez(
        os.path.join(SAVE_DIR, f'{patch_id}.npz'),
        label = data['labels'][0],
        image = img_normalised,
        doy = data['doy'],
    )

    patch_metadata = None
    for patch in metadata_dict['features']:
        if patch['properties']['ID_PATCH'] == int(patch_id.split('_')[0]):
            patch_metadata = patch
            break
    
    if patch_metadata is None:
        raise Exception(f'Patch {patch_id} not found')

    if patch_metadata['properties']['Fold'] in (1,2,3,4):
        with open(os.path.join(FILE_LISTS_PATH, FILE_LISTS_TRAIN_NAME), 'a') as f:
            f.write(f'{patch_id}\n')
    elif patch_metadata['properties']['Fold'] == 5:
        with open(os.path.join(FILE_LISTS_PATH, FILE_LISTS_TEST_NAME), 'a') as f:
            f.write(f'{patch_id}\n')

print(f'Conversion of S2 files to .npz successful and txt file created in {FILE_LISTS_PATH}')

