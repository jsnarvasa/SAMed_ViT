import os
import copy
import numpy as np
import argparse
import json
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--timeseries', action='store_true',
                    help='Whether to include timeseries data')
parser.add_argument('--full_channels', action='store_true',
                    help='Whether to include all channels')
parser.add_argument('--full_channels_normalise', action='store_true',
                    help='Whether to include all channels, and normalise them')
args = parser.parse_args()

if args.full_channels and not args.timeseries:
    raise ValueError('Cannot include full channels without timeseries data')

if args.full_channels_normalise and not args.timeseries:
    raise ValueError('Cannot include full channels without timeseries data')

if args.timeseries and not args.full_channels and not args.full_channels_normalise:
    SAVE_PATH = '/home/narvjes/data/PASTIS/SAMed_timeseries'
    FILE_LISTS_PATH = '/home/narvjes/repos/SAMed-jnar/lists/lists_PASTIS_timeseries'
    METADATA_PATH = '/home/narvjes/data/PASTIS/metadata.geojson'
elif args.timeseries and args.full_channels:
    SAVE_PATH = '/home/narvjes/data/PASTIS/SAMed_timeseries_full_channels'
    FILE_LISTS_PATH = '/home/narvjes/repos/SAMed-jnar/lists/lists_PASTIS_timeseries_full_channels'
    METADATA_PATH = '/home/narvjes/data/PASTIS/metadata.geojson'
elif args.timeseries and args.full_channels_normalise:
    SAVE_PATH = '/home/narvjes/data/PASTIS/SAMed_timeseries_full_channels_normalised'
    FILE_LISTS_PATH = '/home/narvjes/repos/SAMed-jnar/lists/lists_PASTIS_timeseries_full_channels_normalised'
    METADATA_PATH = '/home/narvjes/data/PASTIS/metadata.geojson'
else:
    SAVE_PATH = '/home/jesse/data/PASTIS/SAMed'
    FILE_LISTS_PATH = '/home/jesse/repos/SAMed-jnar/lists/lists_PASTIS'

IMAGE_PATH = '/home/narvjes/data/PASTIS/DATA_S2'
SEMANTIC_LABEL_PATH = '/home/narvjes/data/PASTIS/ANNOTATIONS'
FILE_LISTS_NAME = 'train.txt'
FILE_LISTS_TEST_NAME = 'pastis_test_vol.txt'

# Source: https://docs.digitalearthafrica.org/en/latest/data_specs/Sentinel-2_Level-2A_specs.html
MIN_VALUE = -1
MAX_VALUE = 1


if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

if not os.path.exists(FILE_LISTS_PATH):
    os.makedirs(FILE_LISTS_PATH)

# Replace file if already exists, and create empty file
# Do for both train and test files
with open(os.path.join(FILE_LISTS_PATH, FILE_LISTS_NAME), 'w') as f:
    pass
with open(os.path.join(FILE_LISTS_PATH, FILE_LISTS_TEST_NAME), 'w') as f:
    pass


S2_files = os.listdir(IMAGE_PATH)
S2_npy_files = [file for file in S2_files if file.endswith('.npy')]
S2_npy_files_filtered = copy.deepcopy(S2_npy_files)

# load the metadata file
if args.timeseries:
    with open(f'{METADATA_PATH}', 'r') as f:
        metadata_dict = json.load(f)


def calculate_ndvi(near_infrared_channel: np.array, red_channel: np.array) -> np.array:
    '''
    Calculates the Normalised Difference Vegetation Index (NDVI) from the near-infrared and red channels
    '''

    # Source: https://docs.digitalearthafrica.org/en/latest/data_specs/Sentinel-2_Level-2A_specs.html
    MIN_VALUE = -1
    MAX_VALUE = 1

    ndvi_channel = np.divide(
                        (near_infrared_channel - red_channel).astype('float64'),
                        (near_infrared_channel + red_channel).astype('float64'),
                        out=np.zeros_like(near_infrared_channel.astype('float64')),
                        where=(near_infrared_channel + red_channel) !=0
                    )

    ndvi_channel_normalised = (ndvi_channel - MIN_VALUE)/(MAX_VALUE - MIN_VALUE)

    # Clip the values to be between 0 and 1, since they would be extreme from the normalised NDVI values
    ndvi_channel_normalised = np.clip(ndvi_channel_normalised, 0, 1)
    
    return ndvi_channel_normalised


for npy_file in S2_npy_files:
    print(f'Processing file {npy_file}')
    patch_id = npy_file.replace('S2_', '').replace('.npy', '')

    if args.timeseries:
        # the shape will be [n_observations, height, width]

        # NDVI Channels
        if not args.full_channels and not args.full_channels_normalise:
            near_infrared_channel = np.load(os.path.join(IMAGE_PATH, npy_file))[:,6,:,:]
            red_channel = np.load(os.path.join(IMAGE_PATH, npy_file))[:,2,:,:]

            S2_image_normalised = calculate_ndvi(near_infrared_channel, red_channel)

        # Full channels
        elif args.full_channels:
            S2_image_normalised = np.load(os.path.join(IMAGE_PATH, npy_file))

        elif args.full_channels_normalise:
            S2_image_normalised = np.load(os.path.join(IMAGE_PATH, npy_file))
            channel_max = [943, 1266, 1535, 1786, 3023, 3596, 3831, 3918, 2938, 2114]
            channel_min = [276, 500, 382, 863, 1810, 2050, 2175, 2298, 1612, 874]

            for i in range(10):
                S2_image_normalised[:,i,:,:] = np.divide(
                    (S2_image_normalised[:,i,:,:] - channel_min[i]).astype('float64'),
                    (np.array(channel_max[i]) - np.array(channel_min[i])).astype('float64'),
                    out=np.zeros_like(S2_image_normalised[:,i,:,:]).astype('float64'),
                    where=(channel_max[i] - channel_min[i]) !=0
                )
            
            S2_image_normalised = np.clip(S2_image_normalised, 0, 1)

    else:
        # the shape will be [height, width]
        # i.e. there will be no timeseries dimension
        near_infrared_channel = np.load(os.path.join(IMAGE_PATH, npy_file))[0,6,:,:]
        red_channel = np.load(os.path.join(IMAGE_PATH, npy_file))[0,2,:,:]

        S2_image_normalised = calculate_ndvi(near_infrared_channel, red_channel)



    # Note that the reason we are getting the 0th channel
    # is because channel 0 for TARGET files is the semantic labels
    # as shown here: https://github.com/VSainteuf/pastis-benchmark/blob/main/code/dataloader.py
    try:
        S2_semantic_labels = np.load(os.path.join(SEMANTIC_LABEL_PATH, f'TARGET_{patch_id}.npy'))[0]
    except:
        S2_npy_files_filtered.remove(npy_file)
        continue

    if args.timeseries:
        try:
            patch_metadata = [patch for patch in metadata_dict['features'] if patch['properties']['ID_PATCH'] == int(patch_id)][0]
        except IndexError:
            print(f'Patch {patch_id} not found in metadata file, continuing to the next patch')
            continue
        dates_list = list(patch_metadata['properties']['dates-S2'].values())
        doy = [datetime.datetime.strptime(str(date), '%Y%m%d').timetuple().tm_yday for date in dates_list]

        np.savez(
            os.path.join(SAVE_PATH, f'S2_{patch_id}.npz'),
            label = S2_semantic_labels,
            image = S2_image_normalised,
            doy = doy,
        )
    else:
        np.savez(
            os.path.join(SAVE_PATH, f'S2_{patch_id}.npz'),
            label = S2_semantic_labels,
            image = S2_image_normalised,
        )

    # Removing the .npy suffix from the text file, since just need S2_{patch_id}
    # The train.py script will be responsible for appending the .npz extension to the filenames
    if patch_metadata['properties']['Fold'] in (1,2,3,4):
        with open(os.path.join(FILE_LISTS_PATH, FILE_LISTS_NAME), 'a') as f:
            f.write(npy_file.replace('.npy', '') + '\n')
    elif patch_metadata['properties']['Fold'] == 5:
        with open(os.path.join(FILE_LISTS_PATH, FILE_LISTS_TEST_NAME), 'a') as f:
            f.write(npy_file.replace('.npy', '') + '\n')
    

print(f'Conversion of S2 files to .npz successful and txt file created in {FILE_LISTS_PATH}')
