import os
import copy
import numpy as np
import argparse
import json
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--timeseries', action='store_true',
                    help='Whether to include timeseries data')
args = parser.parse_args()

if args.timeseries:
    IMAGE_PATH = '/home/narvjes/data/PASTIS/DATA_S2'
    SEMANTIC_LABEL_PATH = '/home/narvjes/data/PASTIS/ANNOTATIONS'
    SAVE_PATH = '/home/narvjes/data/PASTIS/SAMed_timeseries'
    FILE_LISTS_PATH = '/home/narvjes/repos/SAMed-jnar/lists/lists_PASTIS_timeseries'
    FILE_LISTS_NAME = 'train.txt'
    METADATA_PATH = '/home/narvjes/data/PASTIS/metadata.geojson'
else:
    IMAGE_PATH = '/home/narvjes/data/PASTIS/DATA_S2'
    SEMANTIC_LABEL_PATH = '/home/narvjes/data/PASTIS/ANNOTATIONS'
    SAVE_PATH = '/home/narvjes/data/PASTIS/SAMed'
    FILE_LISTS_PATH = '/home/narvjes/repos/SAMed-jnar/lists/lists_PASTIS'
    FILE_LISTS_NAME = 'train.txt'

# Source: https://docs.digitalearthafrica.org/en/latest/data_specs/Sentinel-2_Level-2A_specs.html
MIN_VALUE = -1
MAX_VALUE = 1


if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

if not os.path.exists(FILE_LISTS_PATH):
    os.makedirs(FILE_LISTS_PATH)

S2_files = os.listdir(IMAGE_PATH)
S2_npy_files = [file for file in S2_files if file.endswith('.npy')]
S2_npy_files_filtered = copy.deepcopy(S2_npy_files)

# load the metadata file
if args.timeseries:
    with open(f'{METADATA_PATH}', 'r') as f:
        metadata_dict = json.load(f)

for npy_file in S2_npy_files:
    print(f'Processing file {npy_file}')
    patch_id = npy_file.replace('S2_', '').replace('.npy', '')

    # NDVI Channels
    if args.timeseries:
        # the shape will be [n_observations, height, width]
        near_infrared_channel = np.load(os.path.join(IMAGE_PATH, npy_file))[:,6,:,:]
        red_channel = np.load(os.path.join(IMAGE_PATH, npy_file))[:,2,:,:]
    else:
        # the shape will be [height, width]
        # i.e. there will be no timeseries dimension
        near_infrared_channel = np.load(os.path.join(IMAGE_PATH, npy_file))[0,6,:,:]
        red_channel = np.load(os.path.join(IMAGE_PATH, npy_file))[0,2,:,:]

    ndvi_channel = np.divide(
        (near_infrared_channel - red_channel).astype('float64'),
        (near_infrared_channel + red_channel).astype('float64'),
        out=np.zeros_like(near_infrared_channel.astype('float64')),
        where=(near_infrared_channel + red_channel) !=0
    )

    
    # Note that we are getting the first observation (index 0) of the S2 image
    S2_image = ndvi_channel
    S2_image_normalised = (S2_image - MIN_VALUE)/(MAX_VALUE - MIN_VALUE)

    # Clip the values to be between 0 and 1, since they would be extreme from the normalised NDVI values
    S2_image_normalised = np.clip(S2_image_normalised, 0, 1)

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
    

with open(os.path.join(FILE_LISTS_PATH, FILE_LISTS_NAME), 'w') as f:
    for patch in S2_npy_files_filtered:
        # Removing the .npy suffix from the text file, since just need S2_{patch_id}
        # The train.py script will be responsible for appending the .npz extension to the filenames
        f.write(patch.replace('.npy', '') + '\n')

print(f'Conversion of S2 files to .npz successful and txt file created in {FILE_LISTS_PATH}')
