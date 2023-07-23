import os
import numpy as np


IMAGE_PATH = '/home/narvjes/data/PASTIS/DATA_S2'
SEMANTIC_LABEL_PATH = '/home/narvjes/data/PASTIS/ANNOTATIONS'
SAVE_PATH = '/home/narvjes/data/PASTIS/SAMed'
FILE_LISTS_PATH = '/home/narvjes/repos/SAMed/lists/lists_PASTIS'
FILE_LISTS_NAME = 'train.txt'


if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

if not os.path.exists(FILE_LISTS_PATH):
    os.makedirs(FILE_LISTS_PATH)

S2_files = os.listdir(IMAGE_PATH)
S2_npy_files = [file for file in S2_files if file.endswith('.npy')]

for npy_file in S2_npy_files:
    print(f'Processing file {npy_file}')
    patch_id = npy_file.replace('S2_', '').replace('.npy', '')
    S2_image = np.load(os.path.join(IMAGE_PATH, npy_file))
    # Note that the reason we are getting the 0th channel
    # is because channel 0 for TARGET files is the semantic labels
    # as shown here: https://github.com/VSainteuf/pastis-benchmark/blob/main/code/dataloader.py
    try:
        S2_semantic_labels = np.load(os.path.join(SEMANTIC_LABEL_PATH, f'TARGET_{patch_id}.npy'))[0]
    except:
        continue

    np.savez(
        os.path.join(SAVE_PATH, f'{patch_id}.npz'),
        label = S2_semantic_labels,
        image = S2_image[0,0,:,:], # note that we are getting the first observation, and the first channel here
    )

with open(os.path.join(FILE_LISTS_PATH, FILE_LISTS_NAME)) as f:
    for patch in S2_npy_files:
        f.write(patch + '\n')

print(f'Conversion of S2 files to .npz successful and txt file created in {FILE_LISTS_PATH}')
