# SAMed with CNN & Temporal Transformer
This repository contains the model for SAMed with CNN & Temporal Transformer.
The core of this repository is based on the work from SAMed, with modifications made to integrate the temporal transformer component from TS-ViT.

Moreover, the model code for SAMed with Temporal Encoder and SAMed with CNN can also be found within this repository, **albeit in other branches**.

> Important Note:
> There are now two manually maintained lists, which keeps track of the custom components that have been added into the model.
> These are `CUSTOM_ITEMS` and `EXCEPTION_LIST`, which are used to ensure that only the relevant component parameters are stored in the .pth export
> and to ensure that the gradients within the custom components added within the Image Encoder do not have their weights frozen, respectively.

## Commands for Reproduction
Preprocessing command
```bash
nohup python preprocess/preprocess_data_pastis.py > 20230726_preprocess.log 2>&1 &
```

Training command
```bash
nohup python train.py  --root_path /home/narvjes/data/PASTIS/SAMed --output /home/narvjes/repos/SAMed/output --n_gpu 1 --batch_size 1  --num_workers 0 --list_dir ./lists/lists_PASTIS --num_classes 20 --img_size 128 --warmup --AdamW > 20230727_PASTIS_run.log 2>&1 &

nohup python train.py  --root_path /home/narvjes/data/PASTIS/SAMed --output /home/narvjes/repos/SAMed-jnar/output --n_gpu 1 --batch_size 22 --list_dir ./lists/lists_PASTIS --num_classes 20 --img_size 128 --stop_epoch 200 --warmup --AdamW > 20230731_PASTIS_run.log 2>&1 &

# On the A40 GPU
# The increase in max_epochs and stop_epoch is not necessary, just for testing
nohup python train.py  --root_path /home/narvjes/data/PASTIS/SAMed --output /home/narvjes/repos/SAMed-jnar/output --n_gpu 1 --batch_size 16 --list_dir ./lists/lists_PASTIS --num_classes 20 --img_size 128 --stop_epoch 400 --max_epochs 450 --warmup --AdamW > 20230731_PASTIS_run.log 2>&1 &

# For timeseries training
nohup python train.py --root_path /home/narvjes/data/PASTIS/SAMed_timeseries --output /home/narvjes/repos/SAMed-jnar/output --n_gpu 1 --batch_size 10 --list_dir ./lists/lists_PASTIS_timeseries --num_classes 20 --img_size 128 --stop_epoch 200 --max_epochs 250 --warmup --AdamW > 20230807_PASTIS_run.log 2>&1 &
```

Testing command
```bash
nohup python test.py --num_classes 20 --list_dir ./lists/lists_PASTIS --img_size 128 --ckpt /home/narvjes/repos/SAMed/checkpoints/sam_vit_b_01e
c64.pth --lora_ckpt /home/narvjes/repos/SAMed/output/Synapse_128_pretrain_vit_b_epo200_bs1_lr0.005/epoch_159.pth --output_dir /home/narvjes/repos/SAMed/output/test --volume_path /home/narvjes/data/PASTIS/SAMed --is_savenii > 20230729_PASTIS_test.log 2>&1 &

nohup python test.py --num_classes 20 --list_dir ./lists/lists_PASTIS --img_size 128 --ckpt /home/narvjes/repos/SAMed-jnar/checkpoints/sam_vit_b_01ec64.pth --lora_ckpt '/home/narvjes/repos/SAMed-jnar/output/Synapse_128_pretrain_vit_b_epo200_bs22_lr0.005_2023-07-30/epoch_199.pth' --output_dir /home/narvjes/repos/SAMed-jnar/output/test --volume_path /home/narvjes/data/PASTIS/SAMed --is_savenii > 20230731_PASTIS_test.log 2>&1 &

# Testing for timeseries data
nohup python test.py --num_classes 20 --list_dir ./lists/lists_PASTIS_timeseries --img_size 128 --ckpt /home/narvjes/repos/SAMed-jnar/checkpoints/sam_vit_b_01ec64.pth --lora_ckpt '/home/narvjes/repos/SAMed-jnar/output/Synapse_128_pretrain_vit_b_epo200_bs22_lr0.005_2023-07-30/epoch_199.pth' --output_dir /home/narvjes/repos/SAMed-jnar/output/test --volume_path /home/narvjes/data/PASTIS/SAMed_timeseries --is_savenii > 20230731_PASTIS_test.log 2>&1 &

# Testing for timeseries data, with the custom component checkpoint
python test.py --num_classes 20 --list_dir ./lists/lists_PASTIS_timeseries --img_size 128 --ckpt /home/narvjes/repos/SAMed-jnar/checkpoints/sam_vit_b_01ec64.pth --lora_ckpt '/home/narvjes/repos/SAMed-jnar/output/Synapse_128_pretrain_vit_b_epo250_bs16_lr0.005_2023-09-05/epoch_199.pth' --custom_ckpt '/home/narvjes/repos/SAMed-jnar/output/Synapse_128_pretrain_vit_b_epo250_bs16_lr0.005_2023-09-05/epoch_199_custom.pth' --output_dir /home/narvjes/repos/SAMed-jnar/output/test --volume_path /home/narvjes/data/PASTIS/SAMed_timeseries --is_savenii

# Testing for timeseries data, with the custom component checkpoint and also with full channels normalised
python test.py --num_classes 20 --list_dir ./lists/lists_PASTIS_timeseries_full_channels_normalised --img_size 128 --ckpt /home/narvjes/repos/SAMed-jnar/checkpoints/sam_vit_b_01ec64.pth --lora_ckpt '/home/narvjes/repos/SAMed-jnar/output/Synapse_128_pretrain_vit_b_epo450_bs64_lr0.005_2023-09-19/epoch_199.pth' --custom_ckpt '/home/narvjes/repos/SAMed-jnar/output/Synapse_128_pretrain_vit_b_epo450_bs64_lr0.005_2023-09-19/epoch_199_custom.pth' --output_dir /home/narvjes/repos/SAMed-jnar/output/test --volume_path /home/narvjes/data/PASTIS/SAMed_timeseries_full_channels_normalised --is_savenii
```

Checking Tensorboard
```bash
tensorboard --logdir output/
```

## Acknowledgements
- [SAMed](https://github.com/hitachinsk/SAMed)
- [TS-ViT](https://github.com/michaeltrs/DeepSatModels)
