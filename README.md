# SAMed
This repository contains the implementation of the following paper:
> **Customized Segment Anything Model for Medical Image Segmentation**<br>
> [Kaidong Zhang](https://hitachinsk.github.io/), and [Dong Liu](https://faculty.ustc.edu.cn/dongeliu/)<br>
> Technical report<br>
[\[Paper\]](https://arxiv.org/pdf/2304.13785.pdf)

Colab online demo: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KCS5ulpZasYl9DgJJn59WsGEB8vwSI_m?usp=sharing)

<img src="materials/teaser.png" height="140px"/> 

## :star: News
- Thanks to the high investment from my supevisor, I can finetune the `vit_h` version of SAM for more accurate medical image segmentation. Now, we release the `vit_h` version of SAMed (We denote this version as SAMed_h), and the comparison between SAMed and SAMed_h is shown in the table below.

Model | DSC | HD | Aorta | Gallbladder | Kidney (L) | Kidney (R) | Liver | Pancreas | Spleen | Stomach
------------ | -------------|-----------|---------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------
SAMed | 81.88 | 20.64 | 87.77 | 69.11 | 80.45 | 79.95 | 94.80 | **72.17** | 88.72 | 82.06
SAMed_h | **84.30** | **16.02** | **87.81** | **74.72** | **85.76** | **81.52** | **95.76** | 70.63 | **90.46** | **87.77**

Without bells and whistles, SAMed_h achieves **much higher performance** than SAMed. Although the model size of `vit_h` version is much larger (above 2G) than `vit_b` version (~350M), the LoRA checkpoint of SAMed_h does not increase a lot (from 18M to 21M). Therefore, **the deployment and storage cost of SAMed_h is nearly on par with SAMed**. Since industry prefers to deploy larger and better performing models, we believe SAMed_h is more promising for computer-assisted diagnosis and preoperative planning in practice. For more details about SAMed_h, please visit [SAMed_h directory](SAMed_h/).

## Overview
<img src="materials/pipeline.png" height="260px"/> 
We propose SAMed, a general solution for medical image segmentation. Different from the previous methods, SAMed is built upon the large-scale image segmentation model, Segment Anything Model (SAM), to explore the new research paradigm of customizing large-scale models for medical image segmentation. SAMed applies the low-rank-based (LoRA) finetuning strategy to the SAM image encoder and finetunes it together with the prompt encoder and the mask decoder on labeled medical image segmentation datasets. We also observe the warmup finetuning strategy and the AdamW optimizer lead SAMed to successful convergence and lower loss. Different from SAM, SAMed could perform semantic segmentation on medical images. Our trained SAMed model achieves 81.88 DSC and 20.64 HD on the Synapse multi-organ segmentation dataset, which is on par with the state-of-the-art methods. We conduct extensive experiments to validate the effectiveness of our design. Since SAMed only updates a small fraction of the SAM parameters, its deployment cost and storage cost are quite marginal in practical usage.

## Todo list
- [ ] Make a demo.
- [ ] Finetune on more datasets
- [ ] ~~Make SAMed based on `vit_l` or `vit_h` mode of SAM~~

## Prerequisites
- Linux (We tested our codes on Ubuntu 18.04)
- Anaconda
- Python 3.7.11
- Pytorch 1.9.1

To get started, first please clone the repo
```
git clone https://github.com/hitachinsk/SAMed.git
```
Then, please run the following commands:
```
conda create -n SAMed python=3.7.11
conda activate SAMed
pip install -r requirements.txt
```

If you have the raw Synapse dataset, we provide the [preprocess script](preprocess/) to process and normalize the data for training. Please refer this folder for more details.

## Quick start
We strongly recommand you to try our online demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KCS5ulpZasYl9DgJJn59WsGEB8vwSI_m?usp=sharing).

Currently, we provide the SAMed and the SAMed_s models for reproducing our results quickly. The LoRA checkpoints and their corresponding configurations are shown in the table below.
Model | Checkpoint | Configuration | DSC | HD
------------ | -------------|-----------|---------------|-------------
SAMed | [Link](https://drive.google.com/file/d/1P0Bm-05l-rfeghbrT1B62v5eN-3A-uOr/view?usp=share_link) | [Cfg](https://drive.google.com/file/d/1pTXpymz3H6665hjztkv-A7uG_rzSWPVg/view?usp=sharing) | 81.88 | 20.64
SAMed_s | [Link](https://drive.google.com/file/d/1rQM2md-h66RlRF3wC0m9N8aheOCvKfYv/view?usp=share_link) | [Cfg](https://drive.google.com/file/d/1x72rB-oNtZ-ZoD_yfOnWdowSb02FMUjT/view?usp=sharing) | 77.78 | 31.72

Here are the instructions: 

1. Change the directory to the rootdir of this repository.
2. Please download the pretrained [SAM model](https://drive.google.com/file/d/1_oCdoEEu3mNhRfFxeWyRerOKt8OEUvcg/view?usp=share_link) (provided by the original repository of SAM) and the [LoRA checkpoint of SAMed](https://drive.google.com/file/d/1P0Bm-05l-rfeghbrT1B62v5eN-3A-uOr/view?usp=share_link). Put them in the `./checkpoints` folder.
3. Please download the [testset](https://drive.google.com/file/d/1RczbNSB37OzPseKJZ1tDxa5OO1IIICzK/view?usp=share_link) and put it in the ./testset folder. Then, unzip and delete this file.
4. Run this commend to test the performance of SAMed.
```bash
python test.py --is_savenii --output_dir <Your output directory>
```
If everything works, you can find the average DSC is 0.8188 (81.88) and HD is 20.64, which correspond to the Tab.1 of the paper. And check the test results in `<Your output directory>`.

What's more, we also provide the [SAMed_s model](https://drive.google.com/file/d/1rQM2md-h66RlRF3wC0m9N8aheOCvKfYv/view?usp=share_link), which utilizes LoRA to finetune the transformer blocks in image encoder and mask decoder. Compared with SAMed, SAMed_s has smaller model size but the performance also drops slightly. If you want to use this model, download and put it in the `./checkpoints_s` folder and run the below command to test its performance.
```bash
python test.py --is_savenii --output_dir <Your output directory> --lora_ckpt checkpoints_s/epoch_159.pth --module sam_lora_image_encoder_mask_decoder
```
The average DSC is 0.7778 (77.78) and HD is 31.72 for SAMed_s, which corresponds to the Tab.3 of the paper. 

## Training
We use 2 RTX 3090 GPUs for training.
1. Please download the processed [training set](https://drive.google.com/file/d/1zuOQRyfo0QYgjcU_uZs0X3LdCnAC2m3G/view?usp=share_link), whose resolution is `224x224`, and put it in `<Your folder>`. Then, unzip and delete this file. We also prepare the [training set](https://drive.google.com/file/d/1F42WMa80UpH98Pw95oAzYDmxAAO2ApYg/view?usp=share_link) with resolution `512x512` for reference, the `224x224` version of training set is downsampled from the `512x512` version.
2. Run this command to train SAMed.
```bash
python train.py --root_path <Your folder> --output <Your output path> --warmup --AdamW 
```
Check the results in `<Your output path>`.

## License
This work is licensed under MIT license. See the [LICENSE](LICENSE) for details.

## Citation
If our work inspires your research or some part of the codes are useful for your work, please cite our paper:
```bibtex
@article{samed,
  title={Customized Segment Anything Model for Medical Image Segmentation},
  author={Kaidong Zhang, and Dong Liu},
  journal={arXiv preprint arXiv:2304.13785},
  year={2023}
}
```

## Contact
If you have any questions, please contact us via 
- richu@mail.ustc.edu.cn

## Acknowledgement
We appreciate the developers of [Segment Anything Model](https://github.com/facebookresearch/segment-anything) and the provider of the [Synapse multi-organ segmentation dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789). The code of SAMed is built upon [TransUnet](https://github.com/Beckschen/TransUNet) and [SAM LoRA](https://github.com/JamesQFreeman/Sam_LoRA), and we express our gratitude to these awesome projects.



# Jesse Updates

> Important Note:
> There are now two manually maintained lists, which keeps track of the custom components that have been added into the model.
> These are `CUSTOM_ITEMS` and `EXCEPTION_LIST`, which are used to ensure that only the relevant component parameters are stored in the .pth export
> and to ensure that the gradients within the custom components added within the Image Encoder do not have their weights frozen, respectively.

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

python test.py --num_classes 20 --list_dir ./lists/lists_PASTIS_timeseries_full_channels_normalised --img_size 128 --ckpt /home/narvjes/repos/SAMed-jnar/checkpoints/sam_vit_b_01ec64.pth --lora_ckpt '/home/narvjes/repos/SAMed-jnar/output/Synapse_128_pretrain_vit_b_epo450_bs64_lr0.005_2023-09-19/epoch_199.pth' --custom_ckpt '/home/narvjes/repos/SAMed-jnar/output/Synapse_128_pretrain_vit_b_epo450_bs64_lr0.005_2023-09-19/epoch_199_custom.pth' --output_dir /home/narvjes/repos/SAMed-jnar/output/test --volume_path /home/narvjes/data/PASTIS/SAMed_timeseries_full_channels_normalised --is_savenii
```

Checking Tensorboard
```bash
tensorboard --logdir output/
```
