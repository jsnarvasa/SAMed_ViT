import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_synapse import Synapse_dataset
from sklearn.metrics import confusion_matrix

from icecream import ic


# class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach', 7: 'aorta', 8: 'pancreas'}
class_to_name = {
    0: 'background',
    1: 'meadow',
    2: 'soft winter wheat',
    3: 'corn',
    4: 'winter barley',
    5: 'winter rapeseed',
    6: 'spring barley',
    7: 'sunflower',
    8: 'grapevine',
    9: 'beet',
    10: 'winter triticale',
    11: 'winter durum wheat',
    12: 'fruits, veges flowers',
    13: 'potatoes',
    14: 'leguminous fodder',
    15: 'soybeans',
    16: 'orchard',
    17: 'mixed cereal',
    18: 'sorghum',
    19: 'void'
}


def confusion_mat(predicted, labels, n_classes):  # , unk_masks=None):
    """
                predicted
            -----------------
    labels |
    """
    #if unk_masks is not None:
    #    predicted = predicted[unk_masks]
    #    labels = labels[unk_masks]
    cm = confusion_matrix(labels, predicted)
    # cm_side = cm.shape[0]
    rem = 0
    if cm.shape[0] < n_classes:
        batch_classes = np.unique(np.concatenate((predicted, labels))).tolist()
        for i in range(n_classes):
            # internal class missing
            if (i - rem) < len(batch_classes):
                if i < batch_classes[i-rem]:
                    cm = np.insert(cm, i, 0., axis=0)
                    cm = np.insert(cm, i, 0., axis=1)
                    rem += 1
                    i += 1
            # outer class(es) missing
            else:
                diff = n_classes - rem - len(batch_classes)  # + 1
                cm_side = cm.shape[0]
                cm = np.concatenate((cm, np.zeros((diff, cm_side))), axis=0)
                cm = np.concatenate((cm, np.zeros((cm_side + diff, diff))), axis=1)
                break
    return cm


def get_prediction_splits(predicted, labels, n_classes):  # , unk_masks=None):  , per_class=False):
    # if per_class:
    #     TP, FP, TN, FN = get_prediction_metrics(predicted, labels, unk_mask=unk_mask, per_class=True)
    # else:
    #     TP, FP, TN, FN = get_prediction_metrics(predicted, labels, unk_mask=unk_mask, per_class=False)
    
    # TN = (allsum - rowsum - colsum + diag).astype(np.float32)
    # ---------------------------------------------------------
    cm = confusion_mat(predicted, labels, n_classes).astype(np.float32)
    diag = np.diagonal(cm)
    rowsum = cm.sum(axis=1)
    colsum = cm.sum(axis=0)
    # allsum = cm.sum()
    TP = (diag).astype(np.float32)
    FN = (rowsum - diag).astype(np.float32)
    FP = (colsum - diag).astype(np.float32)
    IOU = diag / (rowsum + colsum - diag)
    micro_IOU = diag.sum() / (rowsum.sum() + colsum.sum() - diag.sum())
    # ---------------------------------------------------------
    #if unk_masks is not None:
    #    predicted = predicted[unk_masks]
    #    labels = labels[unk_masks]
    num_total = []
    num_correct = []
    for class_ in range(n_classes):
        idx = labels == class_
        is_correct = predicted[idx] == labels[idx]
        if is_correct.size == 0:
            is_correct = np.array(0)
        num_total.append(idx.sum())
        num_correct.append(is_correct.sum())   # previously was .mean()
    num_total = np.array(num_total).astype(np.float32)
    num_correct = np.array(num_correct)
    #if not per_class:
    #    return TP.sum(), FP.sum(), FN.sum(), num_correct.sum(), num_total.sum(), IOU[~np.isnan(IOU)].mean()
    return TP, FP, FN, num_correct, num_total, IOU, micro_IOU


def inference(args, multimask_output, db_config, model, num_classes, test_save_path=None):
    db_test = db_config['Dataset'](base_dir=args.volume_path, list_dir=args.list_dir, split='pastis_test_vol')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_list = 0.0
    label_list = []
    prediction_list = []
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        t, c, h, w = sampled_batch['image'].shape[1:]
        image, label, doy, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['doy'], sampled_batch['case_name'][0]
        metric_i, label_i, prediction_i = test_single_volume(image, label, doy, model, classes=args.num_classes, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.input_size, args.input_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=db_config['z_spacing'])
        metric_list += np.array(metric_i)
        label_list.append(label_i)
        prediction_list.append(prediction_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)

    # Use the label_list and prediction_list to generate the confusion matrix
    label_agg = torch.cat(label_list, dim=0).cpu()
    prediction_agg = torch.cat(prediction_list, dim=0).cpu()

    TP, FP, FN, num_correct, num_total, IOU, micro_IOU = get_prediction_splits(prediction_agg, label_agg, num_classes)
    macro_IOU = IOU[~np.isnan(IOU)].mean()


    for i in range(1, args.num_classes + 1):
        try:
            logging.info('Mean class %d name %s mean_dice %f mean_hd95 %f' % (i, class_to_name[i], metric_list[i - 1][0], metric_list[i - 1][1]))
        except:
            logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    logging.info("Testing Finished!")
    logging.info(f'Micro IOU: {str(micro_IOU)}, Macro_IOU: {str(macro_IOU)}')
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--volume_path', type=str, default='testset/test_vol_h5/')
    parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='/output')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--custom_ckpt', type=str, default=None, help='Path to custom checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='checkpoints/epoch_159.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                    pixel_std=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                    custom_checkpoint=args.custom_ckpt)
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, multimask_output, dataset_config[dataset_name], net, args.num_classes, test_save_path)
