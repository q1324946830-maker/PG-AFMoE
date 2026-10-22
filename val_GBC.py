import argparse
import os
from glob import glob
import random
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from dataset import Dataset
from dataset_config import get_dataset_paths
from metrics import iou_score, indicators
from model_factory import build_model
from utils import AverageMeter, str2bool
from albumentations import Normalize, RandomRotate90, Resize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='busi_RU_GBC_L_div0.01_sca0.1', help='model name')
    parser.add_argument('--tta', default=False, type=str2bool, help='enable multi-scale and flip TTA')
    parser.add_argument('--tta_scales', default='0.75,1.0,1.25', help='comma-separated TTA scales')
    parser.add_argument('--tta_flip', default=True, type=str2bool, help='enable horizontal flip TTA')
    args = parser.parse_args()
    return args


def parse_scales(scales):
    if isinstance(scales, (list, tuple)):
        values = scales
    else:
        values = [item.strip() for item in str(scales).split(',') if item.strip()]
    parsed = [float(value) for value in values]
    return parsed or [1.0]


def _model_logits(model, input):
    output = model(input)
    if isinstance(output, tuple):
        output = output[0]
    return output


def predict_logits(model, input, use_tta=False, scales='1.0', use_flip=True):
    if not use_tta:
        return _model_logits(model, input)

    original_size = input.shape[-2:]
    logits_sum = None
    count = 0
    for scale in parse_scales(scales):
        if abs(scale - 1.0) < 1e-6:
            scaled = input
        else:
            scaled = F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=False)

        logits = _model_logits(model, scaled)
        logits = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)
        logits_sum = logits if logits_sum is None else logits_sum + logits
        count += 1

        if use_flip:
            flipped = torch.flip(scaled, dims=[-1])
            flip_logits = _model_logits(model, flipped)
            flip_logits = torch.flip(flip_logits, dims=[-1])
            flip_logits = F.interpolate(flip_logits, size=original_size, mode='bilinear', align_corners=False)
            logits_sum = logits_sum + flip_logits
            count += 1

    return logits_sum / max(count, 1)


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_spec, img_dir, mask_dir = get_dataset_paths(config['dataset'])
    config['dataset_input'] = config['dataset']
    config['dataset'] = dataset_spec['name']
    config['dataset_key'] = dataset_spec['key']
    config['dataset_folder'] = dataset_spec['folder']
    config['img_ext'] = dataset_spec['img_ext']
    config['mask_ext'] = dataset_spec['mask_ext']
    config['num_classes'] = dataset_spec['num_classes']

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    gbc_kwargs = {
        'gbc_num_balls': config.get('gbc_num_balls', 32),
        'gbc_proj_dim': None if config.get('gbc_proj_dim', 0) == 0 else config.get('gbc_proj_dim'),
        'use_diag_cov': config.get('use_diag_cov', True),
        'tau': config.get('tau', 1.0),
        'frequency_mode': config.get('frequency_mode', 'add'),
        'frequency_stages': config.get('frequency_stages', 'none'),
        'frequency_gate': config.get('frequency_gate', True),
        'frequency_gamma': config.get('frequency_gamma', 0.0),
        'frequency_high_ratio': config.get('frequency_high_ratio', 0.25),
        'frequency_experts': config.get('frequency_experts', 'add,dwt,gabor,dct'),
        'moe_top_k': config.get('moe_top_k', 2),
        'moe_temperature': config.get('moe_temperature', 1.0),
        'moe_prior': config.get('moe_prior', 'auto'),
        'moe_balance_weight': config.get('moe_balance_weight', 0.0),
        'moe_entropy_weight': config.get('moe_entropy_weight', 0.0),
        'use_align': config.get('use_align', False),
        'align_stages': config.get('align_stages', 'none'),
        'align_max_flow': config.get('align_max_flow', 4.0),
        'context_module': config.get('context_module', 'none'),
        'context_rates': config.get('context_rates', '1,2,3'),
        'context_dropout': config.get('context_dropout', 0.0),
    }
    model = build_model(
        config['arch'],
        related_models_root=config.get('related_models_root'),
        base_channels=config.get('base_channels', 32),
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config.get('deep_supervision', False),
        **gbc_kwargs
    )

    model = model.cuda()

    # Data loading code
    img_ext = dataset_spec['img_ext']
    mask_ext = dataset_spec['mask_ext']
    img_ids = glob(os.path.join(img_dir, '*' + img_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    model.load_state_dict(torch.load('models/%s/model.pth' % args.name))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()
    recall_avg_meter = AverageMeter()
    specificity_avg_meter = AverageMeter()
    precision_avg_meter = AverageMeter()
    accuracy_avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', args.name, str(c)),
                    exist_ok=True)

    with torch.no_grad():
        for input, target, meta in val_loader:
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = predict_logits(
                model,
                input,
                use_tta=args.tta,
                scales=args.tta_scales,
                use_flip=args.tta_flip,
            )

            # iou, dice = iou_score(output, target)
            iou, dice, hd, hd95, recall, specificity, precision, accuracy = indicators(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            hd_avg_meter.update(hd, input.size(0))
            hd95_avg_meter.update(hd95, input.size(0))
            recall_avg_meter.update(recall, input.size(0))
            specificity_avg_meter.update(specificity, input.size(0))
            precision_avg_meter.update(precision, input.size(0))
            accuracy_avg_meter.update(accuracy, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', args.name, str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('Accuracy: %.4f' % accuracy_avg_meter.avg)
    print('Hd: %.4f' % hd_avg_meter.avg)
    print('Hd95: %.4f' % hd95_avg_meter.avg)
    print('Recall: %.4f' % recall_avg_meter.avg)
    print('Specificity: %.4f' % specificity_avg_meter.avg)
    print('Precision: %.4f' % precision_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
