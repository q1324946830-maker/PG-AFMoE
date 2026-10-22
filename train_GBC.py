import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations as A
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from albumentations import Normalize, RandomRotate90, Resize
import losses
from model_factory import ARCH_NAMES, build_model
from dataset import Dataset
from dataset_config import get_dataset_paths
from metrics import iou_score, pixel_accuracy
from utils import AverageMeter, str2bool
import time
try:
    from tensorboardX import SummaryWriter
except ImportError:
    class SummaryWriter:
        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

LOSS_NAMES = list(losses.__all__) + ['BCEWithLogitsLoss']


def loss_requires_model(criterion):
    return bool(getattr(criterion, "requires_model", False))


def criterion_forward(criterion, output, target, model):
    if loss_requires_model(criterion):
        return criterion(output, target, model)
    return criterion(losses.unpack_seg_output(output), target)


def as_output_list(outputs):
    if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict):
        return [outputs]
    if isinstance(outputs, list):
        return outputs
    return [outputs]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N', help='mini-batch size(default: 8)')
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--dataseed', default=41, type=int,
                        help='') # default 2981

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default='GBC_Rolling_Unet_S', choices=ARCH_NAMES)
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--input_w', default=256, type=int, help='image width(default: 256)')
    parser.add_argument('--input_h', default=256, type=int, help='image height(default: 256)')
    parser.add_argument('--base_channels', default=32, type=int,
                        help='base channels for the in-project U-Net++ implementation')
    parser.add_argument('--related_models_root',
                        default=os.environ.get('RELATED_MODELS_ROOT', '../Related-Models'),
                        help='directory containing the six external model repositories')

    # loss
    parser.add_argument('--loss', default='BCEDiceWithGeometryLoss', choices=LOSS_NAMES,
                        help='loss: ' + ' | '.join(LOSS_NAMES) + ' (default: BCEDiceWithGeometryLoss)')
    parser.add_argument('--div_weight', type=float, default=0.01, help='Weight of diversity loss')
    parser.add_argument('--scale_weight', type=float, default=0.01, help='Weight of scale consistency loss')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='alpha for focal BCE-Dice loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='gamma for focal BCE-Dice loss')
    parser.add_argument('--hard_pixel_ratio', type=float, default=0.0,
                        help='top hard-pixel ratio used by focal BCE-Dice; 0 disables hard mining')

    # data
    parser.add_argument('--dataset', default='ISIC17',
                        help='dataset name, e.g. BUSI, CVC-ClinicDB, GlaS, ISIC16, ISIC17, ISIC18, PH2')
    parser.add_argument('--img_ext', default='.png', help='deprecated; resolved from dataset registry')
    parser.add_argument('--mask_ext', default='.png', help='deprecated; resolved from dataset registry')

    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR',
                        help='initial learning rate(default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay(default: 1e-4)')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')

    # GBC
    parser.add_argument('--gbc_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate for GBC module (centers and logit_scale)')
    parser.add_argument('--gbc_weight_decay', default=1e-4, type=float,
                        help='weight decay for GBC centers')
    parser.add_argument('--gbc_num_balls', default=32, type=int, help='number of granular balls (shared)')
    parser.add_argument('--gbc_proj_dim', default=0, type=int,
                        help='projection dim for GBC; 0 means use kan_input_dim (auto)')
    parser.add_argument('--use_diag_cov', default=True, type=str2bool, help='Use diagonal covariance (K×D scales)') 
    parser.add_argument('--tau', default=1.0, type=float,
                        help='softmax temperature')                 

    # Frequency-domain skip fusion
    parser.add_argument('--frequency_mode', default='add',
                        choices=['add', 'fft', 'dct', 'dwt', 'laplacian', 'gabor', 'afmoe'],
                        help='frequency fusion mode for skip connections')
    parser.add_argument('--frequency_stages', default='none',
                        help='comma-separated skip stages: none, t3, t3,t2, t3,t2,t1, or all')
    parser.add_argument('--frequency_gate', default=True, type=str2bool,
                        help='use feature-dependent gate in frequency fusion')
    parser.add_argument('--frequency_gamma', default=0.0, type=float,
                        help='initial residual weight for frequency branch')
    parser.add_argument('--frequency_high_ratio', default=0.25, type=float,
                        help='cutoff ratio for FFT/DCT high-frequency branch')
    parser.add_argument('--frequency_experts', default='add,dwt,gabor,dct',
                        help='comma-separated experts for afmoe mode')
    parser.add_argument('--moe_top_k', default=2, type=int,
                        help='number of active experts per pixel in afmoe mode; <=0 uses dense softmax')
    parser.add_argument('--moe_temperature', default=1.0, type=float,
                        help='softmax temperature for afmoe router')
    parser.add_argument('--moe_prior', default='auto',
                        help='afmoe prior, e.g. auto, uniform, or add:0.1,dwt:0.45,gabor:0.25,dct:0.2')
    parser.add_argument('--moe_balance_weight', default=0.01, type=float,
                        help='weight for afmoe expert-load prior regularization')
    parser.add_argument('--moe_entropy_weight', default=0.001, type=float,
                        help='weight for afmoe routing entropy minimization')

    # TorchCV-inspired refinement
    parser.add_argument('--use_align', default=False, type=str2bool,
                        help='enable SFNet-style spatial alignment before skip fusion')
    parser.add_argument('--align_stages', default='none',
                        help='comma-separated aligned skip stages: none, t3, t3,t2, t3,t2,t1, or all')
    parser.add_argument('--align_max_flow', default=4.0, type=float,
                        help='maximum learned alignment displacement in pixels')
    parser.add_argument('--context_module', default='none', choices=['none', 'aspp', 'light_aspp'],
                        help='bottleneck context module')
    parser.add_argument('--context_rates', default='1,2,3',
                        help='dilation rates for the ASPP context module')
    parser.add_argument('--context_dropout', default=0.0, type=float,
                        help='dropout in the ASPP context module')
    parser.add_argument('--strong_aug', default=False, type=str2bool,
                        help='enable stronger Albumentations training augmentation')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N', help='early stopping (default: -1)')

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'accuracy': AverageMeter()}

    model.train()

    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = as_output_list(model(input))
            loss = 0
            for output in outputs:
                loss += criterion_forward(criterion, output, target, model)
            loss /= len(outputs)

            iou, dice = iou_score(outputs[-1], target)
            accuracy = pixel_accuracy(outputs[-1], target)
            
        else:
            output = model(input)
            loss = criterion_forward(criterion, output, target, model)
            iou, dice = iou_score(output, target)
            accuracy = pixel_accuracy(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['accuracy'].update(accuracy, input.size(0))

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('accuracy', avg_meters['accuracy'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'accuracy': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = as_output_list(model(input))
                loss = 0
                for output in outputs:
                    loss += criterion_forward(criterion, output, target, model)

                loss /= len(outputs)
                iou, dice = iou_score(outputs[-1], target)
                accuracy = pixel_accuracy(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion_forward(criterion, output, target, model)
                iou, dice = iou_score(output, target)
                accuracy = pixel_accuracy(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['accuracy'].update(accuracy, input.size(0))

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('accuracy', avg_meters['accuracy'].avg)])


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
    config = vars(parse_args())

    current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
    my_writer = SummaryWriter()

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    dataset_spec, img_dir, mask_dir = get_dataset_paths(config['dataset'])
    config['dataset_input'] = config['dataset']
    config['dataset'] = dataset_spec['name']
    config['dataset_key'] = dataset_spec['key']
    config['dataset_folder'] = dataset_spec['folder']
    config['img_ext'] = dataset_spec['img_ext']
    config['mask_ext'] = dataset_spec['mask_ext']
    config['num_classes'] = dataset_spec['num_classes']

    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif config['loss'] == 'BCEDiceWithGeometryLoss':
        criterion = losses.BCEDiceWithGeometryLoss(
            div_weight=config.get('div_weight'),
            scale_weight=config.get('scale_weight'),
        ).cuda()
    elif config['loss'] == 'FocalBCEDiceWithGeometryLoss':
        criterion = losses.FocalBCEDiceWithGeometryLoss(
            div_weight=config.get('div_weight'),
            scale_weight=config.get('scale_weight'),
            focal_alpha=config.get('focal_alpha'),
            focal_gamma=config.get('focal_gamma'),
            hard_pixel_ratio=config.get('hard_pixel_ratio'),
        ).cuda()
    elif config['loss'] == 'FocalBCEDiceLoss':
        criterion = losses.FocalBCEDiceLoss(
            alpha=config.get('focal_alpha'),
            gamma=config.get('focal_gamma'),
            hard_pixel_ratio=config.get('hard_pixel_ratio'),
        ).cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    gbc_kwargs = {
        'gbc_num_balls': config.get('gbc_num_balls'),
        'gbc_proj_dim': None if config.get('gbc_proj_dim') == 0 else config.get('gbc_proj_dim'),
        'use_diag_cov': config.get('use_diag_cov'),
        'tau': config.get('tau'),
        'frequency_mode': config.get('frequency_mode'),
        'frequency_stages': config.get('frequency_stages'),
        'frequency_gate': config.get('frequency_gate'),
        'frequency_gamma': config.get('frequency_gamma'),
        'frequency_high_ratio': config.get('frequency_high_ratio'),
        'frequency_experts': config.get('frequency_experts'),
        'moe_top_k': config.get('moe_top_k'),
        'moe_temperature': config.get('moe_temperature'),
        'moe_prior': config.get('moe_prior'),
        'moe_balance_weight': config.get('moe_balance_weight'),
        'moe_entropy_weight': config.get('moe_entropy_weight'),
        'use_align': config.get('use_align'),
        'align_stages': config.get('align_stages'),
        'align_max_flow': config.get('align_max_flow'),
        'context_module': config.get('context_module'),
        'context_rates': config.get('context_rates'),
        'context_dropout': config.get('context_dropout'),
    }

    # create model
    model = build_model(
        config['arch'],
        related_models_root=config['related_models_root'],
        base_channels=config['base_channels'],
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision'],
        **gbc_kwargs
    )

    model = model.cuda()

    #params = filter(lambda p: p.requires_grad, model.parameters())
    # === parameter groups: separate kan params and gbc centers if desired ===
    params = []
    gbc_centers_params = []
    others = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        # recognize GBC centers parameter by name 'centers' or 'gbc'
        if 'centers' in lname or 'gbc' in lname:
            gbc_centers_params.append(param)
        else:
            others.append(param)

    if len(others) > 0:
        params.append({'params': others, 'lr': config['lr'], 'weight_decay': config['weight_decay']})
    if len(gbc_centers_params) > 0:
        # give GBC centers same lr as kan (you can change)
        params.append({'params': gbc_centers_params, 'lr': config['gbc_lr'], 'weight_decay': config['gbc_weight_decay']})

    # fallback if nothing matched (shouldn't happen)
    if len(params) == 0:
        params = [{'params': model.parameters(), 'lr': config['lr'], 'weight_decay': config['weight_decay']}]

    if config['optimizer'] == 'Adam':
        # dup_ids = {id(p) for p in others}.intersection({id(p) for p in gbc_centers_params})
        # [(name, id(param)) for name, param in model.named_parameters() if id(param) in dup_ids]
        # print(dup_ids)
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    img_ids = glob(os.path.join(img_dir, '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    if config.get('strong_aug'):
        train_transform = Compose([
            RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=20,
                border_mode=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=0.3,
            ),
            Resize(config['input_h'], config['input_w']),
            Normalize(),
        ])
    else:
        train_transform = Compose([
            RandomRotate90(),
            A.HorizontalFlip(), #transforms.Flip()
            Resize(config['input_h'], config['input_w']),
            Normalize(),
        ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
        pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('accuracy', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_accuracy', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        current_lr = optimizer.param_groups[0]['lr']
        print(
            'Epoch [%d/%d] - %.1fs - lr %.6g - loss %.4f - iou %.4f - accuracy %.4f '
            '- val_loss %.4f - val_iou %.4f - val_dice %.4f - val_accuracy %.4f'
            % (
                epoch + 1, config['epochs'], time.time() - epoch_start, current_lr,
                train_log['loss'], train_log['iou'], train_log['accuracy'],
                val_log['loss'], val_log['iou'], val_log['dice'], val_log['accuracy'],
            ),
            flush=True,
        )

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['accuracy'].append(train_log['accuracy'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_accuracy'].append(val_log['accuracy'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        my_writer.add_scalar('loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('accuracy', train_log['accuracy'], global_step=epoch)
        my_writer.add_scalar('val_loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val_iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val_dice', val_log['dice'], global_step=epoch)
        my_writer.add_scalar('val_accuracy', val_log['accuracy'], global_step=epoch)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print("=> saved best model")
            print('IoU: %.4f' % best_iou)
            print('Dice: %.4f' % best_dice)
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
