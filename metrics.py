import torch

try:
    from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision
except ImportError:
    jc = dc = hd = hd95 = recall = specificity = precision = None


def iou_score(output, target):
    ### --- 新增：解包元组 --- ###
    # 检查 output 是否为元组 (在训练时，模型会返回 (seg_output, loss_intermediates))
    # 评估指标只关心 seg_output
    if isinstance(output, tuple):
        output = output[0]
    ### --- 修改结束 --- ###
    
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou, dice


def dice_coef(output, target):
    if isinstance(output, tuple):
        output = output[0]

    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def pixel_accuracy(output, target):
    """Return binary pixel accuracy after applying the standard 0.5 threshold."""
    if isinstance(output, tuple):
        output = output[0]

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5
    return (output_ == target_).mean()


def indicators(output, target):
    if jc is None:
        raise ImportError("medpy is required for validation indicators. Install it with `pip install medpy`.")

    if isinstance(output, tuple):
        output = output[0]
        
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    hd_ = hd(output_, target_)
    hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)
    accuracy_ = (output_ == target_).mean()

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_, accuracy_
