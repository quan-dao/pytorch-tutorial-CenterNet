import torch
import torch.nn.functional as F
from collections import OrderedDict

from .resnet import ResNet18
from .centernet_head import CenterNetHead


def create_model(backbone_type, backbone_outC, nClasses=20, midC=64):
    """Create CenterNet-style model given a backbone type

    Args:
        backbone_type (str): type of backbone (e.g. ResNet18, ResNet34)
        backbone_outC (int): number of channels of backbone's feature map
        nClasses (int): number of classes of objects in the current dataset
        midC (int): number of channels of output of the 1st conv2d in a branch of Head
    Returns:
        torch.nn.Module: a CenterNet-style model
    """
    assert backbone_type.lower() == 'resnet18', "Only support ResNet18 for the moment"
    return torch.nn.Sequential(OrderedDict([
        ('backbone', ResNet18()),
        ('head', CenterNetHead(backbone_outC, nClasses, midC))
    ]))


def load_weights(model, path):
    """Load model's pretrained weights from checkpoint

    Args:
        model (torch.nn.Module): model
        path (str): path to checkpoint
    """
    checkpoint = torch.load(path)

    state_dict = model.state_dict()
    param_names = list(state_dict.keys())

    # Pretrained model's state dict
    pretrained_state_dict = checkpoint['state_dict']
    pretrained_param_names = list(pretrained_state_dict.keys())

    for i, param in enumerate(pretrained_param_names):
        state_dict[param_names[i]] = pretrained_state_dict[param]

    model.load_state_dict(state_dict)
    print('Loaded weights from {}'.format(path))


def idx1d_to_indices3d(idx_1d, nRows, nCols):
    """Convert index in a flattened tensor, which was originally of size (C, nRows, nCols), into its 3D version

    Args:
        idx_1d (torch.Tensor): indices in flattened tensor, shape (nIndices)
        nRows (int): height of the original tensor
        nCols (int): width of the original tensor
    Returns:
        tuple[torch.Tensor]: (channel_idx, row_idx, col_idx), each tensor has shape (nIndices)
    """
    col_idx = ...  # shape (nIndices) - TODO
    row_idx = ...  # shape (nIndices) - TODO
    ch_idx = ...  # shape (nIndices) - TODO
    return ch_idx, row_idx, col_idx


def nms_heat_map(heatmap):
    """Perform non-max suppression on heat map such that only pixels whose
        class probability is higher than its 8 immediately adjacent neighbors'
        are allowed to keep their value. The rest of pixels have their class probability
        set to 0.

    Args:
        heatmap (torch.Tensor): a batch of heatmap, shape (N, nClasses, H, W)
    Return:
        torch.Tensor: non-max suppressed heatmap
    """
    heat_max = F.max_pool2d(heatmap, kernel_size=0,
                            stride=0, padding=0)  # (N, nClasses, H, W) - TODO: declare kernel_size, stride, padding
    heat_keep_mask = (heatmap == heat_max).float()  # (N, nClasses, H, W)
    heat_peak_only = heatmap * heat_keep_mask  # (N, nClasses, H, W)
    # NOTE: `heat_peak_only` has the same size as heatmap, zero everywhere except for peaks where it has the same
    # value as heatmap
    return heat_peak_only


def get_topK_peaks(heatmaps, topK=100):
    """Find the location of topK peaks in nClasses heatmaps predicted for an image

    Args:
        heatmaps (torch.Tensor): nClasses heatmaps predicted for an image, shape (nClasses, H, W)
        topK (int): number of peaks that are kept
    Returns:
        tuple[torch.Tensor]: (channel_idx, row_idx, col_idx, class_prob), each tensor has shape (nPeaks)
    """
    assert len(heatmaps.size()) == 3, "Wrong heatmap size, expected (nClasses, H, W) get {}".format(heatmaps.size())
    # sort the flatten version of heatmaps into descending order
    heat_sorted, indices1d = torch.flatten(heatmaps, start_dim=0).sort(descending=True)  # (nClasses*H*W)
    # take the topK peaks
    indices1d = ...  # shape (topK) - TODO: keep the first `topK` elements in indices1d
    # store class probability (i.e. score) of each peak
    score = ...  # shape (topK) - TODO: keep the first `topK` elements in heat_sorted
    # retrieve (channel, y, x) from indices1d by invoking idx1d_to_indices3d
    chs, ys, xs = idx1d_to_indices3d(...)  # each has shape (topK) - TODO
    return chs, ys, xs, score


@torch.no_grad()
def decode_prediction(prediction, nClasses=20, topK=100):
    """Decode prediction made by a model in CenterNet-family

    Args:
        prediction (torch.Tensor): shape (N, nClasses+4, H, W)
        nClasses (int): number of classes in the dataset
        topK (int): number of centers that are kept for each image
    Returns:
        list[torch.Tensor]: each tensor has shape (nBoxes, 6), each row is (center_x, center_y, w, h, class, score)
    """
    batch_size, H, W = prediction.size(0), prediction.size(2), prediction.size(3)

    # perform NMS on heatmap using MaxPool2d
    heat = nms_heat_map(...)  # (N, nClasses, H, W) - TODO: invoke `nms_heat_map` on heatmap of `prediction`
    # Reminder: heat map is the first nClasses of prediction
    # Note: in `heat`, centers survive in 'heat' as non-zero entries

    batch_boxes = list()  # to store tensor of decoded boxes for each images in the batch
    for im_idx in range(batch_size):
        # TODO: extract top_k peaks from heat[im_idx] which has the shape of (nClasses, H, W)
        chs, ys, xs, score = ...  # TODO

        # retrieve centers
        offset_x = prediction[im_idx, -2, ys, xs]  # (topK)
        offset_y = prediction[...]  # (topK) - TODO: similar to offset_x, extract offset_y from `prediction`
        # TODO: compute the center coordinate according to the first 2 terms of Equation. 4
        # TODO cont: and concatenate the result using torch.cat to have a tensor of size (topK, 2)
        # TODO cont: result is stored in variable `center`
        center = torch.cat(...)  # (topK, 2) TODO
        # Hint: refer to the creation of `this_im_boxes`

        # retrieve size
        w = prediction[...]  # (topK) - TODO: similar to offset_x, extract w from `prediction`
        h = prediction[...]  # (topK) - TODO: similar to offset_x, extract h from `prediction`

        # concatenate boxes
        this_im_boxes = torch.cat([center, w.unsqueeze(1), h.unsqueeze(1), chs.unsqueeze(1),
                                   score.unsqueeze(1)], dim=1)  # (topK, 6)
        batch_boxes.append(this_im_boxes)

    return batch_boxes




