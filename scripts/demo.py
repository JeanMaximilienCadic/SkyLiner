# Copyright 2020 Toyota Research Institute.  All rights reserved.

# This script provides a demo inference a model trained on Cityscapes dataset.
import warnings
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision.models.detection.image_list import ImageList
from gnutools.utils import listfiles, parent
from realtime_panoptic.models.rt_pano_net import RTPanoNet
from realtime_panoptic.config import cfg
import realtime_panoptic.data.panoptic_transform as P
from realtime_panoptic.utils.visualization import visualize_segmentation_image,visualize_detection_image
import os
from tqdm import tqdm

cityscapes_colormap_sky = np.array([[  0,   0,   0] if k!=10 else [ 70, 130, 180] for k in range(20)])
warnings.filterwarnings("ignore", category=UserWarning)


def inference(model, input, transform, device="cuda"):
    input_image = Image.open(input)
    data = {'image': input_image}
    # data pre-processing
    data = transform(data)
    with torch.no_grad():
        input_image_list = ImageList([data['image'].to(device)], image_sizes=[input_image.size[::-1]])
        panoptic_result, _ = model.forward(input_image_list)
        semseg_logics = [o.to('cpu') for o in panoptic_result["semantic_segmentation_result"]]
        # Export the result
        output = input.replace("/data/", "/output/")
        os.makedirs(parent(output), exist_ok=True)
        assert os.path.exists(parent(output))
        semseg_prob = [torch.argmax(semantic_logit, dim=0) for semantic_logit in semseg_logics]
        seg_vis = visualize_segmentation_image(semseg_prob[0], input_image, cityscapes_colormap_sky)
        Image.fromarray(seg_vis.astype('uint8')).save(output)


def main(args):
    # General config object from given config files.
    cfg.merge_from_file(args.config_file)

    # Initialize model.
    model = RTPanoNet(
        backbone=cfg.model.backbone, 
        num_classes=cfg.model.panoptic.num_classes,
        things_num_classes=cfg.model.panoptic.num_thing_classes,
        pre_nms_thresh=cfg.model.panoptic.pre_nms_thresh,
        pre_nms_top_n=cfg.model.panoptic.pre_nms_top_n,
        nms_thresh=cfg.model.panoptic.nms_thresh,
        fpn_post_nms_top_n=cfg.model.panoptic.fpn_post_nms_top_n,
        instance_id_range=cfg.model.panoptic.instance_id_range)
    device = args.device
    model.to(device)
    model.load_state_dict(torch.load(args.pretrained_weight))

    # Print out mode architecture for sanity checking.
    print(model)

    # Prepare for model inference.
    model.eval()
    normalize_transform = P.Normalize(mean=cfg.input.pixel_mean, std=cfg.input.pixel_std, to_bgr255=cfg.input.to_bgr255)
    transform = P.Compose([
        P.ToTensor(),
        normalize_transform,
    ])

    inputs = listfiles(args.root, [f".{args.ext}"])
    for input in tqdm(inputs, desc="Processing files") :
        inference(model, input, transform, args.device)


if __name__ == "__main__":
    # Parse the input arguments.
    parser = argparse.ArgumentParser(description="Simple demo for real-time-panoptic model")
    parser.add_argument("--config-file", metavar="FILE", help="path to config", required=True)
    parser.add_argument("--pretrained-weight", metavar="FILE", help="path to pretrained_weight", required=True)
    parser.add_argument("--root", metavar="FILE", help="path to root folder jpg/png file", required=True)
    parser.add_argument("--device", help="inference device", default='cuda')
    parser.add_argument("--ext", help="image fileformat", default='jpg')
    args = parser.parse_args()
    main(args)
