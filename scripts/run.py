#!/usr/bin/env python


import datetime
import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F


from atten import generate_atten
from odise.modeling.meta_arch.ldm import LdmImplicitCaptionerExtractor
from odise.modeling.backbone.feature_extractor import FeatureExtractorBackbone
from utils import load_cityscapes_dataloader, UnsupervisedMetrics

warnings.simplefilter('ignore')
torch.set_printoptions(precision=3, sci_mode=False)

now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d-%H-%M")
dir_for_output = "/workspace/maskdiffusion/outputs/" + current_time
os.makedirs(dir_for_output, exist_ok=True)
print(dir_for_output)


RUN_KMEANS = True

model = FeatureExtractorBackbone(
    feature_extractor=LdmImplicitCaptionerExtractor(
        encoder_block_indices=(5, 7),
        unet_block_indices=(2, 5, 8, 11),
        decoder_block_indices=(2, 5),
        steps=(0,),
        learnable_time_embed=True,
        num_timesteps=1,
        clip_model_name="ViT-L-14-336",
    ),
    out_features=["s2", "s3", "s4", "s5"],
    use_checkpoint=False,
    slide_training=True,
).cuda()
checkpoint = torch.load("model_weights/feature_extractor.pth")
model.load_state_dict(checkpoint, strict=False)

val_dataloader = load_cityscapes_dataloader(is_val=True, batch_size=1)
convert27to19 = torch.tensor([0, 1, -1, -1, 2, 3, 4, -1, -1, -1, 5, -1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, 16, 17, 18, -1])
classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "light", "sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
colors = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]])
colors = colors[:,::-1]

semseg_metrics = UnsupervisedMetrics("test/maskdiffusion/", 19, 0, False)

def generate_semseg_from_feat_atten(feat, atten):
    "mean of all pixels"
    feat, atten = feat.cuda(), atten.cuda()
    B, C, H, W = atten.shape
    class2feat = []
    feat = F.normalize(feat, dim=1, eps=1e-10)
    for i in range(C):
        class2feat.append((feat * atten[:,i,:,:] / atten[:,i,:,:].sum(dim=-1).sum(dim=-1).sum(dim=0)).sum(dim=-1).sum(dim=-1).sum(dim=0))
    class2feat = torch.stack(class2feat)
    segmentation_map = torch.einsum('bfhw,cf->bchw', feat, class2feat)
    return segmentation_map

from sklearn.cluster import KMeans
def image_segmentation(feature, num_clusters=20, image_id=0):
    if len(feature.shape) == 4:
        f = feature[0].cpu().numpy()
    elif len(feature.shape) == 3:
        f = feature.cpu().numpy()
    else:
        raise ValueError("feature shape is invalid")

    d, height, width = f.shape

    pixels = f.transpose(1, 2, 0).reshape(-1, d).astype(float)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_pred = kmeans.fit_predict(pixels)
    segmented_image = np.reshape(cluster_pred, (height, width))

    return segmented_image

for idx, (images, targets) in enumerate(val_dataloader):
    images = images.cuda()
    targets = convert27to19[targets].cuda()
    with torch.no_grad():
        features = model(images)

    feats = []
    for stage, feat in features.items():
        if stage in ["s4", "s5"]:
            feats.append(F.interpolate(feat, size=(256, 352), mode="bilinear", align_corners=False))
    feats = torch.cat(feats, dim=1).cuda()
    
    resize_img = F.interpolate(images, size=(512, 512), mode="bilinear", align_corners=False) * 2 - 1
    atten, self_attention = generate_atten(resize_img, classes)
    
    prob = generate_semseg_from_feat_atten(feats, atten)

    semseg_metrics.update(prob.argmax(1), targets)
    print(idx, semseg_metrics.compute())
    
    if RUN_KMEANS:
        kmeans_img = image_segmentation(feats, num_clusters=len(targets.unique()), image_id=idx)

    if idx % 10 == 0:
        print(dir_for_output)
        if RUN_KMEANS:
            cv2.imwrite(f"{dir_for_output}/kmeans_{idx}.png", colors[kmeans_img])
        a = atten.argmax(1)[0].cpu().numpy()
        cv2.imwrite(f"{dir_for_output}/atten_seg_{idx}.png", colors[a])
        p = prob.argmax(1)[0].cpu().numpy()
        cv2.imwrite(f"{dir_for_output}/semseg_{idx}.png", colors[p])
        rgb_img = np.array(torch.permute(images[0], (1, 2, 0)).cpu())[:,:,::-1]
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min()) * 255
        cv2.imwrite(f"{dir_for_output}/input_{idx}.png", rgb_img)
        pred = targets[0].cpu().numpy()
        cv2.imwrite(f"{dir_for_output}/targets_{idx}.png", colors[pred])
