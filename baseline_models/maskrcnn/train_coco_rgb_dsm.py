"""Train a four-channel RGB+DSM Mask R-CNN on COCO polygon annotations."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import omegaconf
import rasterio
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_eval import CocoEvaluator
from coco_utils import convert_to_coco_api


class CocoRGBDSMDataset(Dataset):
    """COCO polygon dataset whose image records contain RGB and DSM paths."""

    def __init__(self, ann_file, dsm_normalization='max', train=False):
        self.ann_file = Path(ann_file)
        self.root = self.ann_file.parent.parent
        coco = json.loads(self.ann_file.read_text(encoding='utf-8'))
        self.images = coco['images']
        self.annotations = {}
        for ann in coco['annotations']:
            self.annotations.setdefault(int(ann['image_id']), []).append(ann)
        self.dsm_normalization = dsm_normalization
        self.train = train

    def __len__(self):
        return len(self.images)

    def _resolve(self, file_name):
        path = Path(file_name)
        return path if path.is_absolute() else self.root / path

    def _read_dsm(self, image_info):
        dsm_path = image_info.get('dsm_path')
        if not dsm_path:
            raise ValueError(f"Image {image_info['id']} has no dsm_path field")
        with rasterio.open(self._resolve(dsm_path)) as source:
            dsm = source.read(1).astype(np.float32)
        if self.dsm_normalization == 'max':
            maximum = float(np.nanmax(dsm)) if dsm.size else 0.0
            dsm = dsm / maximum if maximum > 0 else np.zeros_like(dsm)
        elif self.dsm_normalization == 'minmax':
            minimum, maximum = float(np.nanmin(dsm)), float(np.nanmax(dsm))
            dsm = ((dsm - minimum) / (maximum - minimum)
                   if maximum > minimum else np.zeros_like(dsm))
        else:
            raise ValueError('dsm_normalization must be max or minmax')
        return np.nan_to_num(dsm, nan=0.0, posinf=0.0, neginf=0.0)

    def __getitem__(self, index):
        info = self.images[index]
        image_path = self._resolve(info['file_name'])
        with Image.open(image_path) as source:
            rgb = source.convert('RGB')
        width, height = rgb.size
        dsm = self._read_dsm(info)
        if dsm.shape != (height, width):
            raise ValueError(f'RGB/DSM dimensions differ: {image_path}, {dsm.shape}')

        boxes, masks, areas, crowds = [], [], [], []
        for ann in self.annotations.get(int(info['id']), []):
            segmentation = ann.get('segmentation', [])
            if not isinstance(segmentation, list):
                raise ValueError('Expected polygon-list segmentations in COCO annotations')
            mask_image = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask_image)
            for polygon in segmentation:
                if len(polygon) >= 6:
                    draw.polygon(list(zip(polygon[0::2], polygon[1::2])), fill=1)
            mask = np.asarray(mask_image, dtype=np.uint8)
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            masks.append(mask)
            areas.append(float(mask.sum()))
            crowds.append(int(ann.get('iscrowd', 0)))

        if masks:
            mask_tensor = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            box_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            mask_tensor = torch.zeros((0, height, width), dtype=torch.uint8)
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
        target = {
            'boxes': box_tensor,
            'labels': torch.ones(len(boxes), dtype=torch.int64),
            'masks': mask_tensor,
            'image_id': int(info['id']),
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(crowds, dtype=torch.int64),
        }

        rgb_tensor = TF.to_tensor(rgb)
        dsm_tensor = torch.from_numpy(dsm).unsqueeze(0)
        image_tensor = torch.cat((rgb_tensor, dsm_tensor), dim=0)
        # Match train_coco_rgb.py: 50% horizontal flip for training only.
        # Flip all input channels and the boxes/masks with the same transform.
        if self.train and random.random() < 0.5:
            image_tensor = image_tensor.flip(-1)
            target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]
            target['masks'] = target['masks'].flip(-1)
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    old_conv = model.backbone.body.conv1
    new_conv = torch.nn.Conv2d(
        4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
        stride=old_conv.stride, padding=old_conv.padding, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3].copy_(old_conv.weight)
        new_conv.weight[:, 3:4].zero_()
    model.backbone.body.conv1 = new_conv
    model.transform.image_mean = [0.485, 0.456, 0.406, 0.0]
    model.transform.image_std = [0.229, 0.224, 0.225, 1.0]

    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes)
    return model


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    evaluator = CocoEvaluator(convert_to_coco_api(loader.dataset), ['bbox', 'segm'])
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        predictions = {
            int(target['image_id']): {
                key: value.detach().cpu() for key, value in output.items()
            }
            for target, output in zip(targets, outputs)
        }
        evaluator.update(predictions)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator


def main(config_path):
    cfg = omegaconf.OmegaConf.load(config_path)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = CocoRGBDSMDataset(cfg.ann_train, cfg.dsm_normalization, train=True)
    val_set = CocoRGBDSMDataset(cfg.ann_val, cfg.dsm_normalization)
    test_set = CocoRGBDSMDataset(cfg.ann_test, cfg.dsm_normalization)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate_fn)

    model = build_model(cfg.num_classes).to(device)
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),
                                lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    output = Path(cfg.save_ckpt_path)
    output.mkdir(parents=True, exist_ok=True)
    best_map = float('-inf')
    state = None

    for epoch in range(cfg.num_epochs):
        model.train()
        losses = []
        for images, targets in tqdm(train_loader,
                                    desc=f'train {epoch + 1}/{cfg.num_epochs}'):
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) if isinstance(value, torch.Tensor)
                        else value for key, value in target.items()}
                       for target in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        train_loss = float(np.mean(losses)) if losses else float('nan')
        evaluator = evaluate(model, val_loader, device)
        bbox_map = float(evaluator.coco_eval['bbox'].stats[0])
        segm_map = float(evaluator.coco_eval['segm'].stats[0])
        print(f'epoch={epoch + 1} train_loss={train_loss:.6f} '
              f'val_bbox_mAP={bbox_map:.6f} val_segm_mAP={segm_map:.6f}')
        state = {
            'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss, 'val_bbox_mAP': bbox_map,
            'val_segm_mAP': segm_map,
            'input_channels': ['R', 'G', 'B', 'DSM'],
            'dsm_normalization': cfg.dsm_normalization,
        }
        if (epoch + 1) % cfg.save_every_epochs == 0:
            torch.save(state, output / f'maskrcnn_epoch_{epoch + 1}.pt')
        if segm_map > best_map:
            best_map = segm_map
            torch.save(state, output / 'best_model.pt')

    if state is not None:
        torch.save(state, output / 'last_model.pt')
    print(f'Training complete. Test images available: {len(test_set)}; '
          'run a separate test evaluation after model selection.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python train_coco_rgb_dsm.py CONFIG.yaml')
    main(sys.argv[1])
