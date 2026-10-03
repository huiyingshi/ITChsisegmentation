"""Train RGB Mask R-CNN on standard COCO polygon annotations."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import omegaconf
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_eval import CocoEvaluator
from coco_utils import convert_to_coco_api


class CocoRGBDataset(Dataset):
    def __init__(self, ann_file, train=False):
        self.ann_file = Path(ann_file)
        self.root = self.ann_file.parent.parent
        coco = json.loads(self.ann_file.read_text(encoding='utf-8'))
        self.images = coco['images']
        self.annotations = {}
        for ann in coco['annotations']:
            self.annotations.setdefault(ann['image_id'], []).append(ann)
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = Path(image_info['file_name'])
        if not image_path.is_absolute():
            image_path = self.root / image_path
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        width, height = image.size

        boxes, masks, labels, areas, crowds = [], [], [], [], []
        for ann in self.annotations.get(image_info['id'], []):
            segmentation = ann.get('segmentation', [])
            polygons = segmentation if isinstance(segmentation, list) else []
            mask_image = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask_image)
            for polygon in polygons:
                if len(polygon) >= 6:
                    draw.polygon(list(zip(polygon[0::2], polygon[1::2])), fill=1)
            mask = np.asarray(mask_image, dtype=np.uint8)
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            masks.append(mask)
            labels.append(1)  # Mask R-CNN reserves label 0 for background.
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
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': mask_tensor,
            'image_id': int(image_info['id']),
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(crowds, dtype=torch.int64),
        }

        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]
            target['masks'] = target['masks'].flip(-1)

        image_tensor = TF.to_tensor(image)
        mean = image_tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = image_tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        return (image_tensor - mean) / std, target


def collate_fn(batch):
    return tuple(zip(*batch))


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


def build_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes)
    return model


def main(config_path):
    cfg = omegaconf.OmegaConf.load(config_path)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = CocoRGBDataset(cfg.ann_train, train=True)
    val_set = CocoRGBDataset(cfg.ann_val)
    test_set = CocoRGBDataset(cfg.ann_test)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate_fn)

    model = build_model(cfg.num_classes).to(device)
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),
                                lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    output = Path(cfg.save_ckpt_path)
    output.mkdir(parents=True, exist_ok=True)
    best_map = float('-inf')

    for epoch in range(cfg.num_epochs):
        model.train()
        losses = []
        for images, targets in tqdm(train_loader, desc=f'train {epoch + 1}/{cfg.num_epochs}'):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in target.items()} for target in targets]
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

        state = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'train_loss': train_loss, 'val_bbox_mAP': bbox_map,
                 'val_segm_mAP': segm_map}
        if (epoch + 1) % cfg.save_every_epochs == 0:
            torch.save(state, output / f'maskrcnn_epoch_{epoch + 1}.pt')
        if segm_map > best_map:
            best_map = segm_map
            torch.save(state, output / 'best_model.pt')

    torch.save(state, output / 'last_model.pt')
    print(f'Training complete. Test images available: {len(test_loader.dataset)}; '
          'run inference/evaluation on test separately to avoid tuning on test data.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python train_coco_rgb.py CONFIG.yaml')
    main(sys.argv[1])
