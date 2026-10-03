"""Per-image validation report and side-by-side visualization for RGB Mask R-CNN."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_coco_rgb import CocoRGBDataset, build_model


def mask_iou_matrix(pred_masks, gt_masks):
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    pred = pred_masks.reshape(len(pred_masks), -1).astype(bool)
    gt = gt_masks.reshape(len(gt_masks), -1).astype(bool)
    intersection = pred.astype(np.uint32) @ gt.astype(np.uint32).T
    union = pred.sum(1)[:, None] + gt.sum(1)[None, :] - intersection
    return intersection / np.maximum(union, 1)


def average_precision_at_iou(iou, scores, threshold, num_gt):
    if num_gt == 0:
        return float('nan')
    order = np.argsort(-scores)
    matched = set()
    tp, fp = [], []
    for pred_idx in order:
        candidates = np.argsort(-iou[pred_idx]) if iou.shape[1] else []
        match = next((int(j) for j in candidates
                      if iou[pred_idx, j] >= threshold and int(j) not in matched), None)
        if match is None:
            tp.append(0)
            fp.append(1)
        else:
            matched.add(match)
            tp.append(1)
            fp.append(0)
    if not tp:
        return 0.0
    tp = np.cumsum(tp, dtype=np.float64)
    fp = np.cumsum(fp, dtype=np.float64)
    recall = tp / num_gt
    precision = tp / np.maximum(tp + fp, 1)
    # COCO-style 101 point interpolated AP for this image and IoU threshold.
    return float(np.mean([
        precision[recall >= r].max() if np.any(recall >= r) else 0.0
        for r in np.linspace(0, 1, 101)
    ]))


def overlay(image, masks, color, alpha=0.38):
    arr = np.asarray(image.convert('RGB'), dtype=np.float32).copy()
    for mask in masks:
        mask = mask.astype(bool)
        arr[mask] = arr[mask] * (1 - alpha) + np.asarray(color) * alpha
    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)))


def draw_boxes(image, boxes, color, width=3):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for box in boxes:
        draw.rectangle(tuple(float(x) for x in box), outline=color, width=width)
    return canvas


def save_comparison(image_path, gt_masks, pred_masks, gt_boxes, pred_boxes,
                    score_thr, gt_count, pred_count, ap50, ap75, out_path):
    with Image.open(image_path) as src:
        image = src.convert('RGB')
    gt = draw_boxes(overlay(image, gt_masks, (40, 210, 70)), gt_boxes, (20, 150, 40))
    pred = draw_boxes(overlay(image, pred_masks, (255, 90, 40)), pred_boxes, (210, 45, 20))
    panel_w, panel_h = image.size
    header_h = 64
    canvas = Image.new('RGB', (panel_w * 2, panel_h + header_h), 'white')
    canvas.paste(gt, (0, header_h))
    canvas.paste(pred, (panel_w, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f'Ground truth: {gt_count}', fill='black')
    draw.text((panel_w + 12, 8),
              f'Prediction: {pred_count} (score>{score_thr:g})', fill='black')
    ap50_text = 'NA' if np.isnan(ap50) else f'{ap50:.3f}'
    ap75_text = 'NA' if np.isnan(ap75) else f'{ap75:.3f}'
    draw.text((12, 34), f'Per-image AP50: {ap50_text}   AP75: {ap75_text}', fill='black')
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--score-thr', type=float, default=0.3)
    args = parser.parse_args()

    import omegaconf
    cfg = omegaconf.OmegaConf.load(args.config)
    dataset = CocoRGBDataset(cfg.ann_val)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg.num_classes)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / 'images'
    vis_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for idx, image_info in enumerate(dataset.images):
            image_tensor, target = dataset[idx]
            output = model([image_tensor.to(device)])[0]
            scores = output['scores'].detach().cpu().numpy()
            pred_masks_all = (output['masks'].detach().cpu().numpy()[:, 0] >= 0.5)
            keep = scores >= args.score_thr
            pred_masks = pred_masks_all[keep]
            pred_boxes = output['boxes'].detach().cpu().numpy()[keep]
            pred_scores = scores[keep]
            gt_masks = target['masks'].numpy().astype(bool)
            iou = mask_iou_matrix(pred_masks, gt_masks)
            ap50 = average_precision_at_iou(iou, pred_scores, 0.50, len(gt_masks))
            ap75 = average_precision_at_iou(iou, pred_scores, 0.75, len(gt_masks))

            image_path = Path(image_info['file_name'])
            if not image_path.is_absolute():
                image_path = dataset.root / image_path
            name = image_path.stem
            comparison_path = vis_dir / f'{name}_comparison.png'
            save_comparison(image_path, gt_masks, pred_masks,
                            target['boxes'].numpy(), pred_boxes, args.score_thr,
                            len(gt_masks), int(keep.sum()), ap50, ap75,
                            comparison_path)
            row = {
                'image_id': image_info['id'],
                'image': str(image_path),
                'gt_crowns': len(gt_masks),
                'pred_crowns': int(keep.sum()),
                'ap50': '' if np.isnan(ap50) else f'{ap50:.6f}',
                'ap75': '' if np.isnan(ap75) else f'{ap75:.6f}',
                'score_threshold': args.score_thr,
                'comparison': str(comparison_path),
            }
            rows.append(row)
            print(f"{name}: GT={row['gt_crowns']} Pred={row['pred_crowns']} "
                  f"AP50={row['ap50'] or 'NA'} AP75={row['ap75'] or 'NA'}")

    csv_path = out_dir / 'per_image_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved per-image table: {csv_path}')
    print(f'Saved side-by-side images: {vis_dir}')
    print('AP50/AP75 are per-image mask AP using score-filtered predictions and 101-point interpolation.')


if __name__ == '__main__':
    main()
