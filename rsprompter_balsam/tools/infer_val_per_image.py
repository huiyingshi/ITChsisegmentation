"""Run an RSPrompter checkpoint on val and report per-image crown metrics."""

import argparse
import csv
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image, ImageDraw


def polygon_to_mask(segmentation, height, width):
    if not segmentation:
        return np.zeros((height, width), dtype=np.uint8)
    rles = mask_utils.frPyObjects(segmentation, height, width)
    decoded = mask_utils.decode(rles)
    if decoded.ndim == 2:
        return decoded.astype(np.uint8)
    return np.any(decoded, axis=2).astype(np.uint8)


def decode_rle(rle):
    if isinstance(rle.get('counts'), str):
        rle = dict(rle, counts=rle['counts'].encode('ascii'))
    return mask_utils.decode(rle).astype(bool)


def mask_ious(pred_masks, gt_masks):
    if not pred_masks or not gt_masks:
        return np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    pred = np.stack(pred_masks).reshape(len(pred_masks), -1)
    gt = np.stack(gt_masks).reshape(len(gt_masks), -1)
    inter = pred.astype(np.uint32) @ gt.astype(np.uint32).T
    union = pred.sum(1)[:, None] + gt.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1)


def per_image_ap(ious, scores, threshold, gt_count):
    if gt_count == 0:
        return float('nan')
    matched = set()
    tp, fp = [], []
    for pred_idx in np.argsort(-scores):
        candidates = np.argsort(-ious[pred_idx]) if ious.shape[1] else []
        match = next((int(j) for j in candidates
                      if ious[pred_idx, j] >= threshold and int(j) not in matched), None)
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
    recall = tp / gt_count
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.mean([
        precision[recall >= r].max() if np.any(recall >= r) else 0.0
        for r in np.linspace(0, 1, 101)
    ]))


def rle_union(masks, shape):
    if not masks:
        return np.zeros(shape, dtype=bool)
    return np.logical_or.reduce([decode_rle(rle) for rle in masks])


def draw_panel(image, masks, boxes, color):
    canvas = np.asarray(image.convert('RGB'), dtype=np.float32).copy()
    for mask in masks:
        canvas[mask] = canvas[mask] * 0.62 + np.asarray(color) * 0.38
    result = Image.fromarray(np.uint8(np.clip(canvas, 0, 255)))
    draw = ImageDraw.Draw(result)
    for box in boxes:
        draw.rectangle(tuple(float(v) for v in box), outline=color, width=3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rsprompter/rsprompter_anchor-trees-pca3-dsm-alllabeled.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='Confidence cutoff used for the reported prediction count and overlay.')
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    config = Path(args.config)
    if not config.is_absolute():
        config = project_dir / config
    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_file = out_dir / 'rsprompter_val_predictions.pkl'

    command = [
        sys.executable, str(project_dir / 'tools' / 'test.py'), str(config),
        '--checkpoint', str(checkpoint),
        '--work-dir', str(out_dir / 'mmdet_work'),
        '--out', str(dump_file),
        '--cfg-options',
        'test_dataloader.dataset.ann_file=annotations/val.json',
    ]
    print('Running RSPrompter validation inference:')
    print(' '.join(command))
    subprocess.run(command, cwd=project_dir, check=True)

    data_root = Path('/home/huiying/Documents/ITCsegmentation/Data/DatasetNew/processed/pca_alllabeled_1024_stride256_png')
    ann_file = data_root / 'annotations' / 'val.json'
    coco = __import__('json').loads(ann_file.read_text(encoding='utf-8'))
    gt_by_id = {}
    for image in coco['images']:
        anns = [ann for ann in coco['annotations'] if ann['image_id'] == image['id']]
        h, w = image['height'], image['width']
        masks = [polygon_to_mask(ann['segmentation'], h, w).astype(bool)
                 for ann in anns]
        gt_by_id[image['id']] = (image, anns, masks)

    with dump_file.open('rb') as f:
        samples = pickle.load(f)
    pred_by_id = {}
    for sample in samples:
        image_id = sample.get('img_id')
        pred = sample.get('pred_instances')
        if hasattr(image_id, 'item'):
            image_id = image_id.item()
        pred_by_id[int(image_id)] = {
            'scores': np.asarray(pred.get('scores')),
            'labels': np.asarray(pred.get('labels')),
            'boxes': np.asarray(pred.get('bboxes')),
            'masks': list(pred.get('masks')),
        }

    vis_dir = out_dir / 'images'
    vis_dir.mkdir(exist_ok=True)
    rows = []
    for image_id, (image_info, anns, gt_masks) in gt_by_id.items():
        prediction = pred_by_id.get(image_id, {
            'scores': np.zeros(0), 'labels': np.zeros(0),
            'boxes': np.zeros((0, 4)), 'masks': [],
        })
        scores = prediction['scores']
        keep = scores >= args.score_thr
        all_pred_masks = [decode_rle(rle) for rle in prediction['masks']]
        ious = mask_ious(all_pred_masks, gt_masks)
        ap50 = per_image_ap(ious, scores, 0.50, len(gt_masks))
        ap75 = per_image_ap(ious, scores, 0.75, len(gt_masks))
        pred_masks = [mask for mask, selected in zip(all_pred_masks, keep) if selected]
        pred_boxes = prediction['boxes'][keep]
        gt_boxes = [ann['bbox'] for ann in anns]
        gt_xyxy = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in gt_boxes]

        image_path = Path(image_info['file_name'])
        if not image_path.is_absolute():
            image_path = data_root / image_path
        with Image.open(image_path) as src:
            image = src.convert('RGB')
        left = draw_panel(image, gt_masks, gt_xyxy, (30, 190, 70))
        right = draw_panel(image, pred_masks, pred_boxes, (230, 70, 40))
        canvas = Image.new('RGB', (image.width * 2, image.height + 62), 'white')
        canvas.paste(left, (0, 62))
        canvas.paste(right, (image.width, 62))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f'Ground truth: {len(gt_masks)} crowns', fill='black')
        draw.text((image.width + 12, 8),
                  f'Prediction: {int(keep.sum())} crowns (score >= {args.score_thr:g})',
                  fill='black')
        ap50_text = 'NA' if np.isnan(ap50) else f'{ap50:.3f}'
        ap75_text = 'NA' if np.isnan(ap75) else f'{ap75:.3f}'
        draw.text((12, 34), f'Per-image mask AP50: {ap50_text}   AP75: {ap75_text}', fill='black')
        vis_path = vis_dir / f"{Path(image_path).stem}_comparison.png"
        canvas.save(vis_path)

        row = {
            'image_id': image_id,
            'image': str(image_path),
            'gt_crowns': len(gt_masks),
            'pred_crowns': int(keep.sum()),
            'ap50': '' if np.isnan(ap50) else f'{ap50:.6f}',
            'ap75': '' if np.isnan(ap75) else f'{ap75:.6f}',
            'score_threshold': args.score_thr,
            'comparison': str(vis_path),
        }
        rows.append(row)
        print(f"{Path(image_path).name}: GT={row['gt_crowns']} "
              f"Pred={row['pred_crowns']} AP50={row['ap50'] or 'NA'} "
              f"AP75={row['ap75'] or 'NA'}")

    csv_path = out_dir / 'per_image_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nPer-image results: {csv_path}')
    print(f'Side-by-side comparisons: {vis_dir}')
    print('Per-image AP uses all model predictions; score threshold affects predicted counts and overlays.')


if __name__ == '__main__':
    main()
