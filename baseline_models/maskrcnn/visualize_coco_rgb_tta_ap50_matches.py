"""Overlay one-to-one AP50 mask matches from TTA prediction JSON on RGB images."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

from coco_utils import convert_to_coco_api
from train_coco_rgb import CocoRGBDataset


# Match both interior fill and contour. A blue/red overlap makes mask boundary
# disagreements visible while the green region shows the shared area.
GT_COLOR = np.array([35, 105, 255], dtype=np.uint8)
PRED_COLOR = np.array([255, 55, 45], dtype=np.uint8)
MATCH_COLOR = np.array([35, 205, 95], dtype=np.uint8)


def mask_iou_matrix(gt_masks, pred_masks):
    if not gt_masks or not pred_masks:
        return np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    gt = np.stack(gt_masks).reshape(len(gt_masks), -1).astype(bool)
    pred = np.stack(pred_masks).reshape(len(pred_masks), -1).astype(bool)
    intersection = gt.astype(np.uint32) @ pred.astype(np.uint32).T
    union = gt.sum(axis=1)[:, None] + pred.sum(axis=1)[None, :] - intersection
    return intersection / np.maximum(union, 1)


def match_masks(iou, threshold):
    """Maximize total IoU one-to-one, then discard pairs below the threshold."""
    if not iou.size:
        return []
    rows, cols = linear_sum_assignment(-iou)
    return [(int(gt_idx), int(pred_idx), float(iou[gt_idx, pred_idx]))
            for gt_idx, pred_idx in zip(rows, cols)
            if iou[gt_idx, pred_idx] >= threshold]


def blend_region(canvas, mask, color, alpha):
    if mask.any():
        canvas[mask] = (canvas[mask].astype(np.float32) * (1 - alpha)
                        + color.astype(np.float32) * alpha).astype(np.uint8)


def draw_contour(canvas, mask, color, width=2):
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    # Extract a one-pixel inner boundary without requiring OpenCV/scikit-image.
    padded = np.pad(mask.astype(bool), 1)
    interior = (padded[1:-1, 1:-1] & padded[:-2, 1:-1]
                & padded[2:, 1:-1] & padded[1:-1, :-2]
                & padded[1:-1, 2:])
    edge = mask.astype(bool) & ~interior
    ys, xs = np.nonzero(edge)
    for x, y in zip(xs, ys):
        draw.ellipse((x - width + 1, y - width + 1,
                      x + width - 1, y + width - 1), fill=tuple(map(int, color)))
    return np.asarray(image)


def render_overlay(image, gt_masks, pred_masks, matches, title, alpha, threshold):
    canvas = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
    matched_gt = {gt_idx for gt_idx, _, _ in matches}
    matched_pred = {pred_idx for _, pred_idx, _ in matches}
    for idx, mask in enumerate(gt_masks):
        blend_region(canvas, mask, MATCH_COLOR if idx in matched_gt else GT_COLOR, alpha)
    for idx, mask in enumerate(pred_masks):
        blend_region(canvas, mask,
                     MATCH_COLOR if idx in matched_pred else PRED_COLOR, alpha)

    # Draw contours last so the shared area remains green while differing
    # silhouettes retain distinct blue (GT) and red (prediction) boundaries.
    for idx, mask in enumerate(gt_masks):
        canvas = draw_contour(canvas, mask, MATCH_COLOR if idx in matched_gt else GT_COLOR)
    for idx, mask in enumerate(pred_masks):
        canvas = draw_contour(canvas, mask,
                              MATCH_COLOR if idx in matched_pred else PRED_COLOR)

    output = Image.fromarray(canvas)
    header = 58
    result = Image.new('RGB', (output.width, output.height + header), 'white')
    result.paste(output, (0, header))
    draw = ImageDraw.Draw(result)
    tp = len(matches)
    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp
    draw.text((10, 7), title, fill='black')
    draw.text((10, 29), f'IoU>={threshold:.2f}   TP={tp}  FP={fp}  FN={fn}',
              fill='black')
    legend_y = 7
    legend_x = max(10, result.width - 340)
    for offset, (label, color) in enumerate((
            ('Matched', MATCH_COLOR), ('GT only', GT_COLOR), ('Pred only', PRED_COLOR))):
        x = legend_x + offset * 108
        draw.rectangle((x, legend_y, x + 12, legend_y + 12), fill=tuple(map(int, color)))
        draw.text((x + 17, legend_y), label, fill='black')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(
        Path(__file__).resolve().parent / 'configs/config_rgb_alllabeled_rgb_dsm.yaml'))
    parser.add_argument('--tta-out-dir', default=(
        '/home/huiying/Documents/ITCsegmentation/Code/Output/'
        'maskrcnn_rgb_dsm_alllabeled/seed_0/val_tta_score_0.3_iou_0.6_contain_0.8'))
    parser.add_argument('--out-dir', default=None,
                        help='Defaults to <tta-out-dir>/ap50_match_visualization')
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.28)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error('--alpha must be between 0 and 1')

    cfg = OmegaConf.load(args.config)
    dataset = CocoRGBDataset(cfg.ann_val)
    gt = convert_to_coco_api(dataset)
    tta_dir = Path(args.tta_out_dir)
    out_dir = Path(args.out_dir) if args.out_dir else tta_dir / 'ap50_match_visualization'
    images_dir = out_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = tta_dir / 'segm_predictions.json'
    if not prediction_path.is_file():
        raise FileNotFoundError(f'TTA segmentation predictions not found: {prediction_path}')
    predictions = json.loads(prediction_path.read_text(encoding='utf-8'))
    predictions_by_image = {}
    for record in predictions:
        predictions_by_image.setdefault(int(record['image_id']), []).append(record)

    annotations_by_image = {}
    for ann in gt.dataset['annotations']:
        annotations_by_image.setdefault(int(ann['image_id']), []).append(ann)

    rows = []
    total_tp = total_fp = total_fn = 0
    for info in dataset.images:
        image_id = int(info['id'])
        gt_masks = [gt.annToMask(ann).astype(bool)
                    for ann in annotations_by_image.get(image_id, [])]
        records = predictions_by_image.get(image_id, [])
        # TTA records are appended in final mask order. Decode COCO RLEs as
        # Fortran-contiguous full-resolution masks.
        pred_masks = [mask_utils.decode(record['segmentation']).astype(bool)
                      for record in records]
        iou = mask_iou_matrix(gt_masks, pred_masks)
        matches = match_masks(iou, args.iou_threshold)

        image_path = Path(info['file_name'])
        if not image_path.is_absolute():
            image_path = dataset.root / image_path
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        if image.size != (info['width'], info['height']):
            raise ValueError(f'Image dimensions differ from GT: {image_path}')
        target = images_dir / f'{image_id}_{image_path.stem}_ap50_match.png'
        render_overlay(image, gt_masks, pred_masks, matches,
                       image_path.name, args.alpha, args.iou_threshold).save(target)

        tp = len(matches)
        fp = len(pred_masks) - tp
        fn = len(gt_masks) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn
        rows.append({
            'image_id': image_id, 'image': str(image_path),
            'gt_count': len(gt_masks), 'prediction_count': len(pred_masks),
            'tp_iou50': tp, 'fp_iou50': fp, 'fn_iou50': fn,
            'mean_matched_iou': (float(np.mean([m[2] for m in matches]))
                                 if matches else ''),
            'visualization': str(target),
        })
        print(f'{image_path.name}: TP={tp} FP={fp} FN={fn} -> {target}', flush=True)

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    summary = {
        'annotation_json': str(Path(cfg.ann_val).resolve()),
        'tta_prediction_json': str(prediction_path.resolve()),
        'iou_match_threshold': args.iou_threshold,
        'matching': 'Hungarian maximum-total-IoU one-to-one assignment, then IoU threshold',
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision_at_iou_threshold': precision,
        'recall_at_iou_threshold': recall,
        'num_images': len(dataset),
        'visualization_dir': str(images_dir.resolve()),
    }
    (out_dir / 'match_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (out_dir / 'per_image_matches.csv').open(
            'w', newline='', encoding='utf-8-sig') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved AP50 match visualizations and summary to {out_dir}', flush=True)


if __name__ == '__main__':
    main()
