"""Compare raw and mask postprocessed Mask R-CNN predictions on COCO validation."""

import argparse
import colorsys
import csv
import hashlib
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from train_coco_rgb import CocoRGBDataset, build_model
from coco_utils import convert_to_coco_api


def parse_values(value):
    return [float(item) for item in value.split(',')]


def mask_box(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def pair_overlap(first, second, first_box, second_box, first_area, second_area):
    x1 = max(int(first_box[0]), int(second_box[0]))
    y1 = max(int(first_box[1]), int(second_box[1]))
    x2 = min(int(first_box[2]), int(second_box[2]))
    y2 = min(int(first_box[3]), int(second_box[3]))
    if x1 >= x2 or y1 >= y2:
        return 0.0, 0.0
    intersection = np.logical_and(first[y1:y2, x1:x2], second[y1:y2, x1:x2]).sum()
    iou = intersection / (first_area + second_area - intersection)
    containment = intersection / min(first_area, second_area)
    return float(iou), float(containment)


def mask_nms(scores, binary_masks, score_thr, iou_thr, contain_thr):
    candidate = np.flatnonzero((scores >= score_thr) & binary_masks.any(axis=(1, 2)))
    ranked = candidate[np.argsort(-scores[candidate], kind='stable')]
    areas = binary_masks.reshape(len(binary_masks), -1).sum(axis=1)
    boxes = [mask_box(mask) for mask in binary_masks]
    kept = []
    removed = []
    for index in ranked:
        duplicate = False
        for previous in kept:
            iou, containment = pair_overlap(
                binary_masks[index], binary_masks[previous],
                boxes[index], boxes[previous], areas[index], areas[previous])
            if iou > iou_thr or containment > contain_thr:
                duplicate = True
                break
        (removed if duplicate else kept).append(int(index))
    return kept, removed, len(candidate)


def assign_pixels(probabilities, kept, mask_thr):
    height, width = probabilities.shape[-2:]
    winner = np.full((height, width), -1, dtype=np.int16)
    best_probability = np.full((height, width), -1, dtype=np.float32)
    for index in kept:
        probability = probabilities[index]
        take = (probability >= mask_thr) & (probability > best_probability)
        winner[take] = index
        best_probability[take] = probability[take]
    final_indices = []
    final_masks = []
    for index in kept:
        final_mask = winner == index
        if final_mask.any():
            final_indices.append(index)
            final_masks.append(final_mask)
    return final_indices, final_masks


def overlap_pixels(masks, shape):
    if not masks:
        return 0
    count = np.zeros(shape, dtype=np.uint16)
    for mask in masks:
        count += mask.astype(np.uint16)
    return int(np.count_nonzero(count > 1))


def color(index):
    rgb = colorsys.hsv_to_rgb((index * 0.618033988749895) % 1.0, 0.85, 1.0)
    return tuple(round(channel * 255) for channel in rgb)


def paint(image, masks):
    result = np.asarray(image).copy()
    for index, mask in enumerate(masks):
        result[mask] = color(index)
    return Image.fromarray(result)


def comparison(image, gt_masks, raw_masks, post_masks, title):
    width, height = image.size
    header = 44
    output = Image.new('RGB', (width * 4, height + header), 'white')
    panels = [image, paint(image, gt_masks), paint(image, raw_masks),
              paint(image, post_masks)]
    labels = ['Original', f'GT: {len(gt_masks)}',
              f'Raw: {len(raw_masks)}', f'{title}: {len(post_masks)}']
    draw = ImageDraw.Draw(output)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        output.paste(panel, (index * width, header))
        draw.text((index * width + 12, 12), label, fill='black')
    return output


def records(image_id, scores, labels, boxes, masks):
    bbox_records, segm_records = [], []
    for score, label, box, mask in zip(scores, labels, boxes, masks):
        x1, y1, x2, y2 = map(float, box)
        common = {'image_id': int(image_id), 'category_id': int(label),
                  'score': float(score)}
        bbox_records.append({**common, 'bbox': [x1, y1, x2 - x1, y2 - y1]})
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        segm_records.append({**common,
                             'segmentation': {'size': rle['size'],
                                              'counts': rle['counts'].decode('ascii')}})
    return bbox_records, segm_records


def coco_metrics(gt, bbox_records, segm_records, image_ids):
    metrics = {}
    for kind, detections in [('bbox', bbox_records), ('segm', segm_records)]:
        if detections:
            predicted = gt.loadRes(detections)
        else:
            predicted = COCO()
            predicted.dataset = {'images': gt.dataset['images'],
                                 'categories': gt.dataset['categories'],
                                 'annotations': []}
            predicted.createIndex()
        evaluator = COCOeval(gt, predicted, kind)
        evaluator.params.imgIds = list(image_ids)
        evaluator.params.catIds = gt.getCatIds()
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics[kind] = {'AP': float(evaluator.stats[0]),
                         'AP50': float(evaluator.stats[1]),
                         'AP75': float(evaluator.stats[2])}
    return metrics


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/config_rgb_alllabeled_rgb.yaml')
    parser.add_argument('--checkpoint', default=(
        '/home/huiying/Documents/ITCsegmentation/Code/Output/'
        'maskrcnn_rgb_alllabeled/seed_0/best_model.pt'))
    parser.add_argument('--out-dir', default=(
        '/home/huiying/Documents/ITCsegmentation/Code/Output/'
        'maskrcnn_rgb_alllabeled/seed_0/val_mask_postprocess'))
    parser.add_argument('--score-thresholds', type=parse_values, default=[0.3, 0.5])
    parser.add_argument('--mask-iou-thresholds', type=parse_values, default=[0.5, 0.6])
    parser.add_argument('--containment-threshold', type=float, default=0.8)
    parser.add_argument('--mask-threshold', type=float, default=0.5)
    parser.add_argument('--pixel-assignment', action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dataset = CocoRGBDataset(cfg.ann_val)
    # Match the training evaluator: polygon masks are rasterized by CocoRGBDataset.
    gt = convert_to_coco_api(dataset)
    image_ids = [int(image['id']) for image in dataset.images]
    if set(image_ids) != set(gt.getImgIds()):
        raise ValueError('Dataset and COCO GT image IDs differ')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combinations = list(product(args.score_thresholds, args.mask_iou_thresholds))
    variants = {'raw': {'bbox': [], 'segm': []}}
    for score_thr, iou_thr in combinations:
        tag = f'score_{score_thr:g}_iou_{iou_thr:g}_contain_{args.containment_threshold:g}'
        variants[f'{tag}/mask_nms'] = {'bbox': [], 'segm': []}
        if args.pixel_assignment:
            variants[f'{tag}/mask_nms_pixel'] = {'bbox': [], 'segm': []}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = build_model(cfg.num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    settings = {
        'checkpoint': str(checkpoint_path.resolve()),
        'checkpoint_sha256': file_sha256(checkpoint_path),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'config': str(Path(args.config).resolve()),
        'coco_gt': str(Path(cfg.ann_val).resolve()),
        'coco_gt_sha256': file_sha256(cfg.ann_val),
        'gt_evaluation': 'convert_to_coco_api(CocoRGBDataset), as used during training',
        'image_ids': image_ids,
        'torchvision': torchvision.__version__,
        'model_bbox_score_threshold': model.roi_heads.score_thresh,
        'model_bbox_nms_threshold': model.roi_heads.nms_thresh,
        'model_max_detections_per_image': model.roi_heads.detections_per_img,
        'mask_binarization_threshold': args.mask_threshold,
        'postprocess_score_thresholds': args.score_thresholds,
        'mask_iou_thresholds': args.mask_iou_thresholds,
        'containment_threshold': args.containment_threshold,
        'pixel_assignment': args.pixel_assignment,
        'cocoeval_max_detections': [1, 10, 100],
        'bbox_policy': 'original boxes for raw and mask NMS; recomputed from final mask after pixel assignment',
    }
    save_json(out_dir / 'settings.json', settings)
    rows = []
    annotations_by_image = defaultdict(list)
    for ann in gt.dataset['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    with torch.inference_mode():
        for index, image_info in enumerate(dataset.images):
            model_input, _ = dataset[index]
            output = model([model_input.to(device)])[0]
            scores = output['scores'].detach().cpu().numpy()
            labels = output['labels'].detach().cpu().numpy()
            boxes = output['boxes'].detach().cpu().numpy()
            probabilities = output['masks'].detach().cpu().numpy()[:, 0]
            binary = probabilities >= args.mask_threshold
            image_id = int(image_info['id'])
            height, width = image_info['height'], image_info['width']
            gt_masks = [gt.annToMask(ann).astype(bool)
                        for ann in annotations_by_image[image_id]]
            image_path = Path(image_info['file_name'])
            if not image_path.is_absolute():
                image_path = dataset.root / image_path
            with Image.open(image_path) as source:
                image = source.convert('RGB')
            if image.size != (width, height):
                raise ValueError(f'Image dimensions differ from GT: {image_path}')
            raw_masks = list(binary)
            bbox, segm = records(image_id, scores, labels, boxes, raw_masks)
            variants['raw']['bbox'].extend(bbox)
            variants['raw']['segm'].extend(segm)
            raw_overlap = overlap_pixels(raw_masks, (height, width))

            for score_thr, iou_thr in combinations:
                tag = f'score_{score_thr:g}_iou_{iou_thr:g}_contain_{args.containment_threshold:g}'
                kept, removed, candidate_count = mask_nms(
                    scores, binary, score_thr, iou_thr, args.containment_threshold)
                nms_masks = [binary[item] for item in kept]
                bbox, segm = records(image_id, scores[kept], labels[kept],
                                     boxes[kept], nms_masks)
                variants[f'{tag}/mask_nms']['bbox'].extend(bbox)
                variants[f'{tag}/mask_nms']['segm'].extend(segm)
                nms_overlap = overlap_pixels(nms_masks, (height, width))
                nms_image = comparison(image, gt_masks, raw_masks, nms_masks,
                                       'Mask NMS')
                image_name = f'{image_id}_{image_path.stem}.png'
                nms_path = out_dir / tag / 'mask_nms' / 'images' / image_name
                nms_path.parent.mkdir(parents=True, exist_ok=True)
                nms_image.save(nms_path)
                final_indices, final_masks = kept, nms_masks
                final_overlap = nms_overlap
                if args.pixel_assignment:
                    final_indices, final_masks = assign_pixels(
                        probabilities, kept, args.mask_threshold)
                    final_boxes = np.stack([mask_box(mask) for mask in final_masks]) \
                        if final_masks else np.zeros((0, 4), dtype=np.float32)
                    bbox, segm = records(image_id, scores[final_indices],
                                         labels[final_indices], final_boxes, final_masks)
                    variants[f'{tag}/mask_nms_pixel']['bbox'].extend(bbox)
                    variants[f'{tag}/mask_nms_pixel']['segm'].extend(segm)
                    final_overlap = overlap_pixels(final_masks, (height, width))
                    final_image = comparison(image, gt_masks, raw_masks, final_masks,
                                             'NMS + pixel')
                    final_path = out_dir / tag / 'mask_nms_pixel' / 'images' / image_name
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    final_image.save(final_path)
                rows.append({
                    'image_id': image_id, 'image': str(image_path), 'setting': tag,
                    'gt_count': len(gt_masks), 'raw_count': len(scores),
                    'raw_overlap_pixels': raw_overlap,
                    'score_filtered_or_empty_count': len(scores) - candidate_count,
                    'mask_nms_count': len(kept), 'nms_removed_count': len(removed),
                    'mask_nms_overlap_pixels': nms_overlap,
                    'final_count': len(final_indices),
                    'final_overlap_pixels': final_overlap,
                    'emptied_by_pixel_assignment': len(kept) - len(final_indices),
                })
            print(f'{image_path.name}: GT={len(gt_masks)} raw={len(scores)} '
                  f'raw overlap pixels={raw_overlap}', flush=True)
            del output, scores, labels, boxes, probabilities, binary

    csv_path = out_dir / 'per_image_counts.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {'settings': settings, 'metrics': {}}
    for name, prediction in variants.items():
        variant_dir = out_dir / name
        save_json(variant_dir / 'bbox_predictions.json', prediction['bbox'])
        save_json(variant_dir / 'segm_predictions.json', prediction['segm'])
        print(f'\nEvaluating {name}', flush=True)
        summary['metrics'][name] = coco_metrics(
            gt, prediction['bbox'], prediction['segm'], image_ids)
    save_json(out_dir / 'metrics_summary.json', summary)
    table_path = out_dir / 'metrics_table.csv'
    with table_path.open('w', newline='', encoding='utf-8-sig') as target:
        writer = csv.writer(target)
        writer.writerow(['version', 'bbox_AP', 'bbox_AP50', 'bbox_AP75',
                         'segm_AP', 'segm_AP50', 'segm_AP75'])
        for name, metrics in summary['metrics'].items():
            writer.writerow([name] + [metrics[kind][key]
                                      for kind in ('bbox', 'segm')
                                      for key in ('AP', 'AP50', 'AP75')])
    print(f'\nMetrics: {out_dir / "metrics_summary.json"}')
    print(f'Metrics table: {table_path}')
    print(f'Per-image counts: {csv_path}')
    print('Review each setting images directory for adjacent crown deletions and large masks.')


if __name__ == '__main__':
    main()
