"""Six-view TTA for Mask R-CNN, followed by mask NMS and pixel ownership.

Views: original, rotations of 90/180/270 degrees counterclockwise, horizontal
flip, and vertical flip. Torchvision returns full-image mask probabilities, not
raw mask-head logits. We convert them to logits, undo each spatial transform,
match same-class instances by mask IoU, and average matched logits.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from coco_utils import convert_to_coco_api
from infer_coco_rgb_postprocess import (
    assign_pixels, coco_metrics, mask_box, mask_nms, paint, records, save_json,
)
from train_coco_rgb_dsm import CocoRGBDSMDataset, build_model


VIEWS = ('original', 'rot90', 'rot180', 'rot270', 'hflip', 'vflip')


def transform(tensor, view, inverse=False):
    if view.startswith('rot'):
        quarter_turns = int(view[3:]) // 90
        return torch.rot90(tensor, -quarter_turns if inverse else quarter_turns,
                           dims=(-2, -1))
    if view == 'hflip':
        return tensor.flip(-1)
    if view == 'vflip':
        return tensor.flip(-2)
    return tensor


def mask_iou(first, second, first_box, second_box, first_area, second_area):
    x1 = max(int(first_box[0]), int(second_box[0]))
    y1 = max(int(first_box[1]), int(second_box[1]))
    x2 = min(int(first_box[2]), int(second_box[2]))
    y2 = min(int(first_box[3]), int(second_box[3]))
    if x1 >= x2 or y1 >= y2:
        return 0.0
    intersection = np.count_nonzero(first[y1:y2, x1:x2] & second[y1:y2, x1:x2])
    return intersection / (first_area + second_area - intersection)


def add_view(clusters, output, view, score_threshold, match_iou, mask_threshold):
    scores = output['scores'].detach().cpu().numpy()
    labels = output['labels'].detach().cpu().numpy()
    indices = np.flatnonzero(scores >= score_threshold)
    indices = indices[np.argsort(-scores[indices], kind='stable')]
    for index in indices:
        # Full-image probability is the only mask value exposed by torchvision.
        probability = transform(output['masks'][int(index), 0].detach().cpu(),
                                view, inverse=True).float()
        binary = (probability >= mask_threshold).numpy()
        if not binary.any():
            continue
        box = mask_box(binary)
        area = int(binary.sum())
        best = None
        best_iou = match_iou
        for cluster in clusters:
            if cluster['label'] != int(labels[index]) or view in cluster['views']:
                continue
            iou = mask_iou(binary, cluster['reference'], box,
                           cluster['box'], area, cluster['area'])
            if iou >= best_iou:
                best_iou, best = iou, cluster
        logits = torch.logit(probability.clamp(1e-4, 1 - 1e-4))
        if best is None:
            clusters.append({
                'label': int(labels[index]), 'score_sum': float(scores[index]),
                'logit_sum': logits, 'count': 1, 'views': {view},
                'reference': binary, 'box': box, 'area': area,
            })
        else:
            best['logit_sum'].add_(logits)
            best['score_sum'] += float(scores[index])
            best['count'] += 1
            best['views'].add(view)


def merge_clusters(clusters, image_shape):
    height, width = image_shape
    scores, labels, probabilities = [], [], []
    for cluster in clusters:
        probability = torch.sigmoid(cluster['logit_sum'] / cluster['count']).numpy()
        scores.append(cluster['score_sum'] / cluster['count'])
        labels.append(cluster['label'])
        probabilities.append(probability)
    if probabilities:
        return (np.asarray(scores, dtype=np.float32),
                np.asarray(labels, dtype=np.int64), np.stack(probabilities))
    return (np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64),
            np.empty((0, height, width), dtype=np.float32))


def triptych(image, gt_masks, final_masks):
    width, height = image.size
    header = 44
    result = Image.new('RGB', (3 * width, height + header), 'white')
    panels = (paint(image, gt_masks), image, paint(image, final_masks))
    labels = (f'GT: {len(gt_masks)}', 'Original',
              f'TTA + NMS + pixel: {len(final_masks)}')
    draw = ImageDraw.Draw(result)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        result.paste(panel, (index * width, header))
        draw.text((index * width + 12, 12), label, fill='black')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(
        Path(__file__).resolve().parent / 'configs/config_rgb_alllabeled_rgb_dsm.yaml'))
    parser.add_argument('--checkpoint', default=(
        '/home/huiying/Documents/ITCsegmentation/Code/Output/'
        'maskrcnn_rgb_dsm_alllabeled/seed_0/best_model.pt'))
    parser.add_argument('--out-dir', default=(
        '/home/huiying/Documents/ITCsegmentation/Code/Output/'
        'maskrcnn_rgb_dsm_alllabeled/seed_0/val_tta_score_0.3_iou_0.6_contain_0.8'))
    parser.add_argument('--score-threshold', type=float, default=0.3)
    parser.add_argument('--match-iou', type=float, default=0.5,
                        help='Mask IoU used to match instances across TTA views')
    parser.add_argument('--nms-iou', type=float, default=0.6)
    parser.add_argument('--containment', type=float, default=0.8)
    parser.add_argument('--mask-threshold', type=float, default=0.5)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dataset = CocoRGBDSMDataset(cfg.ann_val, cfg.dsm_normalization)
    gt = convert_to_coco_api(dataset)
    annotations_by_image = defaultdict(list)
    for ann in gt.dataset['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    image_ids = [int(info['id']) for info in dataset.images]
    if set(image_ids) != set(gt.getImgIds()):
        raise ValueError('Dataset and COCO GT image IDs differ')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = build_model(cfg.num_classes)
    checkpoint_channels = checkpoint['model_state_dict']['backbone.body.conv1.weight'].shape[1]
    if checkpoint_channels != 4:
        raise ValueError(
            f'This RGB+DSM script expects a 4-channel checkpoint, got '
            f'{checkpoint_channels} channels: {args.checkpoint}')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    out_dir = Path(args.out_dir)
    images_dir = out_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'config': str(Path(args.config).resolve()),
        'coco_gt': str(Path(cfg.ann_val).resolve()),
        'input_channels': 4,
        'dsm_normalization': cfg.dsm_normalization,
        'views': VIEWS, 'score_threshold': args.score_threshold,
        'tta_match_mask_iou': args.match_iou,
        'mask_nms_iou': args.nms_iou,
        'mask_nms_containment': args.containment,
        'mask_binarization_threshold': args.mask_threshold,
        'fusion': 'same-class mask IoU matching; mean reconstructed logits and scores over matched views',
        'bbox_policy': 'recomputed from final non-overlapping masks',
        'cocoeval_max_detections': [1, 10, 100],
    }
    save_json(out_dir / 'settings.json', settings)
    bbox_records, segm_records, rows = [], [], []

    with torch.inference_mode():
        for index, info in enumerate(dataset.images):
            model_input, _ = dataset[index]
            height, width = info['height'], info['width']
            clusters = []
            for view in VIEWS:
                augmented = transform(model_input, view).contiguous().to(device)
                output = model([augmented])[0]
                add_view(clusters, output, view, args.score_threshold,
                         args.match_iou, args.mask_threshold)
                del augmented, output
            scores, labels, probabilities = merge_clusters(clusters, (height, width))
            del clusters
            binary = probabilities >= args.mask_threshold
            kept, removed, _ = mask_nms(scores, binary, args.score_threshold,
                                        args.nms_iou, args.containment)
            final_indices, final_masks = assign_pixels(
                probabilities, kept, args.mask_threshold)
            final_boxes = np.stack([mask_box(mask) for mask in final_masks]) \
                if final_masks else np.empty((0, 4), dtype=np.float32)
            image_id = int(info['id'])
            bbox, segm = records(image_id, scores[final_indices],
                                 labels[final_indices], final_boxes, final_masks)
            bbox_records.extend(bbox)
            segm_records.extend(segm)

            image_path = Path(info['file_name'])
            if not image_path.is_absolute():
                image_path = dataset.root / image_path
            with Image.open(image_path) as source:
                image = source.convert('RGB')
            if image.size != (width, height):
                raise ValueError(f'Image dimensions differ from GT: {image_path}')
            gt_masks = [gt.annToMask(ann).astype(bool)
                        for ann in annotations_by_image[image_id]]
            triptych(image, gt_masks, final_masks).save(
                images_dir / f'{image_id}_{image_path.stem}.png')
            rows.append({'image_id': image_id, 'image': str(image_path),
                         'gt_count': len(gt_masks), 'tta_merged_count': len(scores),
                         'nms_removed_count': len(removed),
                         'final_count': len(final_indices)})
            print(f'{image_path.name}: GT={len(gt_masks)} '
                  f'TTA merged={len(scores)} final={len(final_indices)}',
                  flush=True)

    save_json(out_dir / 'bbox_predictions.json', bbox_records)
    save_json(out_dir / 'segm_predictions.json', segm_records)
    with (out_dir / 'per_image_counts.csv').open('w', newline='', encoding='utf-8-sig') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = coco_metrics(gt, bbox_records, segm_records, image_ids)
    summary = {
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'annotation_json': str(Path(cfg.ann_val).resolve()),
        'evaluation_split': 'cfg.ann_val',
        'num_images': len(image_ids),
        'num_ground_truth_instances': len(gt.dataset['annotations']),
        'num_predictions_after_tta_nms_pixel': len(segm_records),
        'postprocessing': {
            'views': list(VIEWS),
            'score_threshold': args.score_threshold,
            'tta_match_mask_iou': args.match_iou,
            'mask_nms_iou': args.nms_iou,
            'mask_nms_containment': args.containment,
            'mask_threshold': args.mask_threshold,
            'pixel_assignment': True,
        },
        'metrics': metrics,
    }
    save_json(out_dir / 'metrics.json', summary)
    with (out_dir / 'metrics_table.csv').open(
            'w', newline='', encoding='utf-8-sig') as target:
        writer = csv.writer(target)
        writer.writerow(['bbox_AP', 'bbox_AP50', 'bbox_AP75',
                         'segm_AP', 'segm_AP50', 'segm_AP75'])
        writer.writerow([metrics[kind][key]
                         for kind in ('bbox', 'segm')
                         for key in ('AP', 'AP50', 'AP75')])
    print(f'Images and metrics saved to {out_dir}', flush=True)


if __name__ == '__main__':
    main()
