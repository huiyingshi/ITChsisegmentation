"""Evaluate a checkpoint on train.json and plot its COCO PR curves."""

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot bbox and mask PR curves on the labeled train split')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    return args


def epoch_from_checkpoint(path):
    match = re.search(r'epoch_(\d+)', Path(path).stem)
    if match is None:
        raise ValueError('Checkpoint name must contain epoch_<number>.')
    return int(match.group(1))


def extract_curves(coco_gt, prediction_file, iou_type):
    coco_dt = coco_gt.loadRes(str(prediction_file))
    evaluator = COCOeval(coco_gt, coco_dt, iou_type)
    evaluator.params.maxDets = [100, 300, 1000]
    evaluator.evaluate()
    evaluator.accumulate()

    # precision dimensions: IoU, recall, category, area range, max detections.
    precision = evaluator.eval['precision'][:, :, 0, 0, -1]
    precision = np.where(precision < 0, np.nan, precision)
    recall = evaluator.params.recThrs
    mean_curve = np.nanmean(precision, axis=0)
    curves = {
        'AP50': precision[0],
        'AP75': precision[5],
        'AP50:95': mean_curve,
    }
    aps = {name: float(np.nanmean(values)) for name, values in curves.items()}
    return recall, curves, aps


def plot_curves(curves_by_type, output_file, epoch):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    colors = {'AP50': '#2b6cb0', 'AP75': '#dd6b20', 'AP50:95': '#2f855a'}
    for axis, (iou_type, (recall, curves, aps)) in zip(
            axes, curves_by_type.items()):
        for name in ('AP50', 'AP75', 'AP50:95'):
            axis.step(
                recall, curves[name], where='post', linewidth=2,
                color=colors[name], label=f'{name} = {aps[name]:.3f}')
        axis.set_title('Bounding-box PR' if iou_type == 'bbox' else 'Mask PR')
        axis.set_xlabel('Recall')
        axis.grid(True, alpha=0.25)
        axis.legend(loc='lower left')
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
    axes[0].set_ylabel('Precision')
    fig.suptitle(f'Train-set Precision–Recall Curves — Epoch {epoch}')
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    epoch = epoch_from_checkpoint(checkpoint)
    output_dir = (Path(args.output_dir).resolve() if args.output_dir else
                  checkpoint.parent / 'train_pr_curves' / f'epoch_{epoch:03d}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # The explicitly supplied local checkpoint is a complete trusted MMEngine
    # checkpoint and contains HistoryBuffer/optimizer state as well as tensors.
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    setup_cache_size_limit_of_dynamo()

    cfg = Config.fromfile(args.config)
    cfg.resume = False
    cfg.load_from = str(checkpoint)
    cfg.work_dir = str(output_dir / 'runner')
    cfg.custom_hooks = []
    train_eval_dataset = deepcopy(cfg.val_dataloader.dataset)
    train_eval_dataset.ann_file = cfg.train_dataloader.dataset.ann_file
    train_eval_dataset.test_mode = True
    cfg.test_dataloader = deepcopy(cfg.val_dataloader)
    cfg.test_dataloader.dataset = train_eval_dataset
    cfg.test_dataloader.sampler = dict(type='DefaultSampler', shuffle=False)
    cfg.test_evaluator = deepcopy(cfg.val_evaluator)
    prediction_prefix = output_dir / 'predictions'
    cfg.test_evaluator.outfile_prefix = str(prediction_prefix)

    runner = Runner.from_cfg(cfg) if 'runner_type' not in cfg else RUNNERS.build(cfg)
    metrics = runner.test()

    annotation_file = Path(cfg.data_root) / train_eval_dataset.ann_file
    coco_gt = COCO(str(annotation_file))
    curves_by_type = {
        'bbox': extract_curves(coco_gt, output_dir / 'predictions.bbox.json', 'bbox'),
        'segm': extract_curves(coco_gt, output_dir / 'predictions.segm.json', 'segm'),
    }
    plot_file = output_dir / f'train_pr_epoch_{epoch:03d}.png'
    plot_curves(curves_by_type, plot_file, epoch)

    summary = {
        iou_type: values[2] for iou_type, values in curves_by_type.items()
    }
    (output_dir / 'pr_summary.json').write_text(
        json.dumps({'metrics': metrics, 'curves': summary}, indent=2),
        encoding='utf-8')
    print(f'PR curve: {plot_file}')


if __name__ == '__main__':
    main()
