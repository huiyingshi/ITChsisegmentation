"""Run inference on the labeled training split and save HSI visualizations.

Each output PNG places ground truth on the left and model prediction on the
right. The checkpoint epoch is inferred from its filename so predictions are
never silently stored under the wrong epoch number.
"""

import argparse
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize predictions on the labeled training split')
    parser.add_argument('config', help='training config file')
    parser.add_argument('checkpoint', help='trusted local training checkpoint')
    parser.add_argument(
        '--work-dir', default=None,
        help='output root; defaults to the checkpoint directory')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='override config values, for example randomness.seed=0')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    return args


def checkpoint_epoch(checkpoint):
    match = re.search(r'epoch_(\d+)', Path(checkpoint).stem)
    if match is None:
        raise ValueError(
            'Checkpoint filename must contain epoch_<number>, for example '
            'epoch_43.pth. This prevents predictions being assigned to the '
            'wrong epoch directory.')
    return int(match.group(1))


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    epoch = checkpoint_epoch(checkpoint)

    # MMEngine checkpoints contain optimizer/history objects in addition to
    # tensors. PyTorch >=2.6 otherwise defaults to weights-only deserialization.
    # This script only accepts a local checkpoint explicitly supplied by user.
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

    setup_cache_size_limit_of_dynamo()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.launcher = args.launcher
    cfg.work_dir = str(Path(args.work_dir).resolve()) if args.work_dir \
        else str(checkpoint.parent)
    cfg.resume = False
    cfg.load_from = str(checkpoint)

    # Reuse the validation pipeline so HSI, DSM and ground-truth masks are
    # loaded exactly as they are during evaluation, but point it at train.json.
    train_eval_dataset = deepcopy(cfg.val_dataloader.dataset)
    train_eval_dataset.ann_file = cfg.train_dataloader.dataset.ann_file
    train_eval_dataset.test_mode = True
    cfg.test_dataloader = deepcopy(cfg.val_dataloader)
    cfg.test_dataloader.dataset = train_eval_dataset
    cfg.test_dataloader.sampler = dict(type='DefaultSampler', shuffle=False)
    cfg.test_evaluator = deepcopy(cfg.val_evaluator)

    output_dir = f'train_predictions/epoch_{epoch:03d}'
    cfg.custom_hooks = [
        dict(
            type='HSITestVisualizationHook',
            output_dir=output_dir,
            rgb_bands=(76, 51, 26),
            score_thr=args.score_thr,
            include_timestamp=False,
        )
    ]

    runner = Runner.from_cfg(cfg) if 'runner_type' not in cfg \
        else RUNNERS.build(cfg)
    metrics = runner.test()
    print(f'Train prediction PNGs: {Path(cfg.work_dir) / output_dir}')
    print(f'Train metrics: {metrics}')


if __name__ == '__main__':
    main()
