# Copyright (c) OpenMMLab. All rights reserved.
import sys
import argparse
import os
import os.path as osp
import csv
import glob
import gc
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "mmdet")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "mmdet/rsprompter")))
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.dist import is_main_process

from mmdet.utils import setup_cache_size_limit_of_dynamo
#from mmdet.datasets import ConcatDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _find_final_checkpoints(work_dir):
    """Return the best checkpoint and the final epoch checkpoint."""
    best_candidates = glob.glob(osp.join(work_dir, 'best_*.pth'))
    if not best_candidates:
        raise FileNotFoundError(
            f'No best_*.pth checkpoint was produced in {work_dir}. '
            'Check that validation ran and default_hooks.checkpoint.save_best '
            'matches a metric emitted by val_evaluator.')
    best_checkpoint = max(best_candidates, key=osp.getmtime)

    last_file = osp.join(work_dir, 'last_checkpoint')
    if osp.isfile(last_file):
        with open(last_file, encoding='utf-8') as f:
            last_checkpoint = f.read().strip()
        if not osp.isabs(last_checkpoint):
            last_checkpoint = osp.join(work_dir, last_checkpoint)
    else:
        epoch_candidates = glob.glob(osp.join(work_dir, 'epoch_*.pth'))
        if not epoch_candidates:
            raise FileNotFoundError(f'No final epoch checkpoint found in {work_dir}.')
        last_checkpoint = max(
            epoch_candidates,
            key=lambda p: int(re.search(r'epoch_(\d+)', osp.basename(p)).group(1))
        )

    if not osp.isfile(last_checkpoint):
        raise FileNotFoundError(
            f'last_checkpoint points to a missing file: {last_checkpoint}')
    return best_checkpoint, last_checkpoint


def _evaluate_checkpoint(cfg, checkpoint, role):
    """Evaluate one checkpoint with visualization disabled and return metrics."""
    eval_cfg = cfg.copy()
    eval_cfg.resume = False
    eval_cfg.load_from = checkpoint
    # Metrics-only pass avoids creating duplicate prediction image folders.
    eval_cfg.custom_hooks = []

    eval_runner = Runner.from_cfg(eval_cfg)
    print(f'Evaluating {role} checkpoint: {checkpoint}')
    metrics = eval_runner.test()
    del eval_runner
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return metrics


def _write_evaluation_csv(work_dir, evaluations):
    """Write best/last metrics in one spreadsheet-friendly CSV."""
    output_path = osp.join(work_dir, 'test_metrics_best_and_last.csv')
    rows = []
    metric_names = set()
    for role, checkpoint, metrics in evaluations:
        match = re.search(r'epoch_(\d+)', osp.basename(checkpoint))
        row = {
            'checkpoint_role': role,
            'epoch': int(match.group(1)) if match else '',
            'checkpoint': osp.basename(checkpoint),
        }
        row.update(metrics)
        metric_names.update(metrics)
        rows.append(row)

    fieldnames = ['checkpoint_role', 'epoch', 'checkpoint'] + sorted(metric_names)
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Best/last test metrics saved to: {output_path}')

def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    print("SEED", cfg.randomness.seed)
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    cfg.work_dir = osp.join(cfg.work_dir, 'seed_' + str(cfg.randomness.seed))
    os.makedirs(cfg.work_dir, exist_ok=True)

    # Keep only the best checkpoint and the latest epoch checkpoint. This
    # also prunes older epoch files when resuming an existing run.
    if cfg.get('default_hooks') and cfg.default_hooks.get('checkpoint'):
        cfg.default_hooks.checkpoint.max_keep_ckpts = 1
    #cfg.vis_backends.0.init_kwargs.id="rsprompter-anchor-trees-imgxdsm-" + str(cfg.randomness.seed)
    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        print("Resuming from last checkpoint")
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        print(f"Resuming from checkpoint {args.resume}")
        cfg.resume = True
        cfg.load_from = args.resume

    
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
        
    #get numpber of parameters of model
    num_params = count_parameters(runner.model)
    print("Number of trainable parameters: ", num_params)

    # start training
    print("start training")
    runner.train()

    # Release the training model before building evaluation runners, so the
    # best and last checkpoints can be tested sequentially on the same GPU.
    work_dir = runner.work_dir
    del runner
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    best_checkpoint, last_checkpoint = _find_final_checkpoints(work_dir)
    evaluations = []
    for role, checkpoint in (('best', best_checkpoint), ('last', last_checkpoint)):
        metrics = _evaluate_checkpoint(cfg, checkpoint, role)
        if is_main_process():
            evaluations.append((role, checkpoint, metrics))
    if is_main_process():
        _write_evaluation_csv(work_dir, evaluations)

if __name__ == '__main__':
    main()
