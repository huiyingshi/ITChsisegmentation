"""Export the three-channel HSI-adapter output for every Test image.

The PNG is a display-only rendering of the adapter's [3, H, W] output before
DSM fusion and before SAM.  Use --save-npy when the unscaled float32 values
are also needed for analysis.
"""
import argparse
import os
import sys
from pathlib import Path

# Match tools/train.py and tools/test.py: make the repository-local mmdet
# package importable when this script is invoked from the tools directory.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import RUNNERS
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Save Test HSI-adapter outputs.')
    parser.add_argument('config', help='Training config path.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint .pth path.')
    parser.add_argument('--out-dir', required=True, help='Directory for PNG outputs.')
    parser.add_argument('--save-npy', action='store_true',
                        help='Also save unscaled [3,H,W] float32 arrays.')
    parser.add_argument('--percentiles', type=float, nargs=2, default=(2.0, 98.0),
                        metavar=('LOW', 'HIGH'), help='Display stretch percentiles.')
    return parser.parse_args()


def render_rgb(adapter_output: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Apply one joint robust stretch to preserve the three-channel contrast."""
    image = np.moveaxis(adapter_output, 0, -1)
    finite = np.isfinite(image)
    values = image[finite]
    if values.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(values, (low_pct, high_pct))
    if high <= low:
        high = low + 1e-6
    return np.round(np.clip((image - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.resume = False
    cfg.load_from = str(checkpoint)
    cfg.work_dir = str(out_dir)
    # Adapter rendering is a direct forward pass, not an evaluator run.
    cfg.custom_hooks = []
    runner = Runner.from_cfg(cfg) if 'runner_type' not in cfg else RUNNERS.build(cfg)

    # PyTorch >= 2.6 changed torch.load's default to weights_only=True.
    # MMEngine checkpoints also contain trusted training-state metadata (for
    # example HistoryBuffer), so a locally produced checkpoint otherwise fails
    # before its model state can be read. Limit this compatibility override to
    # this user-supplied, local visualization checkpoint.
    original_torch_load = torch.load

    def load_trusted_local_checkpoint(*load_args, **load_kwargs):
        load_kwargs.setdefault('weights_only', False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = load_trusted_local_checkpoint
    runner.load_or_resume()
    model = runner.model
    if not hasattr(model, 'hsi_adapter') or model.hsi_adapter is None:
        raise RuntimeError('The configured model has no enabled hsi_adapter.')
    model.eval()

    count = 0
    with torch.no_grad():
        for batch in runner.test_dataloader:
            processed = model.data_preprocessor(batch, False)
            hsi_inputs = processed['inputs'][0]
            adapter_outputs = model.hsi_adapter(hsi_inputs).float().cpu().numpy()
            for output, sample in zip(adapter_outputs, processed['data_samples']):
                stem = Path(sample.metainfo['img_path']).stem
                image_id = sample.metainfo['img_id']
                name = f'{stem}_id{image_id}_adapter_rgb'
                Image.fromarray(render_rgb(output, *args.percentiles)).save(out_dir / f'{name}.png')
                if args.save_npy:
                    np.save(out_dir / f'{name}.npy', output.astype(np.float32))
                count += 1

    (out_dir / 'README.txt').write_text(
        'Each PNG is a joint 2–98 percentile RGB rendering of the HSI adapter output '
        'before DSM fusion and SAM. The optional .npy file retains raw [3,H,W] float32 values.\n',
        encoding='utf-8')
    print(f'Saved {count} adapter-output PNGs to: {out_dir}')


if __name__ == '__main__':
    main()
