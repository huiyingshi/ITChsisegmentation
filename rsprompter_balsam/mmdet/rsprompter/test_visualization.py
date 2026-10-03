"""Save HSI validation and test predictions as RGB comparison images."""

from pathlib import Path

import numpy as np
import rasterio
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class HSITestVisualizationHook(Hook):
    """Draw ground truth (left) and predicted crowns (right) during evaluation.

    RGB bands are one-based rasterio indices, approximately 650/550/450 nm
    for the current 164-band HSI imagery.
    """

    def __init__(self, output_dir='test_predictions',
                 val_output_dir='val_predictions',
                 rgb_bands=(76, 51, 26), score_thr=0.3,
                 include_timestamp=True):
        self.output_dir = output_dir
        self.val_output_dir = val_output_dir
        self.rgb_bands = tuple(rgb_bands)
        self.score_thr = score_thr
        self.include_timestamp = include_timestamp

    def _output_root(self, runner):
        root = Path(runner.work_dir)
        return root / runner.timestamp if self.include_timestamp else root

    def _read_rgb(self, img_path):
        with rasterio.open(img_path) as src:
            if src.count < max(self.rgb_bands):
                raise ValueError(
                    f'{img_path}: RGB preview requires band '
                    f'{max(self.rgb_bands)}, found {src.count}')
            bands = src.read(self.rgb_bands, masked=True).astype(np.float32)

        values = np.ma.filled(bands, np.nan)
        valid = np.all(np.isfinite(values), axis=0)
        height, width = values.shape[1:]
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        for channel in range(3):
            channel_values = values[channel][valid]
            if channel_values.size == 0:
                continue
            low, high = np.percentile(channel_values, (2, 98))
            if high <= low:
                high = low + 1.0
            stretched = np.clip((values[channel] - low) / (high - low), 0, 1)
            rgb[..., channel][valid] = np.round(
                stretched[valid] * 255).astype(np.uint8)
        return rgb

    def before_val_epoch(self, runner):
        self.val_save_dir = (self._output_root(runner) /
                             self.val_output_dir / f'epoch_{runner.epoch:03d}')
        self.val_save_dir.mkdir(parents=True, exist_ok=True)

    def after_val_iter(self, runner, batch_idx, data_batch, outputs):
        self._save_outputs(runner, outputs, self.val_save_dir)

    def before_test(self, runner):
        self.test_save_dir = self._output_root(runner) / self.output_dir
        self.test_save_dir.mkdir(parents=True, exist_ok=True)

    def after_test_iter(self, runner, batch_idx, data_batch, outputs):
        self._save_outputs(runner, outputs, self.test_save_dir)

    def _save_outputs(self, runner, outputs, save_dir):
        for sample in outputs:
            img_path = sample.metainfo['img_path']
            img_id = sample.metainfo['img_id']
            rgb = self._read_rgb(img_path)
            out_file = save_dir / f'{Path(img_path).stem}_id{img_id}.png'
            runner.visualizer.add_datasample(
                name=out_file.stem,
                image=rgb,
                data_sample=sample,
                draw_gt=True,
                draw_pred=True,
                pred_score_thr=self.score_thr,
                out_file=str(out_file),
            )
