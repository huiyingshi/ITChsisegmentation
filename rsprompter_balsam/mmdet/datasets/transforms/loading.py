# NOTICE: This file is a modification of a file from the 
#  RSPrompter project, which is licensed under Apache 2.0.
# The original file can be found at: https://github.com/KyanChen/RSPrompter/blob/release/mmdet/datasets/transforms/loading.py
# All modifications are Copyright (c) 2025 Mélisande Teng.
# Original Copyright (c) OpenMMLab.
# -------------------------------------------------------------------------

from typing import Optional, Tuple, Union
import warnings
from typing import Optional
import mmengine.fileio as fileio
import tifffile
import rasterio
import numpy as np
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile as MMCV_LoadImageFromFile
from mmengine.fileio import get
from mmengine.structures import BaseDataElement

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import get_box_type
from mmdet.structures.bbox.box_type import autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks


@TRANSFORMS.register_module()
class LoadHSIFromFile(BaseTransform):
    """Load a multiband GeoTIFF as float32 HWC without RGB conversion.

    HSI-Adapter keeps the spectral channels and passes float HWC arrays to
    the packing transform. Here a single fixed scale factor is used for all
    bands so their relative amplitudes remain intact. This is not a
    reflectance calibration or the percentile stretch used for RGB previews.
    """

    def __init__(self,
                 expected_channels: int = 164,
                 scale_factor: float = 10000.0,
                 nodata_value: float = 65535.0) -> None:
        if expected_channels <= 0 or scale_factor <= 0:
            raise ValueError('expected_channels and scale_factor must be positive')
        self.expected_channels = expected_channels
        self.scale_factor = float(scale_factor)
        self.nodata_value = nodata_value

    def transform(self, results: dict) -> dict:
        filename = results['img_path']
        with rasterio.open(filename) as src:
            if src.count != self.expected_channels:
                raise ValueError(
                    f'{filename}: expected {self.expected_channels} HSI bands, '
                    f'found {src.count}')
            cube = src.read(out_dtype='float32')  # C, H, W
            nodata = src.nodata if src.nodata is not None else self.nodata_value

        valid = np.ones(cube.shape[1:], dtype=bool)
        for band in cube:
            valid &= np.isfinite(band)
            if nodata is not None:
                valid &= band != nodata
        cube[:, ~valid] = 0
        cube /= self.scale_factor
        img = np.moveaxis(cube, 0, -1)  # H, W, C

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['hsi_valid_fraction'] = float(valid.mean())
        return results


@TRANSFORMS.register_module()
class LoadDSMFromRasterio(BaseTransform):
    """Load the aligned single-band float DSM, replacing nodata with zero."""

    def __init__(self, to_float32: bool = True) -> None:
        if not to_float32:
            raise ValueError('LoadDSMFromRasterio requires float32 DSM inputs')

    def transform(self, results: dict) -> dict:
        filename = results['dsm_path']
        with rasterio.open(filename) as src:
            if src.count != 1:
                raise ValueError(f'{filename}: expected one DSM band, found {src.count}')
            dsm = src.read(1, masked=True).filled(0).astype(np.float32)
        dsm[~np.isfinite(dsm)] = 0
        if 'img_shape' in results and dsm.shape != tuple(results['img_shape'][:2]):
            raise ValueError(
                f'{filename}: DSM shape {dsm.shape} does not match '
                f'HSI shape {results["img_shape"][:2]}')
        results['dsm'] = dsm
        results['dsm_shape'] = dsm.shape
        return results


@TRANSFORMS.register_module()
class HSIDSMRandomGeoAug(BaseTransform):
    """Synchronised flip and scale augmentation for HSI, DSM and instances.

    This transform is deliberately separate from MMDetection's standard
    ``RandomFlip``/``RandomResize``: those transforms update RGB-style image
    data, boxes and masks, but do not know about this project's ``dsm`` key.
    Applying them independently would silently misalign elevation and HSI.

    The output canvas remains the original tile size.  A scale greater than
    one is randomly cropped; a scale below one is randomly zero-padded.
    Instance masks use nearest-neighbour resampling and boxes are rebuilt from
    the transformed masks, avoiding coordinate-rounding inconsistencies.
    """

    def __init__(self,
                 horizontal_prob: float = 0.5,
                 vertical_prob: float = 0.5,
                 scale_prob: float = 0.7,
                 scale_range: Tuple[float, float] = (0.85, 1.15)) -> None:
        for value, name in ((horizontal_prob, 'horizontal_prob'),
                            (vertical_prob, 'vertical_prob'),
                            (scale_prob, 'scale_prob')):
            if not 0 <= value <= 1:
                raise ValueError(f'{name} must be within [0, 1].')
        if scale_range[0] <= 0 or scale_range[0] > scale_range[1]:
            raise ValueError('scale_range must be positive and ordered.')
        self.horizontal_prob = horizontal_prob
        self.vertical_prob = vertical_prob
        self.scale_prob = scale_prob
        self.scale_range = scale_range

    @staticmethod
    def _resize(array: np.ndarray, size: Tuple[int, int], mode: str,
                is_mask: bool = False) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3:  # HWC HSI or NHW instance masks
            tensor = tensor[:, None] if is_mask else tensor.permute(2, 0, 1)[None]
        else:
            raise ValueError(f'Unsupported augmentation array shape {array.shape}.')
        kwargs = {} if mode == 'nearest' else {'align_corners': False}
        tensor = F.interpolate(tensor.float(), size=size, mode=mode, **kwargs)
        if array.ndim == 2:
            return tensor[0, 0].numpy()
        if is_mask:
            return tensor[:, 0].numpy()
        return tensor[0].permute(1, 2, 0).numpy()

    @staticmethod
    def _fit_canvas(array: np.ndarray, height: int, width: int,
                    top: int, left: int, pad_top: int, pad_left: int,
                    is_mask: bool = False) -> np.ndarray:
        """Crop or zero-pad a HWC, HW, or NHW array onto the target canvas."""
        is_hwc = array.ndim == 3 and not is_mask
        if is_hwc:
            cropped = array[top:top + min(height, array.shape[0]), left:left + min(width, array.shape[1]), :]
            output = np.zeros((height, width, array.shape[2]), dtype=array.dtype)
            output[pad_top:pad_top + cropped.shape[0], pad_left:pad_left + cropped.shape[1], :] = cropped
        elif array.ndim == 2:
            cropped = array[top:top + min(height, array.shape[0]), left:left + min(width, array.shape[1])]
            output = np.zeros((height, width), dtype=array.dtype)
            output[pad_top:pad_top + cropped.shape[0], pad_left:pad_left + cropped.shape[1]] = cropped
        else:  # NHW masks
            cropped = array[:, top:top + min(height, array.shape[1]), left:left + min(width, array.shape[2])]
            output = np.zeros((array.shape[0], height, width), dtype=array.dtype)
            output[:, pad_top:pad_top + cropped.shape[1], pad_left:pad_left + cropped.shape[2]] = cropped
        return output

    def transform(self, results: dict) -> dict:
        img, dsm = results['img'], results['dsm']
        height, width = img.shape[:2]
        masks = results.get('gt_masks')
        mask_array = masks.masks.copy() if masks is not None else None

        horizontal = np.random.rand() < self.horizontal_prob
        vertical = np.random.rand() < self.vertical_prob
        if horizontal:
            img, dsm = img[:, ::-1, :], dsm[:, ::-1]
            if mask_array is not None:
                mask_array = mask_array[:, :, ::-1]
        if vertical:
            img, dsm = img[::-1, :, :], dsm[::-1, :]
            if mask_array is not None:
                mask_array = mask_array[:, ::-1, :]

        factor = (np.random.uniform(*self.scale_range)
                  if np.random.rand() < self.scale_prob else 1.0)
        scaled_h, scaled_w = max(1, round(height * factor)), max(1, round(width * factor))
        if (scaled_h, scaled_w) != (height, width):
            img = self._resize(img, (scaled_h, scaled_w), 'bilinear')
            dsm = self._resize(dsm, (scaled_h, scaled_w), 'bilinear')
            if mask_array is not None:
                mask_array = self._resize(mask_array, (scaled_h, scaled_w), 'nearest', is_mask=True).astype(np.uint8)

        crop_top = np.random.randint(0, scaled_h - height + 1) if scaled_h > height else 0
        crop_left = np.random.randint(0, scaled_w - width + 1) if scaled_w > width else 0
        pad_top = np.random.randint(0, height - scaled_h + 1) if scaled_h < height else 0
        pad_left = np.random.randint(0, width - scaled_w + 1) if scaled_w < width else 0
        img = self._fit_canvas(img, height, width, crop_top, crop_left, pad_top, pad_left)
        dsm = self._fit_canvas(dsm, height, width, crop_top, crop_left, pad_top, pad_left)
        if mask_array is not None:
            mask_array = self._fit_canvas(mask_array, height, width, crop_top, crop_left, pad_top, pad_left,
                                          is_mask=True)
            keep = mask_array.reshape(mask_array.shape[0], -1).any(axis=1)
            mask_array = mask_array[keep]
            for key in ('gt_labels', 'gt_ignore_flags', 'gt_instance_ids'):
                if key in results:
                    results[key] = results[key][keep]
            boxes = []
            for instance in mask_array:
                ys, xs = np.where(instance > 0)
                boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            results['gt_bboxes'] = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            results['gt_masks'] = BitmapMasks(mask_array, height, width)

        results['img'] = np.ascontiguousarray(img.astype(np.float32, copy=False))
        results['dsm'] = np.ascontiguousarray(dsm.astype(np.float32, copy=False))
        results['img_shape'] = (height, width)
        results['dsm_shape'] = (height, width)
        results['flip'] = horizontal or vertical
        results['flip_direction'] = ('diagonal' if horizontal and vertical else
                                     'horizontal' if horizontal else
                                     'vertical' if vertical else None)
        results['scale_factor'] = np.array([factor, factor, factor, factor], dtype=np.float32)
        return results


@TRANSFORMS.register_module()
class LoadImageFromFile(MMCV_LoadImageFromFile):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 channel_order: str="bgr",
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.channel_order = channel_order
        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        
        
        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, channel_order = self.channel_order, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                   f"channel_order='{self.channel_order}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str

@TRANSFORMS.register_module()
class LoadDSMFromFile(MMCV_LoadImageFromFile):
    """Load an image from file.

    Required Keys:

    - dsm_path

    Modified Keys:

    - dsm
    - dsm_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'grayscale',
                 imdecode_backend: str = 'tifffile', #should always be used with tifffile
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 #channel_order: str="bgr",
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        #self.channel_order = channel_order
        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['dsm_path']
        try:
            img = tifffile.imread(filename)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            img = img.astype(np.float32)

        results['dsm'] = img
        results['dsm_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                  )

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str

    
@TRANSFORMS.register_module()
class LoadImageFromNDArray(MMCV_LoadImageFromFile):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadMultiChannelImageFromFiles(BaseTransform):
    """Load multi-channel images from a list of separate channel files.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'unchanged'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        file_client_args (dict): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet >= 3.0.0rc7. Defaults to None.
    """

    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = 'unchanged',
        imdecode_backend: str = 'pillow',
        file_client_args: dict = None,
        backend_args: dict = None,
    ) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )

    def transform(self, results: dict) -> dict:
        """Transform functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """
        
        assert isinstance(results['img_path'], list)
        img = []
        for name in results['img_path']:
            img_bytes = get(name, backend_args=self.backend_args)
            img.append(
                mmcv.imfrombytes(
                    img_bytes,
                    flag=self.color_type,
                    backend="pillow")) #self.imdecode_backend))
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='pillow'", #{self.imdecode_backend}', "
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotations(MMCV_LoadAnnotations):
    """Load and process the ``instances`` and ``seg_map`` annotation provided
    by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                'bbox': [x1, y1, x2, y2],

                # Label of image classification.
                'bbox_label': 1,

                # Used in instance/panoptic segmentation. The segmentation mask
                # of the instance or the information of segments.
                # 1. If list[list[float]], it represents a list of polygons,
                # one for each connected component of the object. Each
                # list[float] is one simple polygon in the format of
                # [x1, y1, ..., xn, yn] (n >= 3). The Xs and Ys are absolute
                # coordinates in unit of pixels.
                # 2. If dict, it represents the per-pixel segmentation mask in
                # COCO's compressed RLE format. The dict should have keys
                # “size” and “counts”.  Can be loaded by pycocotools
                'mask': list[list[float]] or dict,

                }
            ]
            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in an image
            'gt_bboxes': BaseBoxes(N, 4)
             # In int type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In built-in class
            'gt_masks': PolygonMasks (H, W) or BitmapMasks (H, W)
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # in (x, y, v) order, float type.
        }

    Required Keys:

    - height
    - width
    - instances

      - bbox (optional)
      - bbox_label
      - mask (optional)
      - ignore_flag

    - seg_map_path (optional)

    Added Keys:

    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_bboxes_labels (np.int64)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (bool)

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to True.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        poly2mask (bool): Whether to convert mask to bitmap. Default: True.
        box_type (str): The box type used to wrap the bboxes. If ``box_type``
            is None, gt_bboxes will keep being np.ndarray. Defaults to 'hbox'.
        reduce_zero_label (bool): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to False.
        ignore_index (int): The label index to be ignored.
            Valid only if reduce_zero_label is true. Defaults is 255.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(
            self,
            with_mask: bool = False,
            poly2mask: bool = True,
            box_type: str = 'hbox',
            # use for semseg
            reduce_zero_label: bool = False,
            ignore_index: int = 255,
            **kwargs) -> None:
        super(LoadAnnotations, self).__init__(**kwargs)
        self.with_mask = with_mask
        self.poly2mask = poly2mask
        self.box_type = box_type
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_bboxes.append(instance['bbox'])
            gt_ignore_flags.append(instance['ignore_flag'])
        if self.box_type is None:
            results['gt_bboxes'] = np.array(
                gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['bbox_label'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def _poly2mask(self, mask_ann: Union[list, dict], img_h: int,
                   img_w: int) -> np.ndarray:
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            np.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _process_masks(self, results: dict) -> list:
        """Process gt_masks and filter invalid polygons.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            list: Processed gt_masks.
        """
        gt_masks = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_mask = instance['mask']
            # If the annotation of segmentation mask is invalid,
            # ignore the whole instance.
            if isinstance(gt_mask, list):
                gt_mask = [
                    np.array(polygon) for polygon in gt_mask
                    if len(polygon) % 2 == 0 and len(polygon) >= 6
                ]
                if len(gt_mask) == 0:
                    # ignore this instance and set gt_mask to a fake mask
                    instance['ignore_flag'] = 1
                    gt_mask = [np.zeros(6)]
            elif not self.poly2mask:
                # `PolygonMasks` requires a ploygon of format List[np.array],
                # other formats are invalid.
                instance['ignore_flag'] = 1
                gt_mask = [np.zeros(6)]
            elif isinstance(gt_mask, dict) and \
                    not (gt_mask.get('counts') is not None and
                         gt_mask.get('size') is not None and
                         isinstance(gt_mask['counts'], (list, str))):
                # if gt_mask is a dict, it should include `counts` and `size`,
                # so that `BitmapMasks` can uncompressed RLE
                instance['ignore_flag'] = 1
                gt_mask = [np.zeros(6)]
            gt_masks.append(gt_mask)
            # re-process gt_ignore_flags
            gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)
        return gt_masks

    def _load_masks(self, results: dict) -> None:
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        h, w = results['ori_shape']
        gt_masks = self._process_masks(results)
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            # fake polygon masks will be ignored in `PackDetInputs`
            gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
        results['gt_masks'] = gt_masks

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if results.get('seg_map_path', None) is None:
            return

        img_bytes = get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze()

        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = self.ignore_index
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == self.ignore_index -
                            1] = self.ignore_index

        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['ignore_index'] = self.ignore_index

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadPanopticAnnotations(LoadAnnotations):
    """Load multiple types of panoptic annotations.

    The annotation format is as the following:

    .. code-block:: python

        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                'bbox': [x1, y1, x2, y2],

                # Label of image classification.
                'bbox_label': 1,
                },
                ...
            ]
            'segments_info':
            [
                {
                # id = cls_id + instance_id * INSTANCE_OFFSET
                'id': int,

                # Contiguous category id defined in dataset.
                'category': int

                # Thing flag.
                'is_thing': bool
                },
                ...
            ]

            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in an image
            'gt_bboxes': BaseBoxes(N, 4)
             # In int type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In built-in class
            'gt_masks': PolygonMasks (H, W) or BitmapMasks (H, W)
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # in (x, y, v) order, float type.
        }

    Required Keys:

    - height
    - width
    - instances
      - bbox
      - bbox_label
      - ignore_flag
    - segments_info
      - id
      - category
      - is_thing
    - seg_map_path

    Added Keys:

    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_bboxes_labels (np.int64)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (bool)

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to True.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Defaults to True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        box_type (str): The box mode used to wrap the bboxes.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet >= 3.0.0rc7. Defaults to None.
    """

    def __init__(self,
                 with_bbox: bool = True,
                 with_label: bool = True,
                 with_mask: bool = True,
                 with_seg: bool = True,
                 box_type: str = 'hbox',
                 imdecode_backend: str = 'cv2',
                 backend_args: dict = None) -> None:
        try:
            from panopticapi import utils
        except ImportError:
            raise ImportError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')
        self.rgb2id = utils.rgb2id

        super(LoadPanopticAnnotations, self).__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            with_keypoints=False,
            box_type=box_type,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)

    def _load_masks_and_semantic_segs(self, results: dict) -> None:
        """Private function to load mask and semantic segmentation annotations.

        In gt_semantic_seg, the foreground label is from ``0`` to
        ``num_things - 1``, the background label is from ``num_things`` to
        ``num_things + num_stuff - 1``, 255 means the ignored label (``VOID``).

        Args:
            results (dict): Result dict from :obj:``mmdet.CustomDataset``.
        """
        # seg_map_path is None, when inference on the dataset without gts.
        if results.get('seg_map_path', None) is None:
            return

        img_bytes = get(
            results['seg_map_path'], backend_args=self.backend_args)
        pan_png = mmcv.imfrombytes(
            img_bytes, flag='color', channel_order='rgb').squeeze()
        pan_png = self.rgb2id(pan_png)

        gt_masks = []
        gt_seg = np.zeros_like(pan_png) + 255  # 255 as ignore

        for segment_info in results['segments_info']:
            mask = (pan_png == segment_info['id'])
            gt_seg = np.where(mask, segment_info['category'], gt_seg)

            # The legal thing masks
            if segment_info.get('is_thing'):
                gt_masks.append(mask.astype(np.uint8))

        if self.with_mask:
            h, w = results['ori_shape']
            gt_masks = BitmapMasks(gt_masks, h, w)
            results['gt_masks'] = gt_masks

        if self.with_seg:
            results['gt_seg_map'] = gt_seg

    def transform(self, results: dict) -> dict:
        """Function to load multiple types panoptic annotations.

        Args:
            results (dict): Result dict from :obj:``mmdet.CustomDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask or self.with_seg:
            # The tasks completed by '_load_masks' and '_load_semantic_segs'
            # in LoadAnnotations are merged to one function.
            self._load_masks_and_semantic_segs(results)

        return results


@TRANSFORMS.register_module()
class LoadProposals(BaseTransform):
    """Load proposal pipeline.

    Required Keys:

    - proposals

    Modified Keys:

    - proposals

    Args:
        num_max_proposals (int, optional): Maximum number of proposals to load.
            If not specified, all proposals will be loaded.
    """

    def __init__(self, num_max_proposals: Optional[int] = None) -> None:
        self.num_max_proposals = num_max_proposals

    def transform(self, results: dict) -> dict:
        """Transform function to load proposals from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded proposal annotations.
        """

        proposals = results['proposals']
        # the type of proposals should be `dict` or `InstanceData`
        assert isinstance(proposals, dict) \
               or isinstance(proposals, BaseDataElement)
        bboxes = proposals['bboxes'].astype(np.float32)
        assert bboxes.shape[1] == 4, \
            f'Proposals should have shapes (n, 4), but found {bboxes.shape}'

        if 'scores' in proposals:
            scores = proposals['scores'].astype(np.float32)
            assert bboxes.shape[0] == scores.shape[0]
        else:
            scores = np.zeros(bboxes.shape[0], dtype=np.float32)

        if self.num_max_proposals is not None:
            # proposals should sort by scores during dumping the proposals
            bboxes = bboxes[:self.num_max_proposals]
            scores = scores[:self.num_max_proposals]

        if len(bboxes) == 0:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros(0, dtype=np.float32)

        results['proposals'] = bboxes
        results['proposals_scores'] = scores
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(num_max_proposals={self.num_max_proposals})'


@TRANSFORMS.register_module()
class FilterAnnotations(BaseTransform):
    """Filter invalid annotations.

    Required Keys:

    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)

    Args:
        min_gt_bbox_wh (tuple[float]): Minimum width and height of ground truth
            boxes. Default: (1., 1.)
        min_gt_mask_area (int): Minimum foreground area of ground truth masks.
            Default: 1
        by_box (bool): Filter instances with bounding boxes not meeting the
            min_gt_bbox_wh threshold. Default: True
        by_mask (bool): Filter instances with masks not meeting
            min_gt_mask_area threshold. Default: False
        keep_empty (bool): Whether to return None when it
            becomes an empty bbox after filtering. Defaults to True.
    """

    def __init__(self,
                 min_gt_bbox_wh: Tuple[int, int] = (1, 1),
                 min_gt_mask_area: int = 1,
                 by_box: bool = True,
                 by_mask: bool = False,
                 keep_empty: bool = True) -> None:
        # TODO: add more filter options
        assert by_box or by_mask
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.min_gt_mask_area = min_gt_mask_area
        self.by_box = by_box
        self.by_mask = by_mask
        self.keep_empty = keep_empty

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to filter annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        if gt_bboxes.shape[0] == 0:
            return results

        tests = []
        if self.by_box:
            tests.append(
                ((gt_bboxes.widths > self.min_gt_bbox_wh[0]) &
                 (gt_bboxes.heights > self.min_gt_bbox_wh[1])).numpy())
        if self.by_mask:
            assert 'gt_masks' in results
            gt_masks = results['gt_masks']
            tests.append(gt_masks.areas >= self.min_gt_mask_area)

        keep = tests[0]
        for t in tests[1:]:
            keep = keep & t

        if not keep.any():
            if self.keep_empty:
                return None

        keys = ('gt_bboxes', 'gt_bboxes_labels', 'gt_masks', 'gt_ignore_flags')
        for key in keys:
            if key in results:
                results[key] = results[key][keep]

        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(min_gt_bbox_wh={self.min_gt_bbox_wh}, ' \
               f'keep_empty={self.keep_empty})'


@TRANSFORMS.register_module()
class LoadEmptyAnnotations(BaseTransform):
    """Load Empty Annotations for unlabeled images.

    Added Keys:
    - gt_bboxes (np.float32)
    - gt_bboxes_labels (np.int64)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (bool)

    Args:
        with_bbox (bool): Whether to load the pseudo bbox annotation.
            Defaults to True.
        with_label (bool): Whether to load the pseudo label annotation.
            Defaults to True.
        with_mask (bool): Whether to load the pseudo mask annotation.
             Default: False.
        with_seg (bool): Whether to load the pseudo semantic segmentation
            annotation. Defaults to False.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Defaults to 255.
    """

    def __init__(self,
                 with_bbox: bool = True,
                 with_label: bool = True,
                 with_mask: bool = False,
                 with_seg: bool = False,
                 seg_ignore_label: int = 255) -> None:
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.seg_ignore_label = seg_ignore_label

    def transform(self, results: dict) -> dict:
        """Transform function to load empty annotations.

        Args:
            results (dict): Result dict.
        Returns:
            dict: Updated result dict.
        """
        if self.with_bbox:
            results['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
            results['gt_ignore_flags'] = np.zeros((0, ), dtype=bool)
        if self.with_label:
            results['gt_bboxes_labels'] = np.zeros((0, ), dtype=np.int64)
        if self.with_mask:
            # TODO: support PolygonMasks
            h, w = results['img_shape']
            gt_masks = np.zeros((0, h, w), dtype=np.uint8)
            results['gt_masks'] = BitmapMasks(gt_masks, h, w)
        if self.with_seg:
            h, w = results['img_shape']
            results['gt_seg_map'] = self.seg_ignore_label * np.ones(
                (h, w), dtype=np.uint8)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label})'
        return repr_str


@TRANSFORMS.register_module()
class InferencerLoader(BaseTransform):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.from_file = TRANSFORMS.build(
            dict(type='LoadImageFromFile', **kwargs))
        self.from_ndarray = TRANSFORMS.build(
            dict(type='mmdet.LoadImageFromNDArray', **kwargs))

    def transform(self, results: Union[str, np.ndarray, dict]) -> dict:
        """Transform function to add image meta information.

        Args:
            results (str, np.ndarray or dict): The result.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if isinstance(results, str):
            inputs = dict(img_path=results)
        elif isinstance(results, np.ndarray):
            inputs = dict(img=results)
        elif isinstance(results, dict):
            inputs = results
        else:
            raise NotImplementedError

        if 'img' in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)


@TRANSFORMS.register_module()
class LoadTrackAnnotations(LoadAnnotations):
    """Load and process the ``instances`` and ``seg_map`` annotation provided
    by dataset. It must load ``instances_ids`` which is only used in the
    tracking tasks. The annotation format is as the following:

    .. code-block:: python
        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                'bbox': [x1, y1, x2, y2],
                # Label of image classification.
                'bbox_label': 1,
                # Used in tracking.
                # Id of instances.
                'instance_id': 100,
                # Used in instance/panoptic segmentation. The segmentation mask
                # of the instance or the information of segments.
                # 1. If list[list[float]], it represents a list of polygons,
                # one for each connected component of the object. Each
                # list[float] is one simple polygon in the format of
                # [x1, y1, ..., xn, yn] (n >= 3). The Xs and Ys are absolute
                # coordinates in unit of pixels.
                # 2. If dict, it represents the per-pixel segmentation mask in
                # COCO's compressed RLE format. The dict should have keys
                # “size” and “counts”.  Can be loaded by pycocotools
                'mask': list[list[float]] or dict,
                }
            ]
            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:
    .. code-block:: python
        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in an image
            'gt_bboxes': np.ndarray(N, 4)
             # In int type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In built-in class
            'gt_masks': PolygonMasks (H, W) or BitmapMasks (H, W)
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # in (x, y, v) order, float type.
        }

    Required Keys:

    - height (optional)
    - width (optional)
    - instances
      - bbox (optional)
      - bbox_label
      - instance_id (optional)
      - mask (optional)
      - ignore_flag (optional)
    - seg_map_path (optional)

    Added Keys:

    - gt_bboxes (np.float32)
    - gt_bboxes_labels (np.int32)
    - gt_instances_ids (np.int32)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (np.bool)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        # TODO: use bbox_type
        for instance in results['instances']:
            # The datasets which are only format in evaluation don't have
            # groundtruth boxes.
            if 'bbox' in instance:
                gt_bboxes.append(instance['bbox'])
            if 'ignore_flag' in instance:
                gt_ignore_flags.append(instance['ignore_flag'])

        # TODO: check this case
        if len(gt_bboxes) != len(gt_ignore_flags):
            # There may be no ``gt_ignore_flags`` in some cases, we treat them
            # as all False in order to keep the length of ``gt_bboxes`` and
            # ``gt_ignore_flags`` the same
            gt_ignore_flags = [False] * len(gt_bboxes)

        results['gt_bboxes'] = np.array(
            gt_bboxes, dtype=np.float32).reshape(-1, 4)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

    def _load_instances_ids(self, results: dict) -> None:
        """Private function to load instances id annotations.

        Args:
            results (dict): Result dict from :obj :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict containing instances id annotations.
        """
        gt_instances_ids = []
        for instance in results['instances']:
            gt_instances_ids.append(instance['instance_id'])
        results['gt_instances_ids'] = np.array(
            gt_instances_ids, dtype=np.int32)

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label, instances id
            and semantic segmentation and keypoints annotations.
        """
        results = super().transform(results)
        self._load_instances_ids(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str
