import json
from pathlib import Path

import numpy as np
from pycocotools import mask as maskUtils


json_path = Path(
    r"D:\KEYAN\TreeMonitor\Data\DatasetNew\processed"
    r"\balsam_hsi_1024_stride256\annotations\train.json"
)

with open(json_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}

bad = []

for i, ann in enumerate(coco["annotations"]):
    ann_id = ann.get("id")
    image_id = ann.get("image_id")
    seg = ann.get("segmentation")

    img = images.get(image_id, {})
    h = img.get("height")
    w = img.get("width")

    try:
        if isinstance(seg, list):
            # Polygon
            if len(seg) == 0:
                raise ValueError("empty polygon list")

            for poly in seg:
                if not isinstance(poly, list):
                    raise TypeError(
                        f"polygon is not list: {type(poly)}"
                    )

                if len(poly) < 6:
                    raise ValueError(
                        f"polygon has fewer than 3 points: {len(poly)} values"
                    )

                if len(poly) % 2 != 0:
                    raise ValueError(
                        f"polygon has odd number of coordinates: {len(poly)}"
                    )

            rles = maskUtils.frPyObjects(seg, h, w)
            rle = maskUtils.merge(rles)
            mask = maskUtils.decode(rle)

        elif isinstance(seg, dict):
            # RLE
            if "size" not in seg or "counts" not in seg:
                raise ValueError(
                    "RLE must contain both 'size' and 'counts'"
                )

            mask = maskUtils.decode(seg)

        else:
            raise TypeError(
                f"unsupported segmentation type: {type(seg)}"
            )

        if mask.shape[:2] != (h, w):
            raise ValueError(
                f"decoded mask shape {mask.shape} != image {(h, w)}"
            )

    except Exception as e:
        bad.append((ann_id, image_id, str(e)))

        print("\nBAD ANNOTATION")
        print("index:", i)
        print("annotation id:", ann_id)
        print("image id:", image_id)
        print("image:", img.get("file_name"))
        print("segmentation type:", type(seg).__name__)
        print("segmentation:", str(seg)[:500])
        print("error:", repr(e))

print("\n==============================")
print("Total annotations:", len(coco["annotations"]))
print("Bad annotations:", len(bad))

if not bad:
    print("ALL MASKS OK")