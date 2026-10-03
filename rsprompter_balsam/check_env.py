import importlib

packages = [
    "torch",
    "torchvision",
    "mmcv",
    "mmengine",
    "mmdet",
    "transformers",
    "peft",
    "einops",
    "pycocotools",
    "rasterio",
    "geopandas",
    "shapely",
    "scipy",
    "cv2",
    "PIL",
    "yaml",
    "tifffile",
    "wandb",
    "pandas",
    "matplotlib",
]

failed = []

for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, "__version__", "unknown")
        print(f"[OK]   {pkg:<15} {version}")
    except Exception as e:
        print(f"[FAIL] {pkg:<15} {type(e).__name__}: {e}")
        failed.append(pkg)

print("\n==============================")

if failed:
    print("Missing / broken packages:")
    for pkg in failed:
        print(" -", pkg)
else:
    print("ALL CORE PACKAGES OK")