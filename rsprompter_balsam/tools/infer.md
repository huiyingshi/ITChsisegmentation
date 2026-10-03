../.venv/bin/python tools/test.py \
  /home/huiying/Documents/ITCsegmentation/Code/BalSAM/rsprompter_balsam/configs/rsprompter/rsprompter_anchor-trees-pca3-dsm-alllabeled.py \
  --checkpoint "/home/huiying/Documents/ITCsegmentation/Code/Output/balsam_pca3_dsm_alllabeled/seed_0/best_coco_segm_mAP_epoch_12.pth" \
  --cfg-options \
    test_dataloader.dataset.ann_file=annotations/val.json \
    default_hooks.visualization.type=DetVisualizationHook \
  --work-dir "/home/huiying/Documents/ITCsegmentation/Code/Output/balsam_pca3_dsm_alllabeled/val_inference" \
  --show-dir val_predictions