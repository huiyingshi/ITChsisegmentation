"""Read-only consistency checks for the DatasetNew HSI/DSM/COCO export."""
import json
from pathlib import Path

import geopandas as gpd
import rasterio
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Data/DatasetNew/processed/balsam_hsi_512_stride256'
SPLITS = ROOT / 'Data/DatasetNew/splits'


def main():
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    assert not summary['manifest_only']
    test_expected=set()
    for site in ['01','02']:
        gpkg=SPLITS/f'site{site}_proposal_v1'/f'site{site}_spatial_split.gpkg'
        tiles=gpd.read_file(gpkg,layer='tiles_1024')
        test_expected.update((site,int(row.pixel_y),int(row.pixel_x))
                             for row in tiles.loc[tiles.zone=='test'].itertuples())
    seen_test=set()
    for zone in ['train','val','test']:
        coco=COCO(str(OUT/'annotations'/f'{zone}.json'))
        images=coco.dataset['images'];anns=coco.dataset['annotations']
        assert len(images)==summary['image_counts'][zone]
        assert len(anns)==summary['annotation_counts'][zone]
        for img in images:
            hsi=Path(img['file_name']);dsm=Path(img['dsm_path'])
            assert hsi.is_file() and dsm.is_file(),(hsi,dsm)
            with rasterio.open(hsi) as a,rasterio.open(dsm) as b:
                assert (a.width,a.height,a.count,a.dtypes[0])==(512,512,164,'uint16')
                assert (b.width,b.height,b.count,b.dtypes[0])==(512,512,1,'float32')
                assert a.transform==b.transform and a.crs==b.crs
            if zone=='test':seen_test.add((img['site'],img['source_row'],img['source_col']))
            one=OUT/zone/'coco'/f'{hsi.stem}.json'
            assert one.is_file(),one
            tile=json.loads(one.read_text(encoding='utf-8'))
            assert len(tile['images'])==1 and tile['images'][0]['id']==img['id']
            assert len(tile['annotations'])==len(coco.getAnnIds(imgIds=[img['id']]))
        for ann in anns:
            assert ann['category_id']==1 and ann['area']>0
            assert all(0<=v<=512 for poly in ann['segmentation'] for v in poly)
        if anns:
            first=anns[0]
            assert coco.annToMask(first).sum()>0
        print(zone,len(images),'images',len(anns),'annotations')
    assert seen_test==test_expected,(seen_test,test_expected)
    print('All file, alignment, annotation, and exact test-grid checks passed.')


if __name__=='__main__':main()
