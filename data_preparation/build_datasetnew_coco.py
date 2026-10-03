"""Export HSI/DSM tiles and one-class COCO masks from DatasetNew.

Training windows: 1024 source pixels resampled to 512 output pixels, stride 256,
wholly within one AOI. Validation/test use exact fixed 1024 source-pixel
windows from the saved split GeoPackages, also exported at 512 pixels.
Only the supervised train_gold AOI is exported, under the output split name train.
"""
import argparse
import json
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds, transform as win_transform
from rasterio.warp import reproject
from shapely import make_valid
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import transform as shp_transform


ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'Data/DatasetNew/raw'
SPLITS = ROOT / 'Data/DatasetNew/splits'
OUT = ROOT / 'Data/DatasetNew/processed/balsam_hsi_512_stride256'
SOURCE_SIZE = 1024
OUTPUT_SIZE = 512
OUTPUT_SCALE = SOURCE_SIZE / OUTPUT_SIZE
STRIDE = 256
MIN_VALID = 0.95


def coco_empty(split, annotation_complete):
    return {'info': {'description': f'DatasetNew {split}; HSI + aligned DSM',
                     'annotation_complete': annotation_complete,
                     'warning': None if annotation_complete else
                     'Existing polygons only. This split has NOT been audited for complete crowns.'},
            'licenses': [], 'images': [], 'annotations': [],
            'categories': [{'id': 1, 'name': 'tree', 'supercategory': 'tree'}]}


def rings_for_coco(geom, transform):
    inverse = ~transform
    def xy(x,y,z=None): return inverse*(x,y)
    pix = shp_transform(xy, geom)
    def polygons(g):
        if isinstance(g, Polygon): return [g]
        if isinstance(g, (MultiPolygon, GeometryCollection)):
            return [p for child in g.geoms for p in polygons(child)]
        return []
    parts = polygons(pix)
    result=[]
    for poly in parts:
        if not isinstance(poly, Polygon) or poly.is_empty: continue
        coords = [(min(OUTPUT_SIZE,max(0,float(x) / OUTPUT_SCALE)),
                   min(OUTPUT_SIZE,max(0,float(y) / OUTPUT_SCALE)))
                  for x,y in list(poly.exterior.coords)[:-1]]
        if len(coords)<3:continue
        flat=[v for p in coords for v in p]
        if len(flat)>=6:result.append(flat)
    return result


def write_hsi(src, window, target):
    profile=src.profile.copy()
    output_transform=win_transform(window,src.transform) * rasterio.Affine.scale(OUTPUT_SCALE)
    profile.update(driver='GTiff',height=OUTPUT_SIZE,width=OUTPUT_SIZE,count=src.count,
                   transform=output_transform,tiled=True,
                   blockxsize=256,blockysize=256,compress='lzw',predictor=2,
                   BIGTIFF='IF_SAFER')
    # Average resampling is appropriate when reducing the linear resolution by 2.
    pixels=src.read(window=window,out_shape=(src.count,OUTPUT_SIZE,OUTPUT_SIZE),
                    resampling=Resampling.average)
    with rasterio.open(target,'w',**profile) as dst:
        dst.write(pixels)


def write_dsm(dsm, src, window, target):
    target_transform=win_transform(window,src.transform) * rasterio.Affine.scale(OUTPUT_SCALE)
    arr=np.full((OUTPUT_SIZE,OUTPUT_SIZE),-32767,dtype=np.float32)
    reproject(source=rasterio.band(dsm,1),destination=arr,
              src_transform=dsm.transform,src_crs=dsm.crs,src_nodata=dsm.nodata,
              dst_transform=target_transform,dst_crs=src.crs,dst_nodata=-32767,
              resampling=Resampling.bilinear)
    profile=dict(driver='GTiff',width=OUTPUT_SIZE,height=OUTPUT_SIZE,count=1,dtype='float32',
                 crs=src.crs,transform=target_transform,nodata=-32767,
                 tiled=True,blockxsize=256,blockysize=256,compress='deflate',predictor=3)
    with rasterio.open(target,'w',**profile) as dst:dst.write(arr,1)
    return float(np.count_nonzero(arr!=-32767)/arr.size)


def windows_for_site(src, tiles, aois):
    zone='train'
    roi=aois.loc[aois.zone=='train_gold','geometry'].iloc[0]
    for row in range(0,src.height-SOURCE_SIZE+1,STRIDE):
        for col in range(0,src.width-SOURCE_SIZE+1,STRIDE):
            win=Window(col,row,SOURCE_SIZE,SOURCE_SIZE)
            if roi.covers(box(*bounds(win,src.transform))):
                yield zone,row,col,win
    for zone in ['val','test']:
        for rec in tiles.loc[tiles.zone==zone].itertuples():
            yield zone,int(rec.pixel_y),int(rec.pixel_x),Window(int(rec.pixel_x),int(rec.pixel_y),SOURCE_SIZE,SOURCE_SIZE)


def add_annotations(coco, crowns, window, src, image_id, ann_start):
    geom=box(*bounds(window,src.transform))
    hits=list(crowns.sindex.query(geom,predicate='intersects'))
    next_id=ann_start
    tr=win_transform(window,src.transform)
    for idx in hits:
        piece=crowns.geometry.iloc[idx].intersection(geom)
        if piece.is_empty or piece.area<=0:continue
        polys=rings_for_coco(piece,tr)
        if not polys:continue
        # Polygon COCO format works with both pycocotools and BalSAM's loader.
        xs=[x for poly in polys for x in poly[::2]]
        ys=[y for poly in polys for y in poly[1::2]]
        left,top,right,bottom=min(xs),min(ys),max(xs),max(ys)
        if right-left<1 or bottom-top<1:continue
        coco['annotations'].append(dict(id=next_id,image_id=image_id,category_id=1,
            segmentation=polys,bbox=[left,top,right-left,bottom-top],
            area=float(piece.area/abs(tr.a*tr.e)/(OUTPUT_SCALE**2)),iscrowd=0,
            source_id=str(crowns.iloc[idx]['id'])))
        next_id+=1
    return next_id


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=OUT)
    ap.add_argument('--manifest-only',action='store_true',help='Write JSON/manifest without TIFF export')
    ap.add_argument('--limit',type=int,help='For a quick local export test; omit for the full dataset')
    ap.add_argument('--resume-partial',action='store_true',help='Resume a previously interrupted export in the selected output directory')
    args=ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.resume_partial:
        raise FileExistsError(f'Output already exists: {args.output}')
    args.output.mkdir(parents=True,exist_ok=True)
    sets={z:coco_empty(z,z=='train') for z in ['train','val','test']}
    manifest=[]
    next_img=1
    next_ann=1
    for site in ['01','02']:
        if args.limit and next_img>args.limit:break
        raster=RAW/'orthophoto'/f'{site}_hsi.tif'
        dsm_path=RAW/'dsm'/f'{site}_dsm.tif'
        label=RAW/'labels'/f'{site}_crowns_new.gpkg'
        layer_names=fiona.listlayers(label)
        # Area 02 file also contains an unrelated site-01 layer. Select by site name.
        matching=[name for name in layer_names if name.startswith(f'{site}_crowns__')]
        if len(matching)!=1:raise ValueError(f'Cannot uniquely select site {site} crowns: {layer_names}')
        crowns=gpd.read_file(label,layer=matching[0])
        bad=crowns.loc[~crowns.is_valid,'id'].astype(str).tolist()
        if bad:print(f'Site {site}: temporarily repairing invalid geometries for export: {bad}',flush=True)
        crowns.geometry=crowns.geometry.map(make_valid)
        split=SPLITS/f'site{site}_proposal_v1'/f'site{site}_spatial_split.gpkg'
        tiles=gpd.read_file(split,layer='tiles_1024')
        aois=gpd.read_file(split,layer='split_aois')
        with rasterio.Env(GDAL_CACHEMAX=128*1024*1024),rasterio.open(raster) as src,rasterio.open(dsm_path) as dsm:
            if crowns.crs != src.crs:crowns=crowns.to_crs(src.crs)
            # Site-02 split GPKG stores the same HK 1980 Grid projection as an
            # unnamed WKT (without EPSG authority); its coordinates match the HSI.
            for zone,row,col,win in windows_for_site(src,tiles,aois):
                if args.limit and next_img>args.limit:break
                sample=src.read(40,window=win,masked=True)
                valid=float(np.count_nonzero(~np.ma.getmaskarray(sample))/sample.size)
                if valid<MIN_VALID:
                    manifest.append(dict(site=site,zone=zone,row=row,col=col,status='skipped_nodata',valid=valid))
                    continue
                image_name=f'{site}_r{row:05d}_c{col:05d}'
                tile_dir=args.output/zone
                hsi_out=tile_dir/'hsi'/f'{image_name}.tif'
                dsm_out=tile_dir/'dsm'/f'{image_name}.tif'
                tile_dir.joinpath('hsi').mkdir(parents=True,exist_ok=True)
                tile_dir.joinpath('dsm').mkdir(parents=True,exist_ok=True)
                if not args.manifest_only:
                    write_hsi(src,win,hsi_out)
                    dsm_valid=write_dsm(dsm,src,win,dsm_out)
                else:dsm_valid=None
                sets[zone]['images'].append(dict(id=next_img,file_name=str(hsi_out.resolve()),
                    dsm_path=str(dsm_out.resolve()),width=OUTPUT_SIZE,height=OUTPUT_SIZE,
                    site=site,source_row=row,source_col=col,valid_fraction=valid,
                    label_status='complete_claimed' if zone=='train' else 'not_audited'))
                if zone in ('train','val','test'):
                    ann_before=len(sets[zone]['annotations'])
                    next_ann=add_annotations(sets[zone],crowns,win,src,next_img,next_ann)
                    one=coco_empty(zone,zone=='train')
                    one['images']=[sets[zone]['images'][-1]]
                    one['annotations']=sets[zone]['annotations'][ann_before:]
                    single_dir=tile_dir/'coco';single_dir.mkdir(exist_ok=True)
                    (single_dir/f'{image_name}.json').write_text(json.dumps(one,ensure_ascii=False),encoding='utf-8')
                manifest.append(dict(site=site,zone=zone,row=row,col=col,status='kept',valid=valid,dsm_valid=dsm_valid))
                next_img+=1
                print(site,zone,image_name,'valid',round(valid,4),'ann',next_ann-1,flush=True)
    ann_dir=args.output/'annotations';ann_dir.mkdir(exist_ok=True)
    for zone,coco in sets.items():
        (ann_dir/f'{zone}.json').write_text(json.dumps(coco,ensure_ascii=False),encoding='utf-8')
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'image_counts':{z:len(c['images']) for z,c in sets.items()},
             'annotation_counts':{z:len(c['annotations']) for z,c in sets.items()},
             'skipped_windows':sum(x['status']!='kept' for x in manifest),
             'manifest_only':args.manifest_only,
             'label_warning':'Train uses the supervised train_gold AOI. Val/test COCO annotations are provisional and cannot support valid AP until exhaustively audited.',
             'hsi_note':'164-band uint16 HSI is average-resampled from 1024x1024 source windows to 512x512 output while preserving the same geographic footprint; a custom hyperspectral loader/adapter is required.',
             'dsm_note':'DSM is bilinearly reprojected to the 512x512 output grid covering the same source HSI window; values are absolute surface elevation, not canopy height.'}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
