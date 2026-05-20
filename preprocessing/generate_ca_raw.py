import os.path as osp
import pandas as pd
import json
import shapely
from shapely.geometry import Point
import geopandas as gpd
from utils import get_root_dir
import ast


def extract_name(cat_str):
    try:
        if pd.isna(cat_str):
            return cat_str
        # 将字符串 " [{'url': '...', 'name': 'Airport'}] " 转换为真正的列表字典
        cat_list = ast.literal_eval(cat_str)
        # 提取第一个元素的 name
        return cat_list[0]['name'] if len(cat_list) > 0 else None
    except:
        return None


if __name__ == "__main__":
    data_path = 'datasets/ca/raw'
    raw_checkins = pd.read_csv(osp.join(data_path, 'loc-gowalla_totalCheckins.txt'), sep='\t', header=None)
    raw_checkins.columns = ['userid', 'datetime', 'checkins_lat', 'checkins_lng', 'id']
    subset1 = pd.read_csv(osp.join(data_path, 'gowalla_spots_subset1.csv'))
    raw_checkins_subset1 = raw_checkins.merge(subset1, on='id')

    with open(osp.join(data_path, 'us_state_polygon_json.json'), 'r') as f:
        us_state_polygon = json.load(f)

    print("decide the range of the two state")
    for i in us_state_polygon['features']:
        if i['properties']['name'].lower() == 'california':
            california = shapely.polygons(i['geometry']['coordinates'][0])
        if i['properties']['name'].lower() == 'nevada':
            nevada = shapely.polygons(i['geometry']['coordinates'][0])

    # check if the checkin took place in California or Nevada
    print("check if the checkin took place in California or Nevada")
    """raw_checkins_subset1['is_ca'] = raw_checkins_subset1.apply(
        lambda x: nevada.intersects(
            shapely.geometry.Point(x['checkins_lng'], x['checkins_lat'])) or california.intersects(
            shapely.geometry.Point(x['checkins_lng'], x['checkins_lat'])), axis=1
    )
    raw_checkins_subset1 = raw_checkins_subset1[raw_checkins_subset1['is_ca']]"""

    gdf = gpd.GeoDataFrame(
        raw_checkins_subset1,
        geometry=gpd.points_from_xy(
            raw_checkins_subset1['checkins_lng'],
            raw_checkins_subset1['checkins_lat']
        ),
        crs='EPSG:4326'
    )

    merged = gpd.GeoSeries([nevada, california], crs='EPSG:4326').union_all()

    gdf['is_ca'] = gdf.intersects(merged)

    raw_checkins_subset1 = gdf[gdf['is_ca']]

    print("output to csv")
    df = raw_checkins_subset1[['userid', 'id', 'spot_categories', 'checkins_lat', 'checkins_lng', 'datetime']].copy()

    df['spot_categories'] = df['spot_categories'].apply(extract_name)

    df.columns = ['UserId', 'PoiId', 'PoiCategoryId', 'Latitude', 'Longitude', 'UTCTime']
    df.to_csv(osp.join(data_path, 'dataset_gowalla_ca_ne.csv'), index=False)
