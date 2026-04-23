import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path


VIOLENCE_CRIME_TYPES = {
    "ASSAULT WITH DEADLY WEAPON ON POLICE OFFICER",
    "ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT",
    "ATTEMPTED ROBBERY",
    "BATTERY - SIMPLE ASSAULT",
    "BATTERY ON A FIREFIGHTER",
    "BATTERY POLICE (SIMPLE)",
    "BATTERY WITH SEXUAL CONTACT",
    "BEASTIALITY, CRIME AGAINST NATURE SEXUAL ASSLT WITH ANIM",
    "BOMB SCARE",
    "BRANDISH WEAPON",
    "CHILD ABUSE (PHYSICAL) - AGGRAVATED ASSAULT",
    "CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT",
    "CRIMINAL HOMICIDE",
    "CRIMINAL THREATS - NO WEAPON DISPLAYED",
    "DISCHARGE FIREARMS/SHOTS FIRED",
    "EXTORTION",
    "FALSE IMPRISONMENT",
    "HUMAN TRAFFICKING - COMMERCIAL SEX ACTS",
    "HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE",
    "INCEST (SEXUAL ACTS BETWEEN BLOOD RELATIVES)",
    "INTIMATE PARTNER - AGGRAVATED ASSAULT",
    "INTIMATE PARTNER - SIMPLE ASSAULT",
    "KIDNAPPING",
    "KIDNAPPING - GRAND ATTEMPT",
    "LEWD/LASCIVIOUS ACTS WITH CHILD",
    "LYNCHING",
    "LYNCHING - ATTEMPTED",
    "MANSLAUGHTER, NEGLIGENT",
    "ORAL COPULATION",
    "OTHER ASSAULT",
    "PANDERING",
    "PIMPING",
    "RAPE, ATTEMPTED",
    "RAPE, FORCIBLE",
    "ROBBERY",
    "SEX,UNLAWFUL(INC MUTUAL CONSENT, PENETRATION W/ FRGN OBJ",
    "SEXUAL PENETRATION W/FOREIGN OBJECT",
    "SHOTS FIRED AT INHABITED DWELLING",
    "SHOTS FIRED AT MOVING VEHICLE, TRAIN OR AIRCRAFT",
    "SODOMY/SEXUAL CONTACT B/W PENIS OF ONE PERS TO ANUS OTH",
    "STALKING",
    "THREATENING PHONE CALLS/LETTERS",
    "THROWING OBJECT AT MOVING VEHICLE",
    "WEAPONS POSSESSION/BOMBING",
}


def classify_crime(desc):
    return "Violence" if str(desc).upper() in VIOLENCE_CRIME_TYPES else "Property"


def get_grid():
    grid = gpd.read_file("../data/City_Boundary/LA_400m_grid.shp").to_crs("EPSG:32611").reset_index(drop=True)

    if "grid_id" not in grid.columns:
        grid["grid_id"] = grid.index + 1

    grid["grid_id"] = grid["grid_id"].astype(int)
    grid["grid_area"] = grid.geometry.area
    return grid[["grid_id", "geometry", "grid_area"]]


def get_monthly_panel():
    cache_path = Path("../data/monthly_crime_panel.csv")

    if cache_path.exists():
        panel = pd.read_csv(cache_path, parse_dates=["month_start"])
        panel["grid_id"] = panel["grid_id"].astype(int)
        for col in ["crime_count", "property_crime", "violence_crime"]:
            panel[col] = panel[col].fillna(0).astype(int)
        return panel.sort_values(["grid_id", "month_start"]).reset_index(drop=True)

    crime_path = Path("../data/Crime_Data_from_2010_to_2024.csv")
    if not crime_path.exists():
        raise FileNotFoundError(f"Crime file not found: {crime_path}")

    crime_df = pd.read_csv(
        crime_path,
        usecols=["DR_NO", "LAT", "LON", "DATE OCC", "Crm Cd Desc"],
    )
    crime_df.columns = crime_df.columns.str.strip()
    crime_df = crime_df.drop_duplicates(subset=["DR_NO"]).copy()

    crime_df["LAT"] = pd.to_numeric(crime_df["LAT"], errors="coerce")
    crime_df["LON"] = pd.to_numeric(crime_df["LON"], errors="coerce")
    crime_df["date_occ"] = pd.to_datetime(crime_df["DATE OCC"], errors="coerce")

    crime_df = crime_df.dropna(subset=["LAT", "LON", "date_occ"]).copy()
    crime_df = crime_df[
        crime_df["LAT"].between(-90, 90)
        & crime_df["LON"].between(-180, 180)
        & (crime_df["LAT"] != 0)
        & (crime_df["LON"] != 0)
    ].copy()

    crime_df["Crime_Group"] = crime_df["Crm Cd Desc"].map(classify_crime)

    grid = get_grid()

    if crime_df.empty:
        return pd.DataFrame(
            columns=["grid_id", "month_start", "crime_count", "property_crime", "violence_crime"]
        )

    crime_gdf = gpd.GeoDataFrame(
        crime_df[["date_occ", "Crime_Group"]],
        geometry=gpd.points_from_xy(crime_df["LON"], crime_df["LAT"]),
        crs="EPSG:4326",
    ).to_crs(grid.crs)

    joined = gpd.sjoin(
        crime_gdf,
        grid[["grid_id", "geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    if joined.empty:
        month_idx = pd.date_range(
            crime_df["date_occ"].min().to_period("M").to_timestamp(),
            crime_df["date_occ"].max().to_period("M").to_timestamp(),
            freq="MS",
        )
        full_idx = pd.MultiIndex.from_product(
            [grid["grid_id"].sort_values().unique(), month_idx],
            names=["grid_id", "month_start"],
        )
        panel = full_idx.to_frame(index=False)
        panel["crime_count"] = 0
        panel["property_crime"] = 0
        panel["violence_crime"] = 0
    else:
        joined["month_start"] = joined["date_occ"].dt.to_period("M").dt.to_timestamp()

        monthly = (
            joined.groupby(["grid_id", "month_start", "Crime_Group"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        monthly["property_crime"] = monthly.get("Property", 0)
        monthly["violence_crime"] = monthly.get("Violence", 0)
        monthly["crime_count"] = monthly["property_crime"] + monthly["violence_crime"]
        monthly = monthly[["grid_id", "month_start", "crime_count", "property_crime", "violence_crime"]]

        month_idx = pd.date_range(monthly["month_start"].min(), monthly["month_start"].max(), freq="MS")
        full_idx = pd.MultiIndex.from_product(
            [grid["grid_id"].sort_values().unique(), month_idx],
            names=["grid_id", "month_start"],
        )
        full_df = full_idx.to_frame(index=False)
        panel = full_df.merge(monthly, on=["grid_id", "month_start"], how="left")
        panel[["crime_count", "property_crime", "violence_crime"]] = panel[
            ["crime_count", "property_crime", "violence_crime"]
        ].fillna(0)

    panel["grid_id"] = panel["grid_id"].astype(int)
    panel["month_start"] = pd.to_datetime(panel["month_start"])
    panel["crime_count"] = panel["crime_count"].astype(int)
    panel["property_crime"] = panel["property_crime"].astype(int)
    panel["violence_crime"] = panel["violence_crime"].astype(int)

    panel = panel.sort_values(["grid_id", "month_start"]).reset_index(drop=True)
    panel.to_csv(cache_path, index=False, date_format="%Y-%m-%d")

    return panel


def get_building_feature():
    grid = get_grid()
    buildings = gpd.read_file("../data/Building_Footprints-shp/building.shp").to_crs(grid.crs)
    buildings = buildings[buildings.geometry.notna()].copy()

    base = grid[["grid_id", "grid_area"]].copy()

    if buildings.empty:
        feat = base.copy()
        feat["building_count"] = 0.0
        feat["building_area_sum"] = 0.0
        feat["building_coverage_ratio"] = 0.0
        feat["mean_building_area"] = 0.0
        return feat[["grid_id", "building_count", "building_area_sum", "building_coverage_ratio", "mean_building_area"]]

    centroids = buildings.copy()
    centroids["geometry"] = centroids.geometry.centroid

    count_df = (
        gpd.sjoin(
            centroids[["geometry"]],
            grid[["grid_id", "geometry"]],
            how="inner",
            predicate="within",
        )
        .groupby("grid_id")
        .size()
        .reset_index(name="building_count")
    )

    inter = gpd.overlay(
        buildings[["geometry"]],
        grid[["grid_id", "geometry"]],
        how="intersection",
    )

    if inter.empty:
        area_df = pd.DataFrame({"grid_id": [], "building_area_sum": []})
    else:
        inter["inter_area"] = inter.geometry.area
        area_df = inter.groupby("grid_id")["inter_area"].sum().reset_index(name="building_area_sum")

    feat = base.merge(count_df, on="grid_id", how="left").merge(area_df, on="grid_id", how="left")
    feat[["building_count", "building_area_sum"]] = feat[["building_count", "building_area_sum"]].fillna(0.0)

    feat["building_coverage_ratio"] = np.where(
        feat["grid_area"] > 0,
        feat["building_area_sum"] / feat["grid_area"],
        0.0,
    )
    feat["mean_building_area"] = np.where(
        feat["building_count"] > 0,
        feat["building_area_sum"] / feat["building_count"],
        0.0,
    )

    return feat[["grid_id", "building_count", "building_area_sum", "building_coverage_ratio", "mean_building_area"]]


def get_osm_raw(grid):
    import math
    import time
    import osmnx as ox
    from shapely.geometry import box

    final_cache = Path("../data/osm_raw_buffer400.gpkg")
    chunk_dir = Path("../data/osm_chunks")

    chunk_dir.mkdir(parents=True, exist_ok=True)

    if final_cache.exists():
        osm = gpd.read_file(final_cache)
        if osm.crs is None:
            osm = osm.set_crs("EPSG:32611")
        return osm.to_crs(grid.crs)

    tags = {
        "amenity": True,
        "shop": True,
        "highway": ["bus_stop"],
        "public_transport": True,
        "railway": ["station", "halt"],
    }

    boundary_wgs84 = grid.to_crs("EPSG:4326").geometry.union_all()
    minx, miny, maxx, maxy = boundary_wgs84.bounds

    tile_size = 0.03
    x_steps = math.ceil((maxx - minx) / tile_size)
    y_steps = math.ceil((maxy - miny) / tile_size)

    chunk_paths = []

    for i in range(x_steps):
        for j in range(y_steps):
            x1 = minx + i * tile_size
            x2 = min(x1 + tile_size, maxx)
            y1 = miny + j * tile_size
            y2 = min(y1 + tile_size, maxy)

            tile = box(x1, y1, x2, y2).intersection(boundary_wgs84)
            if tile.is_empty:
                continue

            chunk_path = chunk_dir / f"osm_chunk_{i}_{j}.gpkg"
            chunk_paths.append(chunk_path)

            if chunk_path.exists():
                continue

            try:
                part = ox.features_from_polygon(tile, tags=tags)

                if not isinstance(part, gpd.GeoDataFrame):
                    part = gpd.GeoDataFrame(part, geometry="geometry", crs="EPSG:4326")
                elif part.crs is None:
                    part = part.set_crs("EPSG:4326")

                part = part.reset_index(drop=False)

                if part.empty:
                    empty_gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
                    empty_gdf.to_file(chunk_path, driver="GPKG")
                    print(f"Chunk ({i}, {j}) is empty.")
                    continue

                keep_cols = ["geometry", "amenity", "shop", "highway", "public_transport", "railway"]
                keep_cols = [c for c in keep_cols if c in part.columns]
                part = part[keep_cols].copy()

                for c in part.columns:
                    if c != "geometry":
                        part[c] = part[c].astype(str)

                part.to_file(chunk_path, driver="GPKG")
                time.sleep(1)

            except Exception as e:
                if "No matching features" in str(e):
                    empty_gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
                    empty_gdf.to_file(chunk_path, driver="GPKG")
                    print(f"Chunk ({i}, {j}) is empty.")
                else:
                    print(f"Chunk ({i}, {j}) failed: {e}")

    gdfs = []
    for chunk_path in chunk_paths:
        if not chunk_path.exists():
            continue

        try:
            g = gpd.read_file(chunk_path)
            if g.crs is None:
                g = g.set_crs("EPSG:4326")
            if not g.empty:
                gdfs.append(g)
        except Exception as e:
            print(f"Cannot read {chunk_path}: {e}")

    if not gdfs:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=grid.crs)

    osm = pd.concat(gdfs, ignore_index=True)
    osm = gpd.GeoDataFrame(osm, geometry="geometry", crs="EPSG:4326")

    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary_wgs84], crs="EPSG:4326")
    osm = gpd.clip(osm, boundary_gdf)

    dedup_cols = [c for c in ["amenity", "shop", "highway", "public_transport", "railway", "geometry"] if c in osm.columns]
    if dedup_cols:
        osm = osm.drop_duplicates(subset=dedup_cols)
    else:
        osm = osm.drop_duplicates()

    osm = osm.to_crs("EPSG:32611").reset_index(drop=True)
    osm["geometry"] = osm.geometry.buffer(400)

    keep_cols = ["geometry", "amenity", "shop", "highway", "public_transport", "railway"]
    keep_cols = [c for c in keep_cols if c in osm.columns]
    osm = osm[keep_cols].copy()

    for c in osm.columns:
        if c != "geometry":
            osm[c] = osm[c].astype(str)

    osm.to_file(final_cache, driver="GPKG")

    return osm.to_crs(grid.crs)


def poi_classification(row):
    amenity = str(row.get("amenity", "")).lower()
    shop = str(row.get("shop", "")).lower()
    highway = str(row.get("highway", "")).lower()
    public_transport = str(row.get("public_transport", "")).lower()
    railway = str(row.get("railway", "")).lower()

    if amenity in {"bar", "pub", "nightclub"}:
        return "nightlife"

    if (shop and shop != "nan") or amenity in {
        "restaurant", "cafe", "fast_food", "marketplace", "pharmacy", "bank", "atm"
    }:
        return "commercial"

    if highway == "bus_stop" or (public_transport and public_transport != "nan") or railway in {"station", "halt"}:
        return "transit"

    if amenity in {"school", "college", "university", "kindergarten"}:
        return "school"

    return None


def entropy(values):
    total = float(values.sum())
    if total <= 0:
        return 0.0

    p = values / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def get_empty_poi_feature(base):
    feat = base.copy()
    feat["commercial_density"] = 0.0
    feat["nightlife_density"] = 0.0
    feat["transit_density"] = 0.0
    feat["school_density"] = 0.0
    feat["poi_total_count"] = 0.0
    feat["poi_diversity"] = 0.0

    return feat[[
        "grid_id",
        "commercial_density",
        "nightlife_density",
        "transit_density",
        "school_density",
        "poi_total_count",
        "poi_diversity",
    ]]


def get_poi_feature():
    grid = get_grid()
    raw = get_osm_raw(grid)
    raw = raw[raw.geometry.notna()].copy()

    base = grid[["grid_id", "grid_area"]].copy()

    if raw.empty:
        return get_empty_poi_feature(base)

    poi = raw.to_crs(grid.crs).copy()
    poi["poi_category"] = poi.apply(poi_classification, axis=1)
    poi = poi[poi["poi_category"].notna()].copy().reset_index(drop=True)

    if poi.empty:
        return get_empty_poi_feature(base)

    joined = gpd.sjoin(
        poi[["poi_category", "geometry"]],
        grid[["grid_id", "geometry"]],
        how="inner",
        predicate="intersects",
    ).reset_index(drop=True)

    if joined.empty:
        return get_empty_poi_feature(base)

    counts = pd.crosstab(joined["grid_id"], joined["poi_category"])

    for c in ["commercial", "nightlife", "transit", "school"]:
        if c not in counts.columns:
            counts[c] = 0.0

    counts = counts[["commercial", "nightlife", "transit", "school"]].astype(float)
    counts["poi_total_count"] = counts.sum(axis=1)
    counts["poi_diversity"] = counts[["commercial", "nightlife", "transit", "school"]].apply(
        lambda r: entropy(r.to_numpy()),
        axis=1,
    )
    counts = counts.reset_index()

    feat = base.merge(counts, on="grid_id", how="left").fillna(0.0)

    for c in ["commercial", "nightlife", "transit", "school"]:
        feat[f"{c}_density"] = np.where(
            feat["grid_area"] > 0,
            feat[c] / feat["grid_area"],
            0.0,
        )

    return feat[[
        "grid_id",
        "commercial_density",
        "nightlife_density",
        "transit_density",
        "school_density",
        "poi_total_count",
        "poi_diversity",
    ]]


def add_time_feature(df):
    out = df.copy()
    out["month"] = out["month_start"].dt.month
    out["year"] = out["month_start"].dt.year
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["time_idx"] = (out["year"] - out["year"].min()) * 12 + out["month"]
    return out


def add_lag_features(df, target_col):
    out = df.sort_values(["grid_id", "month_start"]).copy()

    out[f"{target_col}_lag1"] = out.groupby("grid_id")[target_col].shift(1)
    out[f"{target_col}_lag2"] = out.groupby("grid_id")[target_col].shift(2)
    out[f"{target_col}_lag3"] = out.groupby("grid_id")[target_col].shift(3)
    out[f"{target_col}_lag6"] = out.groupby("grid_id")[target_col].shift(6)
    out[f"{target_col}_lag12"] = out.groupby("grid_id")[target_col].shift(12)

    out[f"{target_col}_roll3"] = out.groupby("grid_id")[target_col].transform(
        lambda s: s.shift(1).rolling(3, min_periods=3).mean()
    )
    out[f"{target_col}_roll6"] = out.groupby("grid_id")[target_col].transform(
        lambda s: s.shift(1).rolling(6, min_periods=6).mean()
    )
    out[f"{target_col}_roll12"] = out.groupby("grid_id")[target_col].transform(
        lambda s: s.shift(1).rolling(12, min_periods=12).mean()
    )

    out[f"{target_col}_target_next"] = out.groupby("grid_id")[target_col].shift(-1)

    return out.dropna(subset=[
        f"{target_col}_lag1",
        f"{target_col}_lag2",
        f"{target_col}_lag3",
        f"{target_col}_lag6",
        f"{target_col}_lag12",
        f"{target_col}_roll3",
        f"{target_col}_roll6",
        f"{target_col}_roll12",
        f"{target_col}_target_next",
    ])


def prepare_safety_data(target_cols=None):
    if target_cols is None:
        target_cols = ["crime_count"]

    grid = get_grid()

    static = (
        grid[["grid_id"]]
        .merge(get_building_feature(), on="grid_id", how="left")
        .merge(get_poi_feature(), on="grid_id", how="left")
        .fillna(0.0)
    )

    panel = get_monthly_panel().merge(static, on="grid_id", how="left").fillna(0.0)
    panel = add_time_feature(panel)

    result = {"panel": panel}
    for col in target_cols:
        result[col] = add_lag_features(panel, col)

    return result


if __name__ == "__main__":
    data = prepare_safety_data(["crime_count"])
    print("safety data ready")
    print("panel shape:", data["panel"].shape)
    print("crime_count shape:", data["crime_count"].shape)