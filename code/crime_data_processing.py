import pandas as pd
from pathlib import Path
import geopandas as gpd

def combine(datafile1,datafile2):
    data1 = pd.read_csv(datafile1)
    data2 = pd.read_csv(datafile2)

    data1.columns = data1.columns.str.strip()
    data2.columns = data2.columns.str.strip()

    data = pd.concat([data1, data2], ignore_index=True)
    data = data.drop_duplicates(subset=["DR_NO"])

    data.to_csv("../data/Crime_Data_from_2010_2024.csv", index=False)

def get_monthly():
    data = pd.read_csv("../data/Crime_Data_from_2010_2024.csv")
    data.drop_duplicates(inplace=True)
    crime_df = data[['DR_NO','LAT','LON',"DATE OCC"]]
    crime_df['Crime_Group'] = crime_df['type'].apply(classify)
    crime_gdf = gpd.GeoDataFrame(
        crime_df,
        geometry=gpd.points_from_xy(crime_df["longitude"], crime_df["latitude"]),
        crs="EPSG:4326"
    ) #convert DataFrame into GeoDataFrame, every row is a point
    grid = gpd.read_file("../data/City_Boundary/LA_400m_grid.shp")
    grid = grid.to_crs("EPSG:32611").reset_index(drop=True) #reset the index and drop the old index since the index might be reordered by reprojection
    grid["grid_id"] = grid.index + 1

    crime_gdf = crime_gdf.to_crs("EPSG:32611")

    joined = gpd.sjoin( #sjoin: connect tables by spatial position
        crime_gdf, #first(in the left) table
        grid[["grid_id","geometry"]], #get grid_id and geometry columns to match the cells, get multiple columns: [[]]
        how = "left", #the left table is the main table, if a point doesn't match any cell, it remains
        predicate = "within" #rule: if the point is inside the cell
    )
    matched = joined.dropna(subset=["grid_id"]).copy()
    matched["date"] = pd.to_datetime(matched["date"])
    matched["month_start"] = matched["date"].dt.to_period("M").dt.to_timestamp()
    result = (
        matched.groupby(["grid_id", "month_start", "Crime_Group"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    result["property_crime"] = result.get("Property", 0)
    result["violence_crime"] = result.get("Violence", 0)
    result["crime_count"] = result["property_crime"] + result["violence_crime"]

    result = result[["grid_id", "month_start", "crime_count", "property_crime", "violence_crime"]]
    
    # add full grid-month combinations
    month_idx = pd.date_range(result["month_start"].min(), result["month_start"].max(), freq="MS")
    full_idx = pd.MultiIndex.from_product(
        [grid["grid_id"].sort_values().unique(), month_idx],
        names=["grid_id", "month_start"]
    )
    full_df = full_idx.to_frame(index=False)

    result = full_df.merge(result, on=["grid_id", "month_start"], how="left")
    result[["crime_count", "property_crime", "violence_crime"]] = (
        result[["crime_count", "property_crime", "violence_crime"]].fillna(0)
    )
    result["grid_id"] = result["grid_id"].astype(int)
    result["month_start"] = result["month_start"].dt.strftime("%Y-%m-%d")
    result["crime_count"] = result["crime_count"].astype(int)
    result["property_crime"] = result["property_crime"].astype(int)
    result["violence_crime"] = result["violence_crime"].astype(int)

    crime_monthly = result.itertuples(index=False, name=None)
    return crime_monthly

def classify(desc):
    desc = str(desc).upper()
    violence_list = {
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
        "WEAPONS POSSESSION/BOMBING"
    }
    return 'Violence' if desc in violence_list else 'Property'

if __name__=="__main__":
    combined = Path("../data/Crime_Data_from_2010_2024.csv")
    if not combined.exists():
        datafile1 = "../data/Crime_Data_from_2010_2019.csv"
        datafile2 = "../data/Crime_Data_from_2020_2024.csv"
        combine(datafile1,datafile2)
    df = pd.read_csv("../data/Crime_Data_from_2010_2024.csv")
    # crime_types = sorted(df['Crm Cd Desc'].dropna().unique())
    # for i, crime in enumerate(crime_types):
    #     print(f"{i+1}. {crime}")
