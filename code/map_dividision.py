import geopandas as gpd
import shapely 
import numpy as np

def create_grid():
    gdf = gpd.read_file("../data/City_Boundary/City_Boundary.shp")
    # print(gdf.crs) # check the original projection (EPSG:3857)
    gdf_proj = gdf.to_crs(epsg=32611) #project it to UTM 11N, unit: meter
    xmin, ymin, xmax, ymax = gdf_proj.total_bounds
    grid_size = 400

    grid_cells = list()
    for x0 in np.arange(xmin,xmax,grid_size): #arange can process float
        for y0 in np.arange(ymin,ymax,grid_size):
            x1 = x0 + grid_size
            y1 = y0 + grid_size
            cell = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            grid_cells.append(cell)
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=gdf_proj.crs)
    result = gpd.clip(grid_gdf, gdf_proj) #clip grid
    result.to_file("../data/City_Boundary/LA_400m_grid.shp")
    

if __name__=="__main__":
    create_grid()