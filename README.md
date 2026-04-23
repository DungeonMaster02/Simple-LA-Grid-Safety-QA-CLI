# Crime Safety Grid Analysis & Q&A System

A machine learning-based system for analyzing crime data, predicting safety risks for geographic grids, and answering user questions about area safety using an AI agent.

## Project Structure

```
Project5/
├── code/                          # Python source code
│   ├── agent.py                   # Q&A system (user-facing interface)
│   ├── main.py                    # Full pipeline orchestration
│   ├── safety_main.py             # Model training & safety scoring
│   ├── safety_model_prepare.py    # Data preparation for models
│   ├── prepare_agent_context.py   # Agent context generation
│   ├── crime_data_processing.py   # Crime data merging utility
│   ├── map_dividision.py          # Grid creation utility
│   ├── requirements.txt     # Python dependencies
│   └── .env                       # Environment variables (not in repo)
│
├── data/                          # Data directory (kept for reference)
│   ├── City_Boundary/             # LA city boundary & 400m grid shapefile
│   ├── Building_Footprints-shp/   # Building footprint data
│   ├── osm_chunks/                # OpenStreetMap chunks
│   ├── Crime_Data_from_2010_to_2024.csv    # Crime incident records
│   ├── monthly_crime_panel.csv          # Aggregated monthly crime panel
│   └── osm_raw_buffer400.gpkg           # OSM buffer dataset
│
├── output/                        # Generated outputs (git-ignored)
│   ├── grid_scores.csv
│   ├── future_monthly_predictions_2025_to_2026.csv
│   ├── final_grid_safety_2026.csv
│   └── agent_*.json               # Agent context files
│
└── .gitignore                     # Git ignore rules
```

## Setup

### 1. Install Dependencies

```bash
cd code
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the `code/` directory:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

For local Ollama models, no API key is needed, but you need to run your ollama serve and make sure llama3.1:latest model is in ollama (check it by ollama list in terminal)

### 3. Data Requirements

Place the following in the `data/` directory:
- `Crime_Data_from_2010_to_2019.csv` - from https://data.lacity.org/Public-Safety/Crime-Data-from-2010-to-2019/63jg-8b9z/about_data
- `Crime_Data_from_2020_to_2024.csv` - from https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-2024/2nrs-mtv8/about_data
- `City_Boundary/City_Boundary.shp` - from https://geohub.lacity.org/datasets/lahub::city-boundary/explore
- `Building_Footprints-shp/building.shp` - from https://geohub.lacity.org/datasets/lahub::building-footprints/explore?location=34.018387%2C-118.410168%2C10

## Usage

### Recommended: Run Full Pipeline + Agent in One Command

```bash
cd code
python main.py
```

`main.py` is the orchestrator. It will:
1. Validate required inputs.
2. Detect missing pipeline stages by checking expected files.
3. Run only missing scripts in order:
  - `map_dividision.py`
  - `crime_data_processing.py`
  - `safety_model_prepare.py`
  - `safety_main.py`
  - `prepare_agent_context.py`
4. Launch `agent.py` automatically when preparation is complete.

This makes reruns efficient: completed stages are skipped based on existing outputs.

### Run Q&A Agent Only

If outputs are already prepared, you can directly start the QA interface:

```bash
cd code
python agent.py
```

Select LLM provider (1=Gemini API, 2=Local Ollama), then ask questions:

```
You> Is Downtown LA safe for walking at night?
Agent> [Safety assessment based on grid data + LLM explanation]
```

### Run Stage-by-Stage (Debug Mode)

Use this when you want to inspect each stage manually:

```bash
cd code
python map_dividision.py
python crime_data_processing.py
python safety_model_prepare.py
python safety_main.py
python prepare_agent_context.py
python agent.py
```

## Pipeline Outputs

| Path | Produced by | Purpose |
|------|-------------|---------|
| `data/City_Boundary/LA_400m_grid.shp` | `map_dividision.py` | 400m analysis grid |
| `data/Crime_Data_from_2010_to_2024.csv` | `crime_data_processing.py` | Combined crime source data |
| `data/monthly_crime_panel.csv` | `safety_model_prepare.py` | Cached monthly panel for modeling |
| `output/future_monthly_predictions_2025_to_2026.csv` | `safety_main.py` | Monthly forecasts |
| `output/final_grid_safety_2026.csv` | `safety_main.py` | Annual aggregated safety metrics |
| `output/grid_scores.csv` | `safety_main.py` | Final grid scores used by agent |
| `output/agent_grid_context_basic.json` | `prepare_agent_context.py` | Compact context for QA |
| `output/agent_grid_profile_index_2026.json` | `prepare_agent_context.py` | Per-grid profile index |
| `output/agent_grid_prompt_snippet.txt` | `prepare_agent_context.py` | Prompt-ready text snippet |

## Model Overview

- **Training Data**: 2010-2024 crime incidents
- **Geographic Unit**: 400m × 400m grids across LA
- **Target Variables**: 
  - Property Crimes
  - Violence Crimes
- **Model**: XGBoost with 12-month lag features
- **Predictions**: 2025-2026 monthly risk forecasts

## Data Processing Details

### 1. Spatial Grid Construction (`map_dividision.py`)

- Reads LA city boundary shapefile and reprojects to `EPSG:32611` (UTM Zone 11N, meter-based).
- Builds a fishnet with fixed `400m × 400m` cell size over the full boundary bounding box.
- Clips fishnet by city boundary polygon to remove outside cells.
- Saves final analysis grid to `data/City_Boundary/LA_400m_grid.shp`.

### 2. Crime Data Harmonization (`crime_data_processing.py` + `safety_model_prepare.py`)

- Merges source files (`2010-2019` + `2020-2024`) and removes duplicated incidents by `DR_NO`.
- Uses the following fields for modeling panel generation:
  - `DR_NO`
  - `LAT`, `LON`
  - `DATE OCC`
  - `Crm Cd Desc`
- Cleans incidents before spatial aggregation:
  - Parses `LAT/LON` to numeric
  - Parses `DATE OCC` to datetime
  - Drops null/invalid coordinates and invalid timestamps
  - Removes zero coordinates `(LAT=0 or LON=0)`
- Maps crime descriptions into 2 groups:
  - `Violence`: if `Crm Cd Desc` appears in predefined violent-crime set
  - `Property`: all other records
- Converts points from `EPSG:4326` to grid CRS and does spatial join (`within`) to assign `grid_id`.
- Aggregates to monthly panel:
  - `property_crime`
  - `violence_crime`
  - `crime_count = property_crime + violence_crime`
- Expands to a complete `grid_id × month_start` panel and fills missing combinations with `0` (important for sparse time series).
- Caches panel to `data/monthly_crime_panel.csv`.

### 3. Static Spatial Features (`safety_model_prepare.py`)

- **Building features** from `Building_Footprints-shp/building.shp`:
  - `building_count`: number of building centroids within grid
  - `building_area_sum`: summed intersection area of buildings with grid
  - `building_coverage_ratio = building_area_sum / grid_area`
  - `mean_building_area = building_area_sum / building_count`
- **POI features** from OpenStreetMap (`osmnx`):
  - Queries tags: `amenity`, `shop`, `highway=bus_stop`, `public_transport`, `railway in {station, halt}`
  - Chunked download + cache:
    - Per-tile cache in `data/osm_chunks/`
    - Merged cache in `data/osm_raw_buffer400.gpkg`
  - Buffers POIs by 400m and joins grids with `intersects`
  - Classifies POIs into `commercial`, `nightlife`, `transit`, `school`
  - Creates:
    - `commercial_density`
    - `nightlife_density`
    - `transit_density`
    - `school_density`
    - `poi_total_count`
    - `poi_diversity` (entropy over POI category counts)

### 4. Time-Series Feature Engineering (`safety_model_prepare.py`)

- Adds calendar features:
  - `month`, `year`
  - `month_sin`, `month_cos` (cyclic month encoding)
  - `time_idx` (monotonic month index)
- For each target (`property_crime`, `violence_crime`) creates:
  - Lags: `lag1`, `lag2`, `lag3`, `lag6`, `lag12`
  - Rolling means using past-only window: `roll3`, `roll6`, `roll12`
  - Label: `target_next` (next-month value via `shift(-1)`)
- Drops rows that do not have complete lag/rolling history or missing `target_next`.

## Modeling Details

### 1. Training Setup (`safety_main.py`)

- Trains **two separate XGBoost regressors**:
  - Property model predicts `property_crime_target_next`
  - Violence model predicts `violence_crime_target_next`
- Default training cutoff: `train_end = 2024-11-30`.
- Feature set: all engineered columns except `grid_id`, `month_start`, and current model target column.
- Core XGBoost settings:
  - `objective=reg:squarederror`
  - `tree_method=hist`
  - `max_depth=4`
  - `eta=0.03`
  - `subsample=0.7`
  - `colsample_bytree=0.7`
  - `gamma=1`
  - `min_child_weight=10`
  - `num_boost_round=600`
  - plus regularization (`reg_alpha`, `reg_lambda`) for property model

### 2. Recursive Forecasting (2025-2026)

- Forecast horizon: `2025-01` to `2026-12` (monthly).
- Uses recursive one-step-ahead prediction:
  1. Build features for `current_month`
  2. Predict next month
  3. Append predicted month back to working panel
  4. Continue to next step
- Predictions are clipped to non-negative values.

### 3. Annual Risk & Safety Scoring

- Aggregates monthly predictions over 2026 per grid:
  - `pred_property_annual = sum(pred_property)`
  - `pred_violence_annual = sum(pred_violence)`
- Weighted risk formula:
  - `risk = 0.4 * pred_property_annual + 0.6 * pred_violence_annual`
- Relative ranking and score:
  - `risk_rank`: descending risk rank (`1` = highest risk)
  - `risk_norm`: percentile rank of risk
  - `safety_score = 100 * (1 - risk_norm)`
- Safety level bins:
  - `Very Unsafe` (<=20)
  - `Unsafe` (20-40]
  - `Moderate` (40-60]
  - `Safe` (60-80]
  - `Very Safe` (>80)
- Marks hotspot flag:
  - `hotspot_2026_top10pct = 1` if grid is in top 10% highest risk.

## Validation & Reproducibility Notes

- `code/safety_modeling.py` is an evaluation script (not part of default `main.py` pipeline):
  - Expanding-window folds from 2020 to 2024
  - Reports RMSE/MAE for property and violence models
  - Reports combined weighted-risk RMSE/MAE
  - Reports hotspot overlap metrics (`Hit Rate`, `Jaccard`)
  - Writes summary to `output/safety_folds_results.csv`
- Pipeline is cache-aware:
  - If `monthly_crime_panel.csv`, OSM cache, and output files already exist, reruns may skip regeneration.
  - To fully rebuild features from raw data, remove cached files in `data/` and `output/` before rerun.
- Dependency note:
  - `osmnx` is required by POI feature preparation logic in `safety_model_prepare.py`.

## Dependencies

See `code/requirements.txt`:
- geopandas
- shapely
- pandas
- numpy
- xgboost
- google-genai (for Gemini API)
- ollama (for local models)
- python-dotenv
