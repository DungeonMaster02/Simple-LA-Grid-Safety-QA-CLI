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
│   ├── Crime_Data_from_2010_2024.csv    # Crime incident records
│   ├── monthly_crime_panel.csv          # Aggregated monthly crime panel
│   └── osm_raw_buffer400.gpkg           # OSM buffer dataset
│
├── output/                        # Generated outputs (git-ignored)
│   ├── grid_scores.csv
│   ├── future_monthly_predictions_2025_2026.csv
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
- `Crime_Data_from_2010_2019.csv` - from https://data.lacity.org/Public-Safety/Crime-Data-from-2010-to-2019/63jg-8b9z/about_data
- `Crime_Data_from_2020_2024.csv` - from https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-2024/2nrs-mtv8/about_data
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
| `data/Crime_Data_from_2010_2024.csv` | `crime_data_processing.py` | Combined crime source data |
| `data/monthly_crime_panel.csv` | `safety_model_prepare.py` | Cached monthly panel for modeling |
| `output/future_monthly_predictions_2025_2026.csv` | `safety_main.py` | Monthly forecasts |
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