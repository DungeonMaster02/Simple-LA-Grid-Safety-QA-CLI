from pathlib import Path
import json
import sys
import subprocess
from crime_data_processing import combine
from map_dividision import create_grid
from prepare_agent_context import prepare_context
import safety_main

def ensure_inputs():
    # check the minimum required input files
    building_shp = Path("../data/Building_Footprints-shp/building.shp")
    city_boundary_shp = Path("../data/City_Boundary/City_Boundary.shp")

    crime_data1 = Path("../data/Crime_Data_from_2010_to_2019.csv")
    crime_data2 = Path("../data/Crime_Data_from_2020_to_2024.csv")
    # or
    combined_crime = Path("../data/Crime_Data_from_2010_to_2024.csv")

    if not building_shp.exists():
        raise FileNotFoundError(f"Missing building shapefile: {building_shp}")

    if not city_boundary_shp.exists():
        raise FileNotFoundError(f"Missing city boundary shapefile: {city_boundary_shp}")

    if not crime_data1.exists():
        if not combined_crime.exists():
            raise FileNotFoundError("Missing 2010-2019 crime CSV files.")
    
    if not crime_data2.exists():
        if not combined_crime.exists():
            raise FileNotFoundError("Missing 2020-2024 crime CSV files.")

def check_stage():
    data_dir = Path("../data")
    output_dir = Path("../output")
    combined_crime = data_dir / "Crime_Data_from_2010_to_2024.csv"
    grid_shp = data_dir / "City_Boundary/LA_400m_grid.shp"
    panel = data_dir / "monthly_crime_panel.csv"
    
    future_pred = output_dir / "future_monthly_predictions_2025_to_2026.csv"
    final_grid = output_dir / "final_grid_safety_2026.csv"
    grid_scores = output_dir / "grid_scores.csv"

    agent_context = output_dir / "agent_grid_context_basic.json"
    agent_profile = output_dir / "agent_grid_profile_index_2026.json"
    agent_prompt = output_dir / "agent_grid_prompt_snippet.txt"

    # return the first missing stage
    if not grid_shp.exists():
        return "grid"

    if not combined_crime.exists():
        return "crime"

    if not panel.exists():
        return "prepare"

    if not future_pred.exists() or not final_grid.exists() or not grid_scores.exists():
        return "safety"

    if not agent_context.exists() or not agent_profile.exists() or not agent_prompt.exists():
        return "context"

    return "agent"

def run_pipeline():
    while True:
        stage = check_stage()

        if stage == "grid":
            print("Missing grid data, running map_dividision.py ...")
            subprocess.run([sys.executable, "map_dividision.py"], check=True)
        elif stage == "crime":
            print("Missing combined crime data, running crime_data_processing.py ...")
            subprocess.run([sys.executable, "crime_data_processing.py"], check=True)

        elif stage == "prepare":
            print("Missing monthly panel data, running safety_model_prepare.py ...")
            subprocess.run([sys.executable, "safety_model_prepare.py"], check=True)

        elif stage == "safety":
            print("Missing safety model outputs, running safety_main.py ...")
            subprocess.run([sys.executable, "safety_main.py"], check=True)

        elif stage == "context":
            print("Missing agent context files, running prepare_agent_context.py ...")
            subprocess.run([sys.executable, "prepare_agent_context.py"], check=True)

        elif stage == "agent":
            break

        else:
            raise RuntimeError(f"Unknown stage: {stage}")

def main():
    ensure_inputs()
    run_pipeline()
    subprocess.run([sys.executable, "agent.py"], check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)