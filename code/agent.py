from __future__ import annotations
import csv
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import ollama
from dotenv import load_dotenv

from google import genai
import geopandas as gpd
from shapely.geometry import Point


load_dotenv()


def to_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default # Handle empty strings and None as default
        return float(value)
    except (TypeError, ValueError):
        return default

def to_int(value, default=0):
    try:
        if value in (None, ""):
            return default # Handle empty strings and None as default
        return int(float(value))
    except (TypeError, ValueError):
        return default

def extract_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    text = re.sub(r"^```json\s*|^```\s*|\s*```$", "", text, flags=re.DOTALL).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}

    return {}

def extract_query_info(question: str, provider: str = "gemini", my_model: str = "gemini-2.5-flash") -> dict:
    """Extract structured information from the user's question using an LLM."""
    question = question.strip()
    if not question:
        return {
            "location_text": None,
            "crime_type": None,
            "needs_clarification": True,
            "clarification_question": "Please provide a location in your question."
        }

    prompt = f"""
        You extract structured information from a user's crime safety question.

        Return JSON only with these keys:
        - location_text
        - crime_type
        - needs_clarification
        - clarification_question

        Rules:
        1. location_text should be copied from the user's question as closely as possible.
        2. Do not expand abbreviations.
        3. Do not rewrite one place into another place.
        4. crime_type should be a short English label if mentioned, otherwise null.
        5. If no location is mentioned, set needs_clarification=true.
        6. Do not answer the question.

        Question: {question}
    """.strip()

    try:
        text = call_llm(prompt, provider, my_model)
        data = extract_json(text or "")

        location_text = data.get("location_text")
        crime_type = data.get("crime_type")
        needs_clarification = bool(data.get("needs_clarification", False))

        if not location_text:
            needs_clarification = True

        clarification_question = data.get("clarification_question")
        if needs_clarification and not clarification_question:
            clarification_question = "Please provide a specific location, such as an address, landmark, or neighborhood."

        return {
            "location_text": location_text.strip() if isinstance(location_text, str) and location_text.strip() else None,
            "crime_type": crime_type.strip() if isinstance(crime_type, str) and crime_type.strip() else None,
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_question,
        }
    
    except Exception as e:
        print(f"[extract_query_info] {provider} error: {type(e).__name__}: {e}", file=sys.stderr)
        return {
            "location_text": None,
            "crime_type": None,
            "needs_clarification": True,
            "clarification_question": f"Error processing your question (LLM unavailable). Please try again or rephrase your question.",
        }

def fetch_json(url: str, params: dict, user_agent: str = "grid-safety-cli/1.0"):
    full_url = f"{url}?{urlencode(params)}"
    req = Request(full_url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))
    
def call_llm(my_prompt: str, provider: str, my_model: str) -> str | None:
    if provider == "gemini":
        client = genai.Client()
        response = client.models.generate_content(
            model=my_model,
            contents=my_prompt,
        )
        text = getattr(response, "text", "") or "" # getattr() can avoid response.text not existing, or "" can handle responnse.text being None and assign "" instead
        return text.strip() if text else None

    if provider == "ollama":
        response = ollama.generate(
            model=my_model,
            prompt=my_prompt,
        )
        text = response["response"]
        return text.strip() if text else None

def geocode_location(location_text: str) -> dict:
    if not location_text:
        return {"ok": False, "error": "empty_location"}

    UA = "grid-safety-research-bot/1.0 (contact: pengshao@usc.edu)" # User agent to avoid 403 errors from geocoding APIs

    queries = [location_text]
    lower = location_text.lower()
    if not any(x in lower for x in ["los angeles", "california", " ca"]):
        queries.append(f"{location_text}, Los Angeles, CA") # complement the location to improve geocoding accuracy

    for query in queries:
        try: # use openstreetmap nominatim API to get data first
            data = fetch_json(
                "https://nominatim.openstreetmap.org/search",
                {"q": query, "format": "jsonv2", "limit": 1}, # get the first match for the query in json format
                user_agent=UA 
            )

            if len(data) > 0:
                item = data[0]
                return {
                    "ok": True,
                    "provider": "nominatim",
                    "lat": to_float(item.get("lat")),
                    "lon": to_float(item.get("lon")),
                    "display_name": item.get("display_name", query),
                }
        except Exception:
            pass

        try: # if nominatim fails, use US census API as a backup
            data = fetch_json(
                "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress",
                {
                    "address": query,
                    "benchmark": "Public_AR_Current",
                    "format": "json",
                },
                user_agent=UA
            )
            matches = data.get("result", {}).get("addressMatches", [])
            if matches:
                m = matches[0]
                coord = m.get("coordinates", {})
                return {
                    "ok": True,
                    "provider": "census",
                    "lat": to_float(coord.get("y")),
                    "lon": to_float(coord.get("x")),
                    "display_name": m.get("matchedAddress", query),
                }
        except Exception:
            pass

    return {"ok": False, "error": "geocoding_failed"}


def load_grid_data(grid_shp_path: Path):
    if gpd is None or Point is None:
        raise RuntimeError("Please install geopandas and shapely first.")

    if not grid_shp_path.exists():
        raise FileNotFoundError(f"Grid shapefile not found: {grid_shp_path}")

    grid = gpd.read_file(grid_shp_path)
    grid = grid.reset_index(drop=True)
    grid["grid_id"] = grid.index + 1
    grid["grid_id"] = grid["grid_id"].astype(int)
    return grid.to_crs("EPSG:4326").reset_index(drop=True)


def match_grid(lat: float, lon: float, grid_gdf) -> int | None:
    point = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]),crs="EPSG:4326").iloc[0]
    matched_grid = grid_gdf.loc[grid_gdf.geometry.intersects(point), "grid_id"]

    if not matched_grid.empty:
        return int(matched_grid.iloc[0])

    return None

def load_scores(score_csv_path: Path) -> dict[int, dict]:
    if not score_csv_path.exists():
        raise FileNotFoundError(f"Score CSV not found: {score_csv_path}")

    scores = {}
    with score_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grid_id = to_int(row.get("grid_id"))
            if grid_id <= 0:
                continue

            scores[grid_id] = {
                "grid_id": grid_id,
                "safety_score": to_float(row.get("safety_score")),
                "safety_level": (row.get("safety_level") or "Unknown").strip(),
                "risk_rank": to_int(row.get("risk_rank")),
                "hotspot_2026_top10pct": to_int(row.get("hotspot_2026_top10pct")),
                "pred_property_annual": to_float(row.get("pred_property_annual")),
                "pred_violence_annual": to_float(row.get("pred_violence_annual")),
                "risk": to_float(row.get("risk")),
            }

    if not scores:
        raise ValueError("No valid score records found.")

    return scores


def build_answer(question_info: dict, geo: dict, grid_id: int, profile: dict, total_grids: int) -> str:
    # Build the anwser
    lines = [
        "Data Summary:",
        f"- Location: {geo.get('display_name', question_info.get('location_text'))}",
        f"- Coordinates: ({geo.get('lat')}, {geo.get('lon')})",
        f"- Matched grid: {grid_id}",
        f"- Safety level: {profile.get('safety_level', 'Unknown')}",
        f"- Safety score: {profile.get('safety_score', 0.0):.2f} / 100",
        f"- Risk rank: {profile.get('risk_rank', 0)} / {total_grids} (1 = highest risk)",
        f"- Top 10% hotspot: {'Yes' if profile.get('hotspot_2026_top10pct', 0) == 1 else 'No'}",
        (
            f"- Annual risk components: "
            f"property={profile.get('pred_property_annual', 0.0):.2f}, "
            f"violence={profile.get('pred_violence_annual', 0.0):.2f}, "
            f"risk={profile.get('risk', 0.0):.4f}"
        ),
    ]

    if question_info.get("crime_type"):
        lines.append(
            f"- Crime type mentioned: {question_info['crime_type']} "
            "(the system still returns overall grid risk, not crime-type-specific prediction)"
        )

    return "\n".join(lines)

def generate_llm_explanation(question: str, geo: dict, grid_id: int, profile: dict, provider: str = "gemini", my_model: str = "gemini-2.5-flash") -> str | None:
    my_prompt = f"""
        You are a crime-safety explanation assistant.

        Your job is to answer the user's question in natural English using ONLY the provided facts.
        Do not invent any crime statistics or details that are not given.
        Do not claim certainty beyond the provided data.
        Keep the answer concise, practical, and easy to understand.

        User question:
        {question}

        Known facts:
        - Location: {geo.get('display_name')}
        - Matched grid: {grid_id}
        - Safety level: {profile.get('safety_level', 'Unknown')}
        - Safety score: {profile.get('safety_score', 0.0):.2f} / 100
        - Risk rank: {profile.get('risk_rank', 0)}
        - Top 10% hotspot: {"Yes" if profile.get("hotspot_2026_top10pct", 0) == 1 else "No"}
        - Annual property risk: {profile.get('pred_property_annual', 0.0):.2f}
        - Annual violence risk: {profile.get('pred_violence_annual', 0.0):.2f}
        - Overall risk: {profile.get('risk', 0.0):.4f}

        Instructions:
        1. Answer the user's practical question directly.
        2. Reference the hotspot and safety score if relevant.
        3. If the user asks about children, commuting alone, or personal safety decisions, give cautious practical guidance.
        4. Do not mention unavailable information.
        5. Output plain text only.
    """.strip()

    try:
        text = call_llm(my_prompt, provider, my_model)
        return text if text else None
    except Exception as e:
        print(f"[generate_llm_explanation] {provider} error: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    
def answer_question(question: str, grid_gdf, scores: dict[int, dict], provider: str, my_model: str) -> dict:
    info = extract_query_info(question, provider, my_model) # extract structured info from the question using LLM
    # check if the extraction was successful and if a location was provided
    if info is None:
        return {
            "status": "error",
            "answer": "Error processing your question. Please try again.",
        }

    if not info["location_text"]:
        return {
            "status": "needs_clarification",
            "answer": "Please provide a specific location, such as an address, landmark, or neighborhood.",
        }

    geo = geocode_location(info["location_text"])
    if not geo.get("ok"):
        return {
            "status": "geocode_failed",
            "answer": "Could not geocode the location. Please try a more specific place name or address.",
        }

    grid_id = match_grid(geo["lat"], geo["lon"], grid_gdf)
    if grid_id is None:
        return {
            "status": "grid_not_found",
            "answer": "The location is outside the study area or does not match any grid.",
        }

    profile = scores.get(grid_id)
    if not profile:
        return {
            "status": "score_not_found",
            "answer": f"Grid {grid_id} was found, but no score record is available.",
        }

    # build the factual answer and LLM explanation
    fact_answer = build_answer(info, geo, grid_id, profile, len(scores))
    llm_explanation = generate_llm_explanation(question, geo, grid_id, profile, provider, my_model)
    if llm_explanation:
        final_answer = fact_answer + "\n\nInterpretation:\n" + llm_explanation
    else:
        final_answer = fact_answer
    return {
        "status": "ok",
        "answer": final_answer
    }


def run_cli(grid_gdf, scores, provider: str, model: str):
    print("Grid Safety QA CLI")
    print("Ask a question in English. Type 'exit' to quit.")

    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "bye", "goodbye"}:
            print("Goodbye.")
            break

        result = answer_question(question, grid_gdf, scores, provider, model)
        print(f"\nAgent> {result['answer']}")

def main():
    grid_shp = Path(r"../data/City_Boundary/LA_400m_grid.shp")
    score_csv = Path(r"../output/grid_scores.csv")

    print("\nSelect LLM provider:")
    print("1. Gemini API")
    print("2. Local model")
    choice = -1
    while choice not in {"1", "2"}:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            provider = "gemini"
            model = "gemini-2.5-flash"
        elif choice == "2":
            provider = "ollama"
            model = "llama3.1:latest"
        else:
            print("Invalid choice. Please enter 1 or 2.")

    grid_gdf = load_grid_data(grid_shp)
    scores = load_scores(score_csv)

    run_cli(grid_gdf, scores, provider, model)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
