#!/usr/bin/env python3
"""
Lineup Optimization Script for Daily Fantasy Sports (DFS)
==========================================================
This script uses PuLP (a linear optimization library) to build multiple DFS 
lineups under salary constraints, aiming to maximize projected points. It 
optionally integrates AI "boost" scores from an external CSV.

Configuration:
--------------
Set (or override) the following environment variables to control file paths:
  - DATA_FOLDER     (default: "SaberFiles")
  - OUTPUT_FOLDER   (default: "SaberOptOutput")
  - AI_BOOST_FILE   (default: "player_output.csv")

Usage:
------
1. Install dependencies:
   pip install pulp pandas
   
2. Place your main DFS CSV (containing columns "Name", "Pos", "Salary", "SS Proj") 
   into DATA_FOLDER (or specify your own path).

3. (Optional) Place your AI boost CSV (player_output.csv) in the same folder 
   or point AI_BOOST_FILE to the correct path. The CSV should contain 
   "Player Name" and desirability score columns.

4. Run the script:
   python lineup_optimization.py

5. The script asks for any forced players by name. If none, press Enter.

6. Check the output CSV ("top_lineups_wide.csv") in the OUTPUT_FOLDER.
"""

import os
import pandas as pd
import pulp
import csv

# ------------------------------------------------------------------------------
# 1) LOAD CONFIGURATION FROM ENVIRONMENT OR USE DEFAULTS
# ------------------------------------------------------------------------------
DATA_FOLDER = os.environ.get("DATA_FOLDER", "SaberFiles")
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", "SaberOptOutput")
AI_BOOST_FILE = os.environ.get("AI_BOOST_FILE", "player_output.csv")

# ------------------------------------------------------------------------------
# STEP 1: READ MAIN PLAYER DATASET
# ------------------------------------------------------------------------------
# List all CSV files in DATA_FOLDER
files_in_dir = os.listdir(DATA_FOLDER)
csv_candidates = [f for f in files_in_dir if f.lower().endswith(".csv")]
if not csv_candidates:
    raise FileNotFoundError(f"No CSV files found in '{DATA_FOLDER}'. Please place a DFS CSV there.")

# Assume the first CSV found is our main DFS data
csv_path = os.path.join(DATA_FOLDER, csv_candidates[0])
print(f"[INFO] Using DFS CSV: {csv_path}")
df = pd.read_csv(csv_path)

# Required columns in the CSV
required_cols = ["Name", "Pos", "Salary", "SS Proj"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in {csv_path}.")

# If "Actual" does not exist, create it and fill with 0.0
if "Actual" not in df.columns:
    df["Actual"] = 0.0

# Drop rows missing essential data
df = df.dropna(subset=["Pos", "Salary", "SS Proj"])

# Convert columns to proper types
df["Salary"] = df["Salary"].astype(int)
df["SS Proj"] = df["SS Proj"].astype(float)
df["Actual"] = pd.to_numeric(df["Actual"], errors='coerce').fillna(0.0)

# ------------------------------------------------------------------------------
# STEP 2: READ AI BOOST FILE -> {PlayerName: AI Boost}
# ------------------------------------------------------------------------------
ai_boost_map = {}
ai_boost_path = AI_BOOST_FILE
if not os.path.isabs(ai_boost_path):
    ai_boost_path = os.path.join(".", ai_boost_path)

if os.path.isfile(ai_boost_path):
    print(f"[INFO] Reading AI boost from: {ai_boost_path}")
    ai_df = pd.read_csv(ai_boost_path)

    # Identify potential desirability columns
    desirability_cols = [
        "Desirability Score Document 1",
        "Desirability Score Document 2",
        "Desirability Score Document 3",
        "Desirability Score Document 4",
        "Desirability Score Document 5",
        "Desirability Score Document 6"
    ]
    existing_cols = [c for c in desirability_cols if c in ai_df.columns]

    if "Player Name" not in ai_df.columns:
        raise ValueError("AI Boost file must have a 'Player Name' column to match players.")

    for _, row_boost in ai_df.iterrows():
        player_name_in_boost = str(row_boost["Player Name"]).strip()
        scores = []
        for c in existing_cols:
            val = row_boost[c]
            # Parse as float if possible
            try:
                fval = float(val)
                scores.append(fval)
            except (ValueError, TypeError):
                pass
        
        ai_boost_map[player_name_in_boost] = sum(scores) / len(scores) if scores else 0.0
else:
    print(f"[WARN] AI Boost file not found at '{ai_boost_path}'. Defaulting boost to 0.0.")

# ------------------------------------------------------------------------------
# DYNAMIC FORCED PLAYERS (ENTER BY "Name" from the main file)
# ------------------------------------------------------------------------------
print("\nEnter the PLAYER NAMES you want forced into each lineup (comma-separated).")
print("If none, just press Enter.")
forced_names_input = input().strip()
if not forced_names_input:
    forced_names = []
else:
    forced_names = [nm.strip() for nm in forced_names_input.split(",")]

# ------------------------------------------------------------------------------
# HELPER: DETERMINE ELIGIBLE SLOTS FOR A GIVEN POSITION
# ------------------------------------------------------------------------------
def eligible_slots(pos_string):
    """Return a list of DFS-eligible slots for a given position string."""
    valid = set()
    for p in pos_string.split("/"):
        p = p.strip().upper()
        if p == "PG":
            valid.update(["PG", "G", "UTIL"])
        elif p == "SG":
            valid.update(["SG", "G", "UTIL"])
        elif p == "SF":
            valid.update(["SF", "F", "UTIL"])
        elif p == "PF":
            valid.update(["PF", "F", "UTIL"])
        elif p == "C":
            valid.update(["C", "UTIL"])
    return list(valid)

# ------------------------------------------------------------------------------
# BUILD THE MODEL
# ------------------------------------------------------------------------------
model = pulp.LpProblem("DK_Optimize", pulp.LpMaximize)
all_slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

# 1) CREATE BINARY DECISION VARIABLES x[(player_name, slot)]
x = {}
for i, row_main in df.iterrows():
    player_name_main = str(row_main["Name"]).strip()
    for sl in eligible_slots(row_main["Pos"]):
        x[(player_name_main, sl)] = pulp.LpVariable(f"x_{player_name_main}_{sl}", cat=pulp.LpBinary)

# 2) OBJECTIVE: MAXIMIZE SUM OF "SS Proj"
model += pulp.lpSum(
    df.loc[i, "SS Proj"] * x[(str(df.loc[i, "Name"]).strip(), slot)]
    for i in df.index
    for slot in all_slots
    if (str(df.loc[i, "Name"]).strip(), slot) in x
)

# 3) SALARY <= 50000
model += pulp.lpSum(
    df.loc[i, "Salary"] * x[(str(df.loc[i, "Name"]).strip(), slot)]
    for i in df.index
    for slot in all_slots
    if (str(df.loc[i, "Name"]).strip(), slot) in x
) <= 50000

# 4) EXACTLY 1 PLAYER PER SLOT
for sl in all_slots:
    model += pulp.lpSum(
        x[(str(df.loc[i, "Name"]).strip(), sl)]
        for i in df.index
        if (str(df.loc[i, "Name"]).strip(), sl) in x
    ) == 1

# 5) EACH PLAYER AT MOST 1 SLOT
for i in df.index:
    pname = str(df.loc[i, "Name"]).strip()
    model += pulp.lpSum(
        x[(pname, s)] for s in all_slots if (pname, s) in x
    ) <= 1

# 6) FORCE SELECTED PLAYERS
for f_name in forced_names:
    if not any((f_name, sl) in x for sl in all_slots):
        raise ValueError(f"Forced player NAME '{f_name}' not found or no eligible slots.")
    model += pulp.lpSum(x[(f_name, sl)] for sl in all_slots if (f_name, sl) in x) == 1

# ------------------------------------------------------------------------------
# SOLVE THE MODEL FOR UP TO 40 UNIQUE LINEUPS
# ------------------------------------------------------------------------------
solver = pulp.PULP_CBC_CMD(msg=0)
top_lineups = []

for lineup_rank in range(1, 41):  # e.g., generate up to 40 lineups
    result = model.solve(solver)
    obj_val = pulp.value(model.objective)
    if obj_val is None:
        print(f"No more feasible solutions after {lineup_rank - 1} lineups.")
        break

    # Gather chosen players from the solution
    chosen_rows = []
    for i in df.index:
        pname_main = str(df.loc[i, "Name"]).strip()
        sal = df.loc[i, "Salary"]
        proj = df.loc[i, "SS Proj"]
        act = df.loc[i, "Actual"]
        for sl in all_slots:
            if (pname_main, sl) in x and pulp.value(x[(pname_main, sl)]) > 0.5:
                chosen_rows.append((pname_main, df.loc[i, "Name"], sl, sal, proj, act))

    used_salary = sum(r[3] for r in chosen_rows)
    used_proj = obj_val
    used_actual = sum(r[5] for r in chosen_rows)

    # Calculate total AI Boost sum, then scale or average it as desired
    sum_ai_boost = 0.0
    for (pkey, _display_name, _slot_used, _sal, _proj, _act) in chosen_rows:
        sum_ai_boost += ai_boost_map.get(pkey, 0.0)
    # Example: dividing by 800.0 to create a small fraction
    total_ai_score = sum_ai_boost / 800.0

    lineup_data = {
        "rank": lineup_rank,
        "total_proj": used_proj,
        "total_ai_score": total_ai_score,
        "total_actual": used_actual,
        "salary_used": used_salary,
        "players": chosen_rows
    }
    top_lineups.append(lineup_data)

    # EXCLUDE this exact set of players from the next solution
    chosen_pnames = set(r[0] for r in chosen_rows)
    exclude_sum = pulp.lpSum(
        pulp.lpSum(x[(pname, sl)] for sl in all_slots if (pname, sl) in x)
        for pname in chosen_pnames
    )
    model += exclude_sum <= len(chosen_pnames) - 1

print(f"\n[INFO] Found {len(top_lineups)} unique solutions.")

# ------------------------------------------------------------------------------
# WRITE THE OUTPUT AS A "WIDE" CSV
# ------------------------------------------------------------------------------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
out_path = os.path.join(OUTPUT_FOLDER, "top_lineups_wide.csv")
slot_columns = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

header = [
    "LineupRank",
    "TotalProj",
    "TotalAIScore",
    "TotalActual",
    "SalaryUsed"
]
for sc in slot_columns:
    header += [f"{sc}_Name", f"{sc}_Sal", f"{sc}_AIBoost", f"{sc}_Proj", f"{sc}_Actual"]

with open(out_path, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for ln in top_lineups:
        row = [
            ln["rank"],
            ln["total_proj"],
            ln["total_ai_score"],
            ln["total_actual"],
            ln["salary_used"]
        ]
        # Map each slot to (Name, Salary, AI_Boost, Proj, Actual)
        slot_map = {sl: ("", 0, 0.0, 0.0, 0.0) for sl in slot_columns}
        
        for (pkey, pname, slot_used, psal, pproj, pact) in ln["players"]:
            aiboost = ai_boost_map.get(pkey, 0.0)
            slot_map[slot_used] = (pname, psal, aiboost, pproj, pact)

        for sc in slot_columns:
            (nm, sal, aiboost, proj, act) = slot_map[sc]
            row += [nm, sal, aiboost, proj, act]

        writer.writerow(row)

print(f"[INFO] Wrote {len(top_lineups)} unique lineups to '{out_path}'.")
