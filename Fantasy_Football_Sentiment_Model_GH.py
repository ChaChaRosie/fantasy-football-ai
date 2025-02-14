#!/usr/bin/env python3
"""
DFS Desirability Script
=======================
This script reads a list of players from an Excel file (containing Name, Pos, Team, etc.),
extracts content from multiple PDF documents, queries the OpenAI API for each player's DFS
desirability, and consolidates the results into a CSV.

Configuration:
--------------
Set the following environment variables (or edit the default paths below):
  - OPENAI_API_KEY (required)
  - INPUT_EXCEL_FILE (e.g., Player_Names.xlsx)
  - OUTPUT_CSV_FILE (e.g., player_output.csv)
  - PDF_PATHS_PHASE1 (comma-separated paths for docs 1-3)
  - PDF_PATHS_PHASE2 (comma-separated paths for docs 4-6)

Usage:
------
1. Install dependencies: 
    pip install openai pandas PyPDF2 xlrd openpyxl
2. Set your environment variables, e.g.:
    export OPENAI_API_KEY="YOUR-KEY-HERE"
    export INPUT_EXCEL_FILE="Player_Names.xlsx"
    export OUTPUT_CSV_FILE="player_output.csv"
    export PDF_PATHS_PHASE1="doc1.pdf,doc2.pdf,doc3.pdf"
    export PDF_PATHS_PHASE2="doc4.pdf,doc5.pdf,doc6.pdf"
3. Run the script:
    python dfs_desirability.py
"""

import os
import csv
import re
import openai
import pandas as pd
import PyPDF2

# ---------------------------------------------------------------------------
# 1) GET ENVIRONMENT VARIABLES OR USE DEFAULT PLACEHOLDERS
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
openai.api_key = OPENAI_API_KEY

# File paths
input_excel_file = os.environ.get("INPUT_EXCEL_FILE", "Player_Names.xlsx")
output_csv_file = os.environ.get("OUTPUT_CSV_FILE", "player_output.csv")

# List of PDF file paths for the FIRST phase (docs 1, 2, 3)
phase1_env = os.environ.get("PDF_PATHS_PHASE1", "doc1.pdf,doc2.pdf,doc3.pdf")
pdf_file_paths_phase1 = [p.strip() for p in phase1_env.split(",")]

# List of PDF file paths for the SECOND phase (docs 4, 5, 6)
phase2_env = os.environ.get("PDF_PATHS_PHASE2", "doc4.pdf,doc5.pdf,doc6.pdf")
pdf_file_paths_phase2 = [p.strip() for p in phase2_env.split(",")]


# ---------------------------------------------------------------------------
# 2) DEFINE SHARED FUNCTIONS AND PROMPTS
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path):
    """Extract text from a PDF file using PyPDF2."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
    except FileNotFoundError:
        print(f"[WARN] File not found: {pdf_path}")
    except Exception as e:
        print(f"[ERROR] Failed to read {pdf_path}: {e}")
    return text

# Prompt template for asking GPT about player performance/desirability
prompt_template = """
You are an assistant tasked with analyzing players for Daily Fantasy Sports (DFS). Each player is assigned a salary, and the goal is to create a lineup under the salary cap. 
The key is to identify players who provide the best value based on their salary, meaning their potential production relative to cost. Use the provided articles and content to analyze player performance and assign an appropriate DFS desirability score.

List of Articles Content:
[List of Articles and Content]

For the player [insert player name], who plays as a [insert position] for [insert team], follow these instructions:

1. Search the articles and content for any mentions of the player [insert player name] and summarize any discussion related to the player's performance, their role, their health status, DFS value, defense they are up against, or any other factors relevant to DFS.
2. Analyze the player's performance in DFS based on the information available, considering factors such as salary, expected production, matchup difficulty, injuries, recent form, and any other relevant data.
3. Assign a numeric desirability score for the player in the context of DFS lineups, on a scale from 1 to 100. A score of 100 indicates the player provides the highest potential value, while a lower score suggests less value for DFS lineups.
4. If the player is not mentioned in the articles and content, simply state "Not mentioned" for both the Notes and the Desirability Score.

**Important:**
- Output **only** the table, in the format below.
- The table **must** have three columns: Player Name, Notes, Desirability Score.
- No additional text outside the table is allowed.

**Format the response strictly as follows:**

| Player Name | Notes (5-6 sentences about performance) | Desirability Score (1-100) |
|-------------|----------------------------------------|----------------------------|
| [insert player name] | [Performance overview and DFS value, or "Not mentioned"] | [Desirability score, or "Not mentioned"] |
"""

# Prompt template for consolidated notes
consolidated_notes_prompt = """
You are an objective sports analyst tasked with summarizing the key points about a player's performance in Daily Fantasy Sports (DFS) based on the aggregated notes provided. Your summary should include all unique points, covering both pros and cons, without omitting any critical information. The summary should be no more than 10 sentences.

Aggregated Notes:
[Aggregated Notes]

Provide a concise summary that helps in evaluating the player's DFS potential.

**Important:**
- Output **only** the summary text.
- Do not include any additional commentary or formatting.
"""

def get_player_info(player_name, player_position, player_team, articles_text):
    """Call OpenAI ChatCompletion to get DFS desirability info for a player."""
    prompt = prompt_template.replace("[insert player name]", player_name)\
                            .replace("[insert position]", player_position)\
                            .replace("[insert team]", player_team)
    full_prompt = prompt.replace("[List of Articles and Content]", articles_text)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        if 'choices' in response and response['choices']:
            return response['choices'][0]['message']['content']
        else:
            return "Error: No response"
    except Exception as e:
        print(f"API call error: {e}")
        return "Error: API call failed"

def parse_response(player_name, response):
    """
    Parse GPT's table response to extract the Notes and Desirability Score for the
    specified player_name. Returns ("Not mentioned", "Not mentioned") if not found.
    """
    try:
        lines = response.strip().split('\n')
        for line in lines:
            if player_name.lower() in line.lower():
                parts = line.strip('|').split('|')
                if len(parts) >= 3:
                    notes = parts[1].strip()
                    desirability_score = parts[2].strip()
                    return notes, desirability_score
        return "Not mentioned", "Not mentioned"
    except Exception as e:
        print(f"Error parsing response: {e}")
        return "Error parsing", "Error parsing"

def generate_consolidated_notes(aggregated_notes):
    """
    Produce a short summary of the aggregated notes by calling OpenAI with a consolidation prompt.
    """
    if not aggregated_notes.strip() or aggregated_notes.strip().lower() == "not mentioned":
        return "Not mentioned"
    prompt = consolidated_notes_prompt.replace("[Aggregated Notes]", aggregated_notes)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        if 'choices' in response and response['choices']:
            return response['choices'][0]['message']['content'].strip()
        else:
            return "Error: No response"
    except Exception as e:
        print(f"API call error: {e}")
        return "Error: API call failed"


# ---------------------------------------------------------------------------
# 3) FIRST PHASE: PROCESS DOC1, DOC2, DOC3 AND CREATE INITIAL CSV
# ---------------------------------------------------------------------------
def main():
    """Main function orchestrating the two-phase DFS desirability analysis."""

    # Read the player list (must contain columns 'Name', 'Pos', 'Team'; 'DFS ID' is optional)
    df_players = pd.read_excel(input_excel_file)
    player_names = df_players['Name'].tolist()
    player_positions = df_players['Pos'].tolist()
    player_teams = df_players['Team'].tolist()

    # Create a new CSV with columns for doc1–doc3 + aggregated notes
    header = [
        "Player Name",
        "Desirability Score Document 1",
        "Desirability Score Document 2",
        "Desirability Score Document 3",
        "Aggregated Notes"
    ]

    with open(output_csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        # For each player, get their scores/notes from doc1, doc2, doc3
        for idx in range(len(player_names)):
            player_name = player_names[idx]
            player_position = player_positions[idx]
            player_team = player_teams[idx]
            print(f"[Phase 1] Processing player: {player_name} ({player_position}, {player_team})...")

            desirability_scores = []
            notes_list = []

            # Phase 1: doc1, doc2, doc3
            for doc_idx, pdf_path in enumerate(pdf_file_paths_phase1, start=1):
                print(f"  -> Checking PDF (Phase 1) #{doc_idx}: {pdf_path}")
                articles_text = extract_pdf_text(pdf_path)
                response_text = get_player_info(player_name, player_position, player_team, articles_text)
                notes, desirability_score = parse_response(player_name, response_text)
                desirability_scores.append(desirability_score)
                notes_list.append(notes)

            # Combine all notes into one aggregated string
            aggregated_notes = ' '.join(notes_list)

            row = [
                player_name,
                desirability_scores[0],
                desirability_scores[1],
                desirability_scores[2],
                aggregated_notes
            ]
            writer.writerow(row)

    print("[INFO] Phase 1 complete. CSV created with doc1–doc3 results.")

    # -----------------------------------------------------------------------
    # 4) SECOND PHASE: READ THE CSV, PROCESS DOC4, DOC5, DOC6, UPDATE CSV
    # -----------------------------------------------------------------------
    df_output = pd.read_csv(output_csv_file)

    # Ensure columns for doc4–doc6 exist
    for i in range(4, 7):
        col_name = f"Desirability Score Document {i}"
        if col_name not in df_output.columns:
            df_output[col_name] = ""

    # Ensure a column for Consolidated Notes
    if "Consolidated Notes" not in df_output.columns:
        df_output["Consolidated Notes"] = ""

    # For DFS ID, build a dictionary if the column exists in Excel
    dfs_id_map = {}
    if "DFS ID" in df_players.columns:
        dfs_id_map = dict(zip(df_players["Name"], df_players["DFS ID"]))

    # Process new documents (doc4, doc5, doc6)
    for idx in range(len(player_names)):
        player_name = player_names[idx]
        player_position = player_positions[idx]
        player_team = player_teams[idx]
        print(f"[Phase 2] Processing player: {player_name} ({player_position}, {player_team})...")

        desirability_scores = []
        notes_list = []

        for doc_idx, pdf_path in enumerate(pdf_file_paths_phase2, start=4):
            print(f"  -> Checking PDF (Phase 2) #{doc_idx}: {pdf_path}")
            articles_text = extract_pdf_text(pdf_path)
            response_text = get_player_info(player_name, player_position, player_team, articles_text)
            notes, desirability_score = parse_response(player_name, response_text)
            desirability_scores.append(desirability_score)
            notes_list.append(notes)

        # Update the DataFrame with the new desirability scores
        df_output.loc[df_output["Player Name"] == player_name, "Desirability Score Document 4"] = desirability_scores[0]
        df_output.loc[df_output["Player Name"] == player_name, "Desirability Score Document 5"] = desirability_scores[1]
        df_output.loc[df_output["Player Name"] == player_name, "Desirability Score Document 6"] = desirability_scores[2]

        # Update the Aggregated Notes (append new notes if not "Not mentioned")
        existing_notes = df_output.loc[df_output["Player Name"] == player_name, "Aggregated Notes"].values[0]
        new_notes_valid = [note for note in notes_list if note not in ["Not mentioned", "Error parsing"]]
        new_notes_str = ' '.join(new_notes_valid).strip()

        if new_notes_str:
            aggregated_notes = (existing_notes + ' ' + new_notes_str).strip()
        else:
            aggregated_notes = existing_notes

        df_output.loc[df_output["Player Name"] == player_name, "Aggregated Notes"] = aggregated_notes

        # Generate consolidated notes from the aggregated notes
        consolidated = generate_consolidated_notes(aggregated_notes)
        df_output.loc[df_output["Player Name"] == player_name, "Consolidated Notes"] = consolidated

    # Add the DFS ID column to df_output if present
    if dfs_id_map:
        df_output["DFS ID"] = df_output["Player Name"].map(dfs_id_map)

    # Reorder columns
    desired_order = [
        "DFS ID",
        "Player Name",
        "Desirability Score Document 1",
        "Desirability Score Document 2",
        "Desirability Score Document 3",
        "Desirability Score Document 4",
        "Desirability Score Document 5",
        "Desirability Score Document 6",
        "Aggregated Notes",
        "Consolidated Notes"
    ]

    existing_columns = df_output.columns.tolist()
    desired_order_filtered = [col for col in desired_order if col in existing_columns]
    remaining_columns = [col for col in existing_columns if col not in desired_order_filtered]
    new_column_order = desired_order_filtered + remaining_columns
    df_output = df_output[new_column_order]

    # Save the updated CSV
    df_output.to_csv(output_csv_file, index=False)
    print("[INFO] Phase 2 complete. CSV now includes doc4–doc6 data, aggregated & consolidated notes, and DFS ID if available.")


if __name__ == "__main__":
    main()
