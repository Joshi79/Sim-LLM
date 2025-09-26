import logging
import sys
import os
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Any

import re
import json


os.makedirs("logs", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logger_file_path = f"logs/dqn_training_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # for SLURM
        logging.FileHandler(logger_file_path),  # separate full log file
    ],
)
logger = logging.getLogger(__name__)


def calculate_ph_rl_reward(df):
    """
    Calculate pH-RL reward based on the methodology:
    """
    df_copy = df.copy()

    required_cols = ['numberMessageReceived', 'numberMessageRead', 'numberRating']
    for col in required_cols:
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame.")

    df_copy['reward'] = 0.0

    for row in df_copy.itertuples():
        if row.numberMessageReceived > 0:
            message_read_fraction = row.numberMessageRead / row.numberMessageReceived
        else:
            message_read_fraction = 0.0

        reward = 0.5 * message_read_fraction + 0.5 * row.numberRating

        # rounded to fit the reward the same as the original implementation
        df_copy.at[row.Index, 'reward'] = round(reward, 6)

    return df_copy

def calculate_rest_columns_for_day(simulated_day, day_no, messages_received_list, user_id, action_list, day_part_list,
                                   motivational_preference, trajectories_individual):
    """
    Takes a list of 3 JSON objects (1 per day part) and computes daily stats.
    Updates each entry with derived fields, and returns the final aggregated stats dict.
    """

    stats = {
        "user_id": user_id,
        "day_part_x": 0,
        "day_no": day_no,
        "all_day_ratings": [],
        "actual_rating": 0,
        "highestRating": 0.0,
        "lowestRating": np.inf,
        "medianRating": 0.0,
        "sdRating": 0.0,
        "numberLowRating": 0,
        "numberMediumRating": 0,
        "numberHighRating": 0,
        "numberRating": 0,
        "numberMessageRead": 0,
        "numberMessageReceived": 0,
        "readAllMessage": False,
        "reward": 0,
        "rl_action": np.nan,
        "motivation_preference": "None"
    }

    if simulated_day:

        all_day_ratings = []

        for i in range(len(simulated_day)):

            entry = simulated_day[i]
            # print(entry)

            ratings_list = entry.get("mood_rating", np.nan)
            number_of_messages_read = entry.get("number_of_messages_read", np.nan)
            number_ratings = entry.get("number_of_inputed_ratings", np.nan)
            messages_received_today = messages_received_list[i]

            if isinstance(ratings_list, (list, tuple)):
                ratings = [int(r) for r in ratings_list]
            elif isinstance(ratings_list, (int, float)):
                ratings = [int(ratings_list)]
            else:
                ratings = []
            # print(ratings)
            all_day_ratings.extend(ratings)
            stats["rl_action"] = action_list[i]
            stats["day_part_x"] = day_part_list[i]

            if ratings:
                stats["actual_rating"] = ratings

                current_highest_number = int(max(ratings))


                non_zero_ratings = [r for r in ratings if r > 0]
                current_lowest_number = int(min(non_zero_ratings)) if non_zero_ratings else np.inf

                if current_highest_number > stats["highestRating"]:
                    stats["highestRating"] = current_highest_number

                if stats["lowestRating"] == 0:
                    stats["lowestRating"] = np.inf

                if current_lowest_number < stats["lowestRating"]:
                    stats["lowestRating"] = current_lowest_number

                # make sure that 0 ratings do not affect the median
                non_zero_ratings_median_std = [r for r in all_day_ratings if r > 0]
                stats["medianRating"] = np.median(non_zero_ratings_median_std) if non_zero_ratings_median_std else 0.0

                stats["sdRating"] = np.std(non_zero_ratings_median_std) if len(non_zero_ratings_median_std) > 1 else 0.0

                stats["numberLowRating"] += sum(1 for r in ratings if 0 < r <= 2)
                stats["numberMediumRating"] += sum(1 for r in ratings if 3 <= r <= 5)
                stats["numberHighRating"] += sum(1 for r in ratings if r >= 6)

                stats["numberMessageRead"] = number_of_messages_read
                stats["numberMessageReceived"] = messages_received_today

                stats["readAllMessage"] = (1 if stats["numberMessageRead"] == stats["numberMessageReceived"] else 0)

                if 0 not in ratings:
                    stats["numberRating"] += len(ratings)
                else:
                    stats["numberRating"] += 0

                if stats["numberLowRating"] == np.inf:
                    stats["numberLowRating"] = 0

                if stats["lowestRating"] == np.inf:
                    stats["lowestRating"] = 0

                day_trajectory = pd.DataFrame([stats], columns=trajectories_individual.columns)

                # Calculate pH-RL reward and add it to the DataFrame
                day_trajectories_with_rewards = calculate_ph_rl_reward(day_trajectory)

                trajectories_individual = pd.concat([trajectories_individual, day_trajectories_with_rewards], ignore_index=True)

    # Fallback if there is an error in the simulation
    if len(simulated_day) < 3:
        # Fill in missing day parts with NaN
        for i in range(len(simulated_day), 3):
            stats["day_part_x"] = np.nan
            stats["rl_action"] = np.nan
            stats["actual_rating"] = np.nan
            stats["highestRating"] = np.nan
            stats["lowestRating"] = np.nan
            stats["medianRating"] = np.nan
            stats["sdRating"] = np.nan
            stats["numberLowRating"] = np.nan
            stats["numberMediumRating"] = np.nan
            stats["numberHighRating"] = np.nan
            stats["numberRating"] = np.nan
            stats["numberMessageRead"] = np.nan
            stats["readAllMessage"] = False

            day_trajectory = pd.DataFrame([stats], columns=trajectories_individual.columns)
            trajectories_individual = pd.concat([trajectories_individual, day_trajectory], ignore_index=True)

    return trajectories_individual



def clean_llm_output(json_string):
    """
    Cleaning the LLM output to be a valid Json
    Is here for a fallback if the pydantic output parser fails
    """
    clean_string = re.sub(r'```json\s*|\s*```+', '', json_string)


    if not clean_string.strip().startswith('['):
        clean_string = '[' + clean_string.strip()

    if not clean_string.strip().endswith(']'):
        clean_string = clean_string.strip() + ']'

    clean_string = re.sub(r'\]\s*\[', ',', clean_string)

    try:

        objects = json.loads(clean_string)
        return objects
    except json.JSONDecodeError:
        objects = []
        decoder = json.JSONDecoder()
        pos = 0

        while pos < len(clean_string):
            try:
                obj, pos = decoder.raw_decode(clean_string, pos)
                if isinstance(obj, dict):
                    objects.append(obj)
                elif isinstance(obj, list):
                    objects.extend(obj)
            except json.JSONDecodeError:
                pos += 1

        return objects





################### Functions for 8B model #########################

def validate_llm_output(output_list, expected_length):
    """
    Needed for the hallucainations of the 8B model
    """
    if len(output_list) == expected_length:
        return output_list

    logger.warning(
        f"LLM output validation failed. "
        f"Expected {expected_length} entries, but got {len(output_list)}. "
        f"Adjusting list size."
    )

    # Pad with default entries if the list is too short
    while len(output_list) < expected_length:
        output_list.append({
            "mood_rating": [],
            "number_of_messages_read": 0,
            "number_of_inputed_ratings": 0
        })

    # Truncate if the list is too long
    if len(output_list) > expected_length:
        output_list = output_list[:expected_length]

    return output_list

def parse_ratings(ratings_list):
    """
    Safely parses the mood_rating value, which could be a list, a single number,
    or a malformed string from the LLM output.
    """
    if isinstance(ratings_list, list):
        return [int(r) for r in ratings_list if isinstance(r, (int, float))]
    if isinstance(ratings_list, (int, float)):
        return [int(ratings_list)]
    return []

def validate_and_fix_object(obj):
    """
    Validate and fix a parsed object to ensure it has all required fields.
    """
    if not obj: return False
    if "mood_rating" not in obj: obj["mood_rating"] = []
    if "number_of_messages_read" not in obj: obj["number_of_messages_read"] = 0
    if "number_of_inputed_ratings" not in obj:
        obj["number_of_inputed_ratings"] = len(obj["mood_rating"]) if obj["mood_rating"] else 0

    if not isinstance(obj["mood_rating"], list):
        obj["mood_rating"] = []

    obj["mood_rating"] = [r for r in obj["mood_rating"] if isinstance(r, (int, float)) and 0 <= r <= 7]
    return True

def parse_line_by_line(text):
    """
    Parse text line by line to extract JSON-like data.
    """
    objects = []
    current_obj = {}
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('{'):
            if current_obj and validate_and_fix_object(current_obj):
                objects.append(current_obj)
            current_obj = {}

        if '"number_of_inputed_ratings"' in line:
            match = re.search(r'(\d+)', line)
            if match: current_obj["number_of_inputed_ratings"] = int(match.group(1))

        if '"mood_rating"' in line:
            match = re.search(r'\[(.*?)\]', line)
            if match:
                ratings = re.findall(r'\d+', match.group(1))
                current_obj["mood_rating"] = [int(r) for r in ratings if 0 <= int(r) <= 7]
            else:
                current_obj["mood_rating"] = []

        if '"number_of_messages_read"' in line:
            match = re.search(r'(\d+)', line)
            if match: current_obj["number_of_messages_read"] = int(match.group(1))

        if line.endswith('}'):
            if validate_and_fix_object(current_obj):
                objects.append(current_obj)
            current_obj = {}

    if current_obj and validate_and_fix_object(current_obj):
        objects.append(current_obj)
    return objects

def extract_key_value_pairs(text):
    """
    Extract key-value pairs from potentially malformed JSON text.
    """
    result = {"mood_rating": [], "number_of_messages_read": 0, "number_of_inputed_ratings": 0}
    try:
        mood_match = re.search(r'"mood_rating"\s*:\s*\[(.*?)\]', text)
        if mood_match:
            ratings = re.findall(r'\d+', mood_match.group(1))
            result["mood_rating"] = [int(r) for r in ratings if 0 <= int(r) <= 7]

        msg_read_match = re.search(r'"number_of_messages_read"\s*:\s*(\d+)', text)
        if msg_read_match:
            result["number_of_messages_read"] = int(msg_read_match.group(1))

        num_ratings_match = re.search(r'"number_of_inputed_ratings"\s*:\s*(\d+)', text)
        if num_ratings_match:
            result["number_of_inputed_ratings"] = int(num_ratings_match.group(1))
        elif result["mood_rating"]:
            result["number_of_inputed_ratings"] = len(result["mood_rating"])
    except Exception as e:
        logger.warning(f"Error extracting key-value pairs: {e}")
    return result

def fix_truncated_json(json_str):
    """
    Attempt to fix truncated JSON by adding missing closing brackets/braces.
    """
    try:
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)

        obj = json.loads(json_str)
        if isinstance(obj, dict):
            if "mood_rating" not in obj: obj["mood_rating"] = []
            if "number_of_messages_read" not in obj: obj["number_of_messages_read"] = 0
            if "number_of_inputed_ratings" not in obj: obj["number_of_inputed_ratings"] = 0
            return obj
    except:
        pass
    return extract_key_value_pairs(json_str)
def clean_llm_output_8B(json_string):
    """
    Robust parser for LLM output that handles truncated and malformed JSON.
    """
    json_string = json_string.replace("END OF OUTPUT", "")
    clean_string = re.sub(r'```json\s*|\s*```+', '', json_string)
    objects = []
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    potential_objects = re.findall(pattern, clean_string, re.DOTALL)

    for obj_str in potential_objects:
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict):
                if "mood_rating" not in obj:
                    obj["mood_rating"] = []
                if "number_of_messages_read" not in obj:
                    obj["number_of_messages_read"] = 0
                if "number_of_inputed_ratings" not in obj:
                    if obj["mood_rating"]:
                        obj["number_of_inputed_ratings"] = len(obj["mood_rating"])
                    else:
                        obj["number_of_inputed_ratings"] = 0
                objects.append(obj)
        except json.JSONDecodeError:
            fixed_obj = fix_truncated_json(obj_str)
            if fixed_obj:
                objects.append(fixed_obj)

    if not objects:
        objects = parse_line_by_line(clean_string)

    while len(objects) < 3:
        objects.append({
            "mood_rating": [],
            "number_of_messages_read": 0,
            "number_of_inputed_ratings": 0
        })

    if len(objects) > 3:
        logger.warning(f"LLM output has {len(objects)} entries, truncating to 3")
        objects = objects[:3]

    return objects



def calculate_rest_columns_8B(simulated_day,day_no,messages_received_list,user_id,action_list,day_part_list,motivational_preference,trajectories_individual):
    """
    Robust version of calculate_rest_columns_for_day that handles edge cases.
    """

    # extra checks because it starts to hallucinates
    if len(messages_received_list) != 3:
        raise ValueError("messages_received_list must have exactly 3 elements")
    if len(action_list) != 3:
        raise ValueError("action_list must have exactly 3 elements")
    if len(day_part_list) != 3:
        raise ValueError("day_part_list must have exactly 3 elements")

    if simulated_day is None:
        simulated_day = []

    # Pad with empty dictionaries if the list is shorter than 3
    while len(simulated_day) < 3:
        simulated_day.append({})

    base_stats = {
        "user_id": user_id, "day_part_x": 0, "day_no": day_no, "all_day_ratings": [],
        "actual_rating": 0, "highestRating": 0.0, "lowestRating": 0.0,
        "medianRating": 0.0, "sdRating": 0.0, "numberLowRating": 0,
        "numberMediumRating": 0, "numberHighRating": 0, "numberRating": 0,
        "numberMessageRead": 0, "numberMessageReceived": 0, "readAllMessage": False,
        "reward": 0, "rl_action": np.nan, "motivation_preference": motivational_preference
    }

    all_day_ratings = []
    day_trajectories = []

    for i in range(3):
        stats = base_stats.copy()
        stats["day_part_x"] = day_part_list[i]
        stats["rl_action"] = action_list[i]
        stats["numberMessageReceived"] = messages_received_list[i]

        entry = simulated_day[i] if i < len(simulated_day) else {}

        if entry:
            ratings_list = entry.get("mood_rating", [])
            number_of_messages_read = entry.get("number_of_messages_read", 0)
            number_ratings = entry.get("number_of_inputed_ratings", 0)

            ratings = parse_ratings(ratings_list)

            if ratings:
                valid_ratings = [r for r in ratings if 0 < r <= 7]
                if valid_ratings:
                    all_day_ratings.extend(valid_ratings)
                    stats["actual_rating"] = valid_ratings
                    stats["highestRating"] = max(valid_ratings)
                    stats["lowestRating"] = min(valid_ratings)
                    stats["numberLowRating"] = sum(1 for r in valid_ratings if 1 <= r <= 2)
                    stats["numberMediumRating"] = sum(1 for r in valid_ratings if 3 <= r <= 5)
                    stats["numberHighRating"] = sum(1 for r in valid_ratings if 6 <= r <= 7)
                else:
                    stats["actual_rating"] = []

            stats["numberMessageRead"] = min(number_of_messages_read, stats["numberMessageReceived"])
            stats["readAllMessage"] = (stats["numberMessageRead"] == stats["numberMessageReceived"])
            stats["numberRating"] = number_ratings

        if all_day_ratings:
            stats["medianRating"] = float(np.median(all_day_ratings))
            stats["sdRating"] = float(np.std(all_day_ratings)) if len(all_day_ratings) > 1 else 0.0

            if i > 0:
                prev_stats = day_trajectories[-1]
                if prev_stats["highestRating"] > 0:
                    stats["highestRating"] = max(stats["highestRating"], prev_stats["highestRating"])
                if prev_stats["lowestRating"] > 0:
                    if stats["lowestRating"] == 0:
                        stats["lowestRating"] = prev_stats["lowestRating"]
                    else:
                        stats["lowestRating"] = min(stats["lowestRating"], prev_stats["lowestRating"])

                stats["numberLowRating"] += prev_stats["numberLowRating"]
                stats["numberMediumRating"] += prev_stats["numberMediumRating"]
                stats["numberHighRating"] += prev_stats["numberHighRating"]

        day_trajectories.append(stats)

    for stats in day_trajectories:
        day_trajectory = pd.DataFrame([stats], columns=trajectories_individual.columns)

        # Calculate pH-RL reward and add it to the DataFrame
        day_trajectories_with_rewards = calculate_ph_rl_reward(day_trajectory)

        trajectories_individual = pd.concat([trajectories_individual, day_trajectories_with_rewards], ignore_index=True)

    return trajectories_individual



def init_custom_df():
    """Intialised the dataframe with the correct columns"""
    cols = [
        'day_no', 'day_part_x', 'user_id', 'actual_rating', 'numberRating',
        'highestRating', 'lowestRating', 'medianRating', 'sdRating',
        'numberLowRating', 'numberMediumRating', 'numberHighRating',
        'numberMessageReceived', 'numberMessageRead', 'readAllMessage',
        'reward', 'rl_action', 'motivation_preference'
    ]
    return pd.DataFrame(columns=cols)

def save_trajectories_to_csv_data(trajectories, filename):
    """
    saves the trajctories in a dataframe to a csv file
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    trajectories.to_csv(filename, index=False)

    return trajectories