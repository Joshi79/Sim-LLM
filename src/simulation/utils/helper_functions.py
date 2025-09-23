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

                # make sure when during the day it happens that  - the lowerst rating wont be 0
                non_zero_ratings = [r for r in ratings if r > 0]
                current_lowest_number = int(min(non_zero_ratings)) if non_zero_ratings else np.inf

                if current_highest_number > stats["highestRating"]:
                    stats["highestRating"] = current_highest_number

                    # make sure that no rating given doesnt change the lowest rating
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
                trajectories_individual = pd.concat([trajectories_individual, day_trajectory], ignore_index=True)

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


def clean_llm_output_8B(json_string):
    """Cleaning the LLM output to be a valid Json for the LLM 8B model"""

    clean_string = re.sub(r'```json\s*|\s*```+', '', json_string)

    # add [] around the string
    if not clean_string.strip().startswith('['):
        clean_string = '[' + clean_string.strip()

    if not clean_string.strip().endswith(']'):
        clean_string = clean_string.strip() + ']'

    clean_string = re.sub(r'\]\s*\[', ',', clean_string)

    try:

        objects = json.loads(clean_string)

        normalized_objects = []
        for obj in objects:
            if isinstance(obj, dict):
                normalized_objects.append(obj)
            elif isinstance(obj, (int, float)):

                normalized_objects.append({
                    "mood_rating": [int(obj)] if obj > 0 else [0],
                    "number_of_messages_read": 0,
                    "number_of_inputed_ratings": 1 if obj > 0 else 0
                })
            elif isinstance(obj, list):

                normalized_objects.append({
                    "mood_rating": [int(x) for x in obj if isinstance(x, (int, float))],
                    "number_of_messages_read": 0,
                    "number_of_inputed_ratings": len([x for x in obj if isinstance(x, (int, float))])
                })
        return normalized_objects
    except json.JSONDecodeError:

        objects = []
        decoder = json.JSONDecoder()
        pos = 0

        while pos < len(clean_string):
            try:
                obj, pos = decoder.raw_decode(clean_string, pos)
                if isinstance(obj, dict):
                    objects.append(obj)
                elif isinstance(obj, (int, float)):
                    objects.append({
                        "mood_rating": [int(obj)] if obj > 0 else [0],
                        "number_of_messages_read": 0,
                        "number_of_inputed_ratings": 1 if obj > 0 else 0
                    })
                elif isinstance(obj, list):
                    if all(isinstance(item, dict) for item in obj):
                        objects.extend(obj)
                    else:
                        objects.append({
                            "mood_rating": [int(x) for x in obj if isinstance(x, (int, float))],
                            "number_of_messages_read": 0,
                            "number_of_inputed_ratings": len([x for x in obj if isinstance(x, (int, float))])
                        })
            except json.JSONDecodeError:
                pos += 1

        return objects

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
