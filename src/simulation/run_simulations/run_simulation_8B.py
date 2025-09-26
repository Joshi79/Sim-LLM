import os
import logging
import sys
import datetime
import random
import time
import pandas as pd
import numpy as np


from src.simulation.set_up_llm.llama_3_1_8B_instruct import CachedLLMPipeline
from src.simulation.utils.synthetic_data_checks import simulation_statistics
from src.simulation.utils.helper_functions import calculate_rest_columns_8B, clean_llm_output_8B, validate_llm_output, init_custom_df, save_trajectories_to_csv_data

from src.simulation.prompt.output_schema import create_output_parser
from src.simulation.prompt.prompt_parts import create_description_dataset, create_dataset_statistics, create_instruction
from src.simulation.prompt.dynamic_and_static_prompt import dynamic_prompt, static_prompt_template



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


# ---------------------------------------------------------------------------#
# Set global constants and configurations
# ---------------------------------------------------------------------------#

ACTIONS = [0, 1, 2, 3]
DAY_PARTS = [0,1,2]
ACTIONS_STRINGS = [
    "send no message",
    "send encouraging message",
    "send informing message",
    "send affirming message",
]
INSTRUCTIONS = create_instruction()
DATASET_DESCRIPTION = create_description_dataset()
DATASET_STATISTICS = create_dataset_statistics()
PARSER = create_output_parser()


static_prompt = static_prompt_template()
final_static_prompt = static_prompt.format(
    dataset_description=DATASET_DESCRIPTION,
    data_statistics=DATASET_STATISTICS,
    instructions=INSTRUCTIONS,
    format_instructions=PARSER.get_format_instructions())



def main(simulated_patients):
    os.makedirs("metrics", exist_ok=True)

    run_id = timestamp  # reuse your global timestamp
    run_start_dt = datetime.datetime.now().isoformat()
    t_run = time.perf_counter()

    metrics_rows = []  # we'll turn this into one DataFrame at the end
    call_no = 0

    llm = CachedLLMPipeline(static_prompt=final_static_prompt, max_new_tokens=250)
    simulations_df = init_custom_df()
    dynamic_prompt_template_day = dynamic_prompt()

    for patient_no in range(simulated_patients):
        user_id = f"patient_{patient_no}"
        trajectories_individual = init_custom_df()
        logger.info(f"\n\n Starting simulation for patient {patient_no + 1} of {simulated_patients}.\n")

        for day_id in range(30):
            action_list = []
            message_type_list = []
            messages_received = 0
            messages_recieved_list = []

            for i in range(3):
                action = random.choice(ACTIONS)
                action_list.append(action)
                message_type = ACTIONS_STRINGS[action]
                message_type_list.append(message_type)
                if message_type != "send no message":
                    messages_received += 1
                messages_recieved_list.append(messages_received)

            logger.info(f"Actions for day {day_id + 1}: {action_list}")
            logger.info(f"\n\nSimulating day {day_id + 1} for {user_id}.\n")

            simulation_statistic = simulation_statistics(simulations_df)
            final_dynamic_prompt = dynamic_prompt_template_day.format(
                patient_id=user_id, day_no=day_id,
                day_part_1="morning", messages_received_1=messages_recieved_list[0], action_1=action_list[0],
                day_part_2="afternoon", messages_received_2=messages_recieved_list[1], action_2=action_list[1],
                day_part_3="evening", messages_received_3=messages_recieved_list[2], action_3=action_list[2],
                current_data_statistics=simulation_statistic
            )

            # --- measure LLM latency ---
            t0 = time.perf_counter()
            try:
                raw_output, full_prompt_used, real_output = llm.invoke(final_dynamic_prompt)
                latency = time.perf_counter() - t0
                logger.info(f"LLM call took {latency:.3f} seconds.")
                logger.info(f"Raw output: {raw_output}")

                output_list = clean_llm_output_8B(raw_output)


                output_list = validate_llm_output(output_list, expected_length=3)

                metrics_rows.append({
                    "run_id": run_id,
                    "call_no": call_no,
                    "patient_no": patient_no,
                    "user_id": user_id,
                    "day_no": day_id,
                    "llm_latency_sec": latency,
                    "success": True,
                    "error_message": None
                })

            except Exception as e:
                logger.error(f"Error during LLM call for patient {patient_no}, day {day_id}: {str(e)}")
                # Create empty output on error
                output_list = [{}, {}, {}]
                metrics_rows.append({
                    "run_id": run_id,
                    "call_no": call_no,
                    "patient_no": patient_no,
                    "user_id": user_id,
                    "day_no": day_id,
                    "llm_latency_sec": None,
                    "success": False,
                    "error_message": str(e)
                })

            call_no += 1

            # Use the robust function
            try:
                day_trajectories = calculate_rest_columns_8B(
                    simulated_day=output_list,
                    day_no=day_id,
                    messages_received_list=messages_recieved_list,
                    user_id=user_id,
                    action_list=action_list,
                    day_part_list=DAY_PARTS,
                    motivational_preference="None",
                    trajectories_individual=trajectories_individual
                )
                simulations_df = pd.concat([simulations_df, day_trajectories], ignore_index=True)

            except Exception as e:
                logger.error(f"Error processing day data for patient {patient_no}, day {day_id}: {str(e)}")

                # That is a fall-back
                for i in range(3):
                    error_entry = {
                        "user_id": user_id,
                        "day_part_x": DAY_PARTS[i] if i < len(DAY_PARTS) else i,
                        "day_no": day_id,
                        "all_day_ratings": [],
                        "actual_rating": np.nan,
                        "highestRating": np.nan,
                        "lowestRating": np.nan,
                        "medianRating": np.nan,
                        "sdRating": np.nan,
                        "numberLowRating": 0,
                        "numberMediumRating": 0,
                        "numberHighRating": 0,
                        "numberRating": 0,
                        "numberMessageRead": 0,
                        "numberMessageReceived": messages_recieved_list[i] if i < len(messages_recieved_list) else 0,
                        "readAllMessage": False,
                        "reward": 0,
                        "rl_action": action_list[i] if i < len(action_list) else np.nan,
                        "motivation_preference": "None"
                    }
                    error_df = pd.DataFrame([error_entry], columns=simulations_df.columns)
                    simulations_df = pd.concat([simulations_df, error_df], ignore_index=True)

    # Save your simulation data as before
    file_path_csv = f"output/run_with_data_checks_30_patients_llama_8B_Instruct_final{timestamp}.csv"
    save_trajectories_to_csv_data(simulations_df, filename=file_path_csv)

    overall_wall_sec = time.perf_counter() - t_run
    end_dt = datetime.datetime.now().isoformat()

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        metrics_df = pd.DataFrame([{
            "run_id": run_id, "call_no": None, "patient_no": None, "user_id": None,
            "day_no": None, "llm_latency_sec": None, "success": None, "error_message": None
        }])

    metrics_df["start_dt"] = run_start_dt
    metrics_df["end_dt"] = end_dt
    metrics_df["wall_time_sec"] = overall_wall_sec
    metrics_df["total_calls"] = len(metrics_df)
    metrics_df["simulated_patients"] = simulated_patients

    out_path = f"metrics/llm_metrics_{run_id}.csv"
    metrics_df.to_csv(out_path, index=False)
    logger.info("Saved combined metrics to %s", out_path)

    if "success" in metrics_df.columns:
        success_rate = metrics_df["success"].sum() / len(metrics_df) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
        if metrics_df["error_message"].notna().any():
            logger.info("Unique errors encountered:")
            for error in metrics_df["error_message"].dropna().unique():
                logger.info(f"  - {error}")


if __name__ == "__main__":
    simulated_patients = 5
    main(simulated_patients)
    logger.info("Simulation completed successfully.")