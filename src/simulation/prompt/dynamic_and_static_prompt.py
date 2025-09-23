from langchain_core.prompts import PromptTemplate

def static_prompt_template():

    static_prompt = PromptTemplate(
        input_variables=[
            "dataset_description",
            "data_statistics",
            "instructions",
            "format_instructions"
        ],
        template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an intelligent assistant helping with data-driven simulation.
    
    Data Statistics - very important Try to imitate the given Data statistics:
    {data_statistics}
    
    Dataset Description:
    {dataset_description}
    
    Instructions:
    {instructions}
    
    Please adhere strictly to the following format:
    
    {format_instructions}
    – You must return exactly three UserProfileOutput objects.
    – Do not output an array (`[]`). Instead, write each object one after the other.
    - Do not Enclose the entire output in a markdown code-fence
    - Think about the Dataset statistics and the dataset description when generating the output. Do not output your thinking.
    - Very Importantly: First choose a total number of ratings for the day (an integer between 0 and 7). 
    - The number_of_inputed_ratings is the cumulative over the day. So in total the entire day shouldnt be higher than 7. Also be aware of the actual statistics. 
    - First, choose a single integer between 0 and 7 to serve as the day’s total number of ratings. Then output three rows—for morning, afternoon, and evening—where each row’s number_of_inputed_ratings is a non-decreasing cumulative total (morning = morning count, afternoon = morning + afternoon, evening = full daily total), 
    - ensuring the final value never exceeds the chosen daily maximum and should orient itself at the dataset statistics.
    - Again the range of the number_of_inputed_ratings for the entire day has to be between 0 and 7. 
    - The Number Message Read cannot be higher than Number Messages Recieved. 
    - The Number Message Read cannot be higher than Number Messages Recieved.
    - The Number Message Read cannot be higher than Number Messages Recieved.
    - The Rating 0 cannot be togther with an actual rating any number between 1 and 7.
    - The Rating 0 cannot be togther with an actual rating any number between 1 and 7.
    <| eot_id |>
    """
    )
    return static_prompt


def dynamic_prompt():
    return PromptTemplate(
        input_variables=[
            "patient_id","day_no",
            "day_part_1", "messages_received_1", "action_1",
            "day_part_2", "messages_received_2", "action_2",
            "day_part_3", "messages_received_3", "action_3",
            "current_data_statistics"
        ],
        template="""
    <|start_header_id|>user<|end_header_id|>
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>

   Context for today’s three timepoints:
   1. Day part: {day_part_1}
   Action taken: {action_1}
   Messages received: {messages_received_1}

   2. Day part: {day_part_2}
   Action taken: {action_2}
   Messages received: {messages_received_2}

   3. Day part: {day_part_3}
   Action taken: {action_3}
   Messages received: {messages_received_3}
   
   Synthetic Data Statistics: 
   {current_data_statistics}
   
   The Current Data statistics is the statistics you created so far. In the end of the simulation the statistics 
   shoule be imitated as close as possible to the original data statistics. The entire simulation simulated in total of 100 patients and each patient has 30 days. 
   So patients from patient_0 to patient_99 and days from 0 to 30. The current patient and day is {patient_id} and {day_no}.
   number_of_messages_read cannot be higher than number_of_messages_read. 
   Importantly at the end of the simulation the statistics of the synthetic dataset should be imitating the statistics of the original dataset.
   
    Return exactly three JSON objects, one after another, with no additional text:
    {{"number_of_inputed_ratings": ..., "mood_rating": [...], "number_of_messages_read": ...}}
    {{"number_of_inputed_ratings": ..., "mood_rating": [...], "number_of_messages_read": ...}}
    {{"number_of_inputed_ratings": ..., "mood_rating": [...], "number_of_messages_read": ...}}
    BEGIN JSON NOW

    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
""")
