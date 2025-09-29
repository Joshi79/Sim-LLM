from src.simulation.utils.create_compact_statistics import  create_compact_statistics_summary
from src.simulation.utils.trend_cycle import  create_trend_analysis_summary

def create_instruction() -> str:
    """Create the improved instructions for the LLM with internal COT reasoning."""
    return (
        """
        <Instructions>
        Before generating any synthetic samples, think through the following reasoning process internally (do not output this reasoning):

        1. Analysis Phase:
           - Review the provided examples to understand the data patterns
           - Identify key statistical properties (rating distributions, temporal patterns, message behaviors)
           - Note the relationship between messages and mood ratings
           - Understand the cumulative of the metrics
           
           Cumlative Structure: 
           - A rating of `[0]` means **no mood was recorded** at that specific time. It does not change any of the cumulative daily totals.
           - A rating from `[1]` to `[7]` is an **actual mood entry**. This entry will update all the cumulative daily features from that point forward.
           
            These features track the total number of ratings submitted over the course of an entire day. They start at 0 at the beginning of each day.
            For each Value: 
            * **`numberRating`**: This is the **total count** of actual ratings (any non-zero value) given *so far* on that day.
            This numberRating is 0 if given [0] for the entire day. Or it is 0 at the day_part_mornig. 
            * **`numberLowRating`**: The total count of ratings between 1-2 given *so far* on that day.So given in Day part Morning Keeps it running for the entire Day
            * **`numberMediumRating`**: The total count of ratings between 3-5 given *so far* on that day.So given in Day part Morning Keeps it running for the entire Day
            * **`numberHighRating`**: The total count of ratings between 6-7 given *so far* on that day. So given in Day part Morning Keeps it running for the entire Day

        2. Planning Phase:
           - Determine appropriate mood rating values (1-7) that fit the observed patterns
           - Plan the progression of cumulative ratings across timepoints
           - Consider realistic message reading behaviors
           - Ensure daily constraints are met (max 7 total ratings)
           - If you want a day with zero ratings, output mood_rating: [0] for the day part x morning (or [0] for all day parts).
           - Think about the highestRating  you want to give for the day. That is the highest Number of the all mood_ratings
           - Think about the lowestRatingfor the day. This is the lowestNumber for the entire day. Thus lowest number of all the mood_ratings
           - Think about how many medium ratings to give (use values 3–5). All medium ratings in mood_ratings count towards it
           - Think about how many low ratings to give (use values 1–2). All medium ratings in mood_ratings count towards it
           - Think about how many high ratings to give (use values 6–7). All medium ratings in mood_ratings count towards it
           - To make these decisions, orient yourself to the proportion counts for all values and the provided statistics.

        3. Validation Phase:
           - Verify that number_of_inputed_ratings increases monotonically within each day
           - Confirm mood_rating arrays contain only new ratings for that timepoint
           - Check that all statistical calculations will be correct
           - Very Important the Number Messages Read cannot be higher than Number Messages Recieved.  
           - Ensure message read flags are logically consistent

        After completing this internal reasoning, generate three synthetic samples that mimic the provided data statistics.

        **Output Requirements:**
        The response must be formatted strictly as a list in JSON format, suitable for direct use in data processing (e.g., conversion to a DataFrame in Python).
        No additional text, reasoning, or numbers should precede the JSON data.

        **Data Structure:**
        You will receive 3 day parts per day (morning, afternoon, evening). For each, generate a JSON object representing only that timepoint.

        **Key Metrics:**
        • `number_of_inputed_ratings`: cumulative across the day (e.g., morning=1, afternoon=3, evening=5). Total ratings submitted up to that point.
        • `mood_rating`: NOT cumulative. Only new ratings submitted during that specific timepoint (delta from last timepoint).
        • Final sum of all new mood ratings across the day cannot exceed 6 total ratings.
        • Number Messages Read cannot exceed the Number Messages Recieved.

        **Core Rules:**

        1. Daily Reset:
           - All mood rating statistics reset at the start of each new day.

        2. Synthetic Data Calculations:
           • highestRating = max of all day's mood ratings
           • lowestRating = min of all day's mood ratings
           • medianRating = median of all day's mood ratings
           • numberHighestRating = count of the highest value in day's mood ratings
           • numberLowestRating = count of the lowest value in day's mood ratings

        3. Mood Rating Constraints:
           - High ratings = 6 or 7
           - Medium ratings = 3, 4, 5
           - Low ratings = 1 or 2

        4. Rating Limits:
           - Maximum total ratings per day = 7

        5. Message Read Logic:
           - `readAllMessage = True` if `numberMessageRead == numberMessageReceived`, else False
           - The numberMessageRead cannot be higher than numberMessageReceived.

        **Simulation Parameters:**
        - 100 patients (IDs 0-99)
        - 30 days per patient (0-29)
        - Generate synthetic dataset that statistically imitates the original dataset as closely as possible

        Generate the synthetic samples now in JSON list format.
        </Instructions>
        """
    )


def create_description_dataset() -> str:
    """Create the description for the LLM."""
    return (
        """
        <Dataset Description>
        Importantly, the dataset holds the statistics of 30 patients which gathered data over a four week period.
        The dataset was created by a study, which measured the adherence of patients to a digital mental health intervention. 
        They applied an RL algorithm which had the goal to maximize the 
        adherence of the patients to the digital mental health intervention. The RL algorithm could send maxium 3 times a day a message to the user. 
        The actions of the RL were ["send no message", "send encouraging message", "send informing message","send affirming message"]. 
        This messages could be sent in the morning, afternoon and evening. 
        Therefore the maxium amount of messages sent per day was 3. 

        numberRating represents the total number of ratings a user has submitted throughout the day, across all available timepoints (e.g., morning, afternoon, evening). 
        Its values range from 0 (no ratings) to 7 (seven ratings were inputted)
        This variable is cumulative over the course of the day. 

        The dataset includes user interaction data, Metadata about their mood ratings including highest and lowest mood ratings, number of highest and lowest mood ratings, median rating
        number of messages read, number of messages recieved. The ratings in the study were inputted on a 7-Point Likert scale, ranging from 1 to 7. 
        High ratings are defines as Ratings of 6 or 7. Medium ratings are defined as Ratings of 3,4 and 5. Lastly, Low ratings are defined as Ratings of 1 or 2. 
        </Dataset Description>
        """

    )

def create_dataset_statistics():
    """Create the dataset statistics for the LLM."""
    # Importantly the path needs to be changed depending on where you have the data stored on your snelius project space
    data_statistics = create_compact_statistics_summary("data/data_splits/original_training_dataset.csv")

    trend_analysis = create_trend_analysis_summary("data/data_splits/original_training_dataset.csv")

    data_statistics += "\n\n" + trend_analysis

    return data_statistics