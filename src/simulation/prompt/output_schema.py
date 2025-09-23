from typing import List
from pydantic import BaseModel, Field, conint, conlist, RootModel
from langchain.output_parsers import PydanticOutputParser

class UserProfileOutput(BaseModel):
    """Schema for a single JSON object the LLM must return. """

    number_of_inputed_ratings: conint(ge=0, le=7) = Field(
        description="Total number of mood ratings inputted by the user per day part. This is cumulative over the day. The total of all of them cannot exceed 7."
    )
    # Prompt engineering just reduce it 2
    mood_rating: conlist(conint(ge=0, le=7), min_length=0, max_length=7) = Field(
        description="List of mood ratings (1-7). Give out the list [0] if no rating was given."
    )
    number_of_messages_read: conint(ge=0, le=3) = Field(
        description="Total messages read by the user so far (0-3). This cannot be higher than the number of messages recieved."
    )


class TripleUserProfileOutput(RootModel[conlist(UserProfileOutput, min_length=3, max_length=3)]):
    """Wrapper enforcing exactly three UserProfileOutput items."""
    pass


def create_output_parser() -> PydanticOutputParser:
    """Return a LangChain parser for TripleUserProfileOutput."""
    return PydanticOutputParser(pydantic_object=TripleUserProfileOutput)