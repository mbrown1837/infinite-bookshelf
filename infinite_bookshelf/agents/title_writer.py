"""
Agent to generate book title
"""

import together
from ..inference import GenerationStatistics


def generate_book_title(prompt: str, additional_instructions: str, model: str, api_key: str = None):
    """
    Returns book title content as well as total tokens and total time for generation.
    """
    if api_key:
        together.api_key = api_key

    USER_PROMPT = f"Create a title for a book. It is very important that use the following subject and additional instructions to write the book. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>"

    system_prompt = "You are an expert at creating book titles. Create a title that is catchy, memorable, and relevant to the subject matter. Respond with only the title, no explanation or additional text."

    response = together.Complete.create(
        prompt=f"{system_prompt}\n\nHuman: {USER_PROMPT}\n\nAssistant:",
        model=model,
        max_tokens=100,
        temperature=0.7,
        top_p=1,
    )

    return GenerationStatistics(model_name=model), response.output.text
