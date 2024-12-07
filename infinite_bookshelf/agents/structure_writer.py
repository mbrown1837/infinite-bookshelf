"""
Agent to generate book structure
"""

import together
from ..inference import GenerationStatistics


def generate_book_structure(
    prompt: str,
    additional_instructions: str,
    model: str,
    api_key: str = None,
    long: bool = False,
):
    """
    Returns book structure content as well as total tokens and total time for generation.
    """
    if api_key:
        together.api_key = api_key

    if long:
        USER_PROMPT = f"Write a comprehensive structure, omiting introduction and conclusion sections (forward, author's note, summary), for a long (>300 page) book. It is very important that use the following subject and additional instructions to write the book. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>"
    else:
        USER_PROMPT = f"Write a comprehensive structure, omiting introduction and conclusion sections (forward, author's note, summary), for a book. Only provide up to one level of depth for nested sections. Make clear titles and descriptions that have no overlap with other sections. It is very important that use the following subject and additional instructions to write the book. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>"

    system_prompt = 'Write in JSON format:\n\n{"Title of section goes here":"Description of section goes here",\n"Title of section goes here":{"Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here"}}'

    response = together.Complete.create(
        prompt=f"{system_prompt}\n\nHuman: {USER_PROMPT}\n\nAssistant:",
        model=model,
        max_tokens=8000,
        temperature=0.3,
        top_p=1,
    )

    return GenerationStatistics(model_name=model), response.output.text
