"""
Agent to generate book structure
"""

import together
from ..inference import GenerationStatistics


def generate_book_structure(
    prompt: str,
    additional_instructions: str,
    model: str,
    long: bool = False,
):
    """
    Returns book structure content as well as total tokens and total time for generation.
    """

    if long:
        USER_PROMPT = f"Write a comprehensive structure, omiting introduction and conclusion sections (forward, author's note, summary), for a long (>300 page) book. It is very important that use the following subject and additional instructions to write the book. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>"
    else:
        USER_PROMPT = f"Write a comprehensive structure, omiting introduction and conclusion sections (forward, author's note, summary), for a book. Only provide up to one level of depth for nested sections. Make clear titles and descriptions that have no overlap with other sections. It is very important that use the following subject and additional instructions to write the book. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>"

    SYSTEM_PROMPT = 'Write in JSON format:\n\n{"Title of section goes here":"Description of section goes here",\n"Title of section goes here":{"Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here"}}'

    prompt = f"<s>[INST] {SYSTEM_PROMPT} [/INST] {USER_PROMPT} [/INST]"

    completion = together.Complete.create(
        prompt=prompt,
        model=model,
        max_tokens=8000,
        temperature=0.3,
        top_p=1,
        stop=["</s>"],
        response_format={"type": "json_object"}
    )

    response = completion.output.text
    usage = completion.usage

    statistics_to_return = GenerationStatistics(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        total_time=usage["time"],
        model_name=model,
    )

    return statistics_to_return, response
