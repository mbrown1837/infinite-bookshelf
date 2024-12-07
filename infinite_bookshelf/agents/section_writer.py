"""
Agent to generate book sections
"""

import together
from ..inference import GenerationStatistics


def generate_book_section(
    prompt: str,
    additional_instructions: str,
    section_title: str,
    section_description: str,
    model: str,
    api_key: str = None,
):
    """
    Returns section content as well as total tokens and total time for generation.
    """
    if api_key:
        together.api_key = api_key

    USER_PROMPT = f"Write a section of a book. It is very important that use the following subject and additional instructions to write the book. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>\n\n<section_title>{section_title}</section_title>\n\n<section_description>{section_description}</section_description>"

    system_prompt = "You are an expert book writer. Write a detailed and engaging section of the book based on the title and description provided. Write in a clear, professional style that maintains reader interest. Include relevant details and examples where appropriate."

    response = together.Complete.create(
        prompt=f"{system_prompt}\n\nHuman: {USER_PROMPT}\n\nAssistant:",
        model=model,
        max_tokens=4000,
        temperature=0.7,
        top_p=1,
        stream=True
    )

    # Initialize variables for collecting streamed response
    full_text = ""
    for chunk in response:
        if hasattr(chunk.output, 'text'):
            full_text += chunk.output.text

    return GenerationStatistics(model_name=model), full_text
