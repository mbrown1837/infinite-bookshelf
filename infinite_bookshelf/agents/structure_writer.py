"""
Agent to generate book structure
"""

from ..inference import GenerationStatistics


def generate_book_structure(
    prompt: str,
    additional_instructions: str,
    model: str,
    together_provider,
    long: bool = False,
):
    """
    Returns book structure content as well as total tokens and total time for generation.
    """

    if long:
        USER_PROMPT = f"Write a comprehensive structure, omiting introduction and conclusion sections (forward, author's note, summary), for a long (>300 page) book. It is very important that use the following subject and additional instructions to write the book. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>"
    else:
        USER_PROMPT = f"Write a comprehensive structure, omiting introduction and conclusion sections (forward, author's note, summary), for a book. Only provide up to one level of depth for nested sections. Make clear titles and descriptions that have no overlap with other sections. It is very important that use the following subject and additional instructions to write the book. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>"

    # Using the new chat completions API
    response = together_provider.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": 'Write in JSON format:\n\n{"Title of section goes here":"Description of section goes here",\n"Title of section goes here":{"Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here"}}'
            },
            {
                "role": "user",
                "content": USER_PROMPT
            }
        ],
        max_tokens=8000,
        temperature=0.3,
        top_p=1
    )

    # Get the response text from the new API structure
    output_text = response.choices[0].message.content

    # Create statistics object
    statistics_to_return = GenerationStatistics(
        input_time=0,
        output_time=0,
        input_tokens=response.usage.prompt_tokens if hasattr(response, 'usage') else len(USER_PROMPT.split()),
        output_tokens=response.usage.completion_tokens if hasattr(response, 'usage') else len(output_text.split()),
        total_time=0,
        model_name=model,
    )

    return statistics_to_return, output_text
