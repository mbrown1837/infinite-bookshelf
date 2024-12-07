"""
Agent to generate book structure
"""

from ..inference import GenerationStatistics
import together


def generate_book_structure(
    prompt: str,
    additional_instructions: str,
    model: str,
    together_client,
    long: bool = False,
):
    """
    Returns book structure content as well as total tokens and total time for generation.
    """

    if long:
        USER_PROMPT = f"Write a comprehensive structure, omiting introduction and conclusion sections (forward, author's note, summary), for a long (>300 page) book. It is very important that use the following subject and additional instructions to write the book. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>"
    else:
        USER_PROMPT = f"Write a comprehensive structure, omiting introduction and conclusion sections (forward, author's note, summary), for a book. Only provide up to one level of depth for nested sections. Make clear titles and descriptions that have no overlap with other sections. It is very important that use the following subject and additional instructions to write the book. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>"

    together.api_key = together_client
    completion = together.Complete.create(
        prompt=f'''<system>Write in JSON format:

{{"Title of section goes here":"Description of section goes here",
"Title of section goes here":{{"Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here"}}}}</system>
<user>{USER_PROMPT}</user>
<assistant>''',
        model=model,
        temperature=0.3,
        max_tokens=8000,
        top_p=1,
        stop=["</assistant>"],
    )

    # Together AI doesn't provide detailed timing info like Groq, so we'll simplify the statistics
    statistics_to_return = GenerationStatistics(
        input_time=0,
        output_time=0,
        input_tokens=completion.prompt_tokens,
        output_tokens=completion.completion_tokens,
        total_time=0,
        model_name=model,
    )

    return completion.output, statistics_to_return
