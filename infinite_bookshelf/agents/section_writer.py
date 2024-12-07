"""
Agent to generate book section content
"""

import together
from ..inference import GenerationStatistics


def generate_section(
    prompt: str, additional_instructions: str, model: str, api_key: str = None
):
    if api_key:
        together.api_key = api_key

    response = together.Complete.create(
        prompt=[
            {
                "role": "system",
                "content": "You are an expert writer. Generate a long, comprehensive, structured chapter for the section provided. If additional instructions are provided, consider them very important. Only output the content.",
            },
            {
                "role": "user",
                "content": f"Generate a long, comprehensive, structured chapter. Use the following section and important instructions:\n\n<section_title>{prompt}</section_title>\n\n<additional_instructions>{additional_instructions}</additional_instructions>",
            },
        ],
        model=model,
        temperature=0.3,
        max_tokens=8000,
        top_p=1,
        stream=True,
    )

    for chunk in response:
        if chunk.output:
            yield chunk.output.text
        if hasattr(chunk, 'usage') and chunk.usage:
            usage = chunk.usage
            statistics_to_return = GenerationStatistics(
                input_time=usage.prompt_time,
                output_time=usage.completion_time,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_time=usage.total_time,
                model_name=model,
            )
            yield statistics_to_return
