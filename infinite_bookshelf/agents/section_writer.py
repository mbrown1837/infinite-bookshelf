"""
Agent to generate book sections
"""

from ..inference import GenerationStatistics


def generate_section(
    prompt: str,
    additional_instructions: str,
    model: str,
    together_provider,
):
    """
    Returns book section content as well as total tokens and total time for generation.
    """

    USER_PROMPT = f"Write a detailed and engaging section for a book. Make it informative and well-structured. Use the following prompt and additional instructions to write the section. \n\n<prompt>{prompt}</prompt>\n\n<additional_instructions>{additional_instructions}</additional_instructions>"

    completion = together_provider.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert writer who writes detailed and engaging book sections.",
            },
            {
                "role": "user",
                "content": USER_PROMPT,
            },
        ],
        temperature=0.7,
        max_tokens=8000,
        top_p=1,
        stream=True,
    )

    for chunk in completion:
        if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
        elif hasattr(chunk, 'usage'):
            statistics_to_return = GenerationStatistics(
                input_time=chunk.usage.prompt_time,
                output_time=chunk.usage.completion_time,
                input_tokens=chunk.usage.prompt_tokens,
                output_tokens=chunk.usage.completion_tokens,
                total_time=chunk.usage.total_time,
                model_name=model,
            )
            yield statistics_to_return
