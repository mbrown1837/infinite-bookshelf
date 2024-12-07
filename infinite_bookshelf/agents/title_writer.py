"""
Agent to generate book title
"""

import together
from ..inference import GenerationStatistics


def generate_book_title(prompt: str, model: str, api_key: str = None):
    """
    Generate a book title using AI.
    """
    if api_key:
        together.api_key = api_key

    response = together.Complete.create(
        prompt=[
            {
                "role": "system",
                "content": "Generate suitable book titles for the provided topics. There is only one generated book title! Don't give any explanation or add any symbols, just write the title of the book. The requirement for this title is that it must be between 7 and 25 words long, and it must be attractive enough!",
            },
            {
                "role": "user",
                "content": f"Generate a book title for the following topic. There is only one generated book title! Don't give any explanation or add any symbols, just write the title of the book. The requirement for this title is that it must be at least 7 words and 25 words long, and it must be attractive enough:\n\n{prompt}",
            },
        ],
        model=model,
        temperature=0.7,
        max_tokens=100,
        top_p=1,
    )

    usage = response.usage
    statistics_to_return = GenerationStatistics(
        input_time=usage.prompt_time,
        output_time=usage.completion_time,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_time=usage.total_time,
        model_name=model,
    )

    return statistics_to_return, response.output.text.strip()
