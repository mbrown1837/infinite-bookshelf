"""
Agent to generate book title
"""

from ..inference import GenerationStatistics


def generate_book_title(prompt: str, model: str, together_provider):
    """
    Generate a book title using AI.
    """
    response = together_provider.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Generate suitable book titles for the provided topics. There is only one generated book title! Don't give any explanation or add any symbols, just write the title of the book. The requirement for this title is that it must be between 7 and 25 words long, and it must be attractive enough!",
            },
            {
                "role": "user",
                "content": f"Generate a book title for the following topic. There is only one generated book title! Don't give any explanation or add any symbols, just write the title of the book. The requirement for this title is that it must be at least 7 words and 25 words long, and it must be attractive enough:\n\n{prompt}",
            },
        ],
        temperature=0.7,
        max_tokens=100,
        top_p=1
    )

    # Create statistics object
    statistics_to_return = GenerationStatistics(
        input_time=0,
        output_time=0,
        input_tokens=response.usage.prompt_tokens if hasattr(response, 'usage') else len(prompt.split()),
        output_tokens=response.usage.completion_tokens if hasattr(response, 'usage') else 0,
        total_time=0,
        model_name=model,
    )

    return statistics_to_return, response.choices[0].message.content.strip()
