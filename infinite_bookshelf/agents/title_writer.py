"""
Agent to generate book title
"""

from ..inference import GenerationStatistics


def generate_book_title(
    prompt: str,
    additional_instructions: str,
    model: str,
    together_provider,
):
    """
    Returns book title content as well as total tokens and total time for generation.
    """

    USER_PROMPT = f"Create a compelling and concise book title. Use the following subject and additional instructions to create the title. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>"

    completion = together_provider.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at creating compelling book titles. Create a title that is engaging and accurately represents the book's content.",
            },
            {
                "role": "user",
                "content": USER_PROMPT,
            },
        ],
        temperature=0.7,
        max_tokens=100,
        top_p=1,
        stream=False,
    )

    usage = completion.usage
    statistics_to_return = GenerationStatistics(
        input_time=usage.prompt_time,
        output_time=usage.completion_time,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_time=usage.total_time,
        model_name=model,
    )

    return statistics_to_return, completion.choices[0].message.content
