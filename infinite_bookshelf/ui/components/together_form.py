"""
Together AI form component for Streamlit
"""

import streamlit as st


def render_together_form(on_submit=None, button_disabled=False, button_text="Generate"):
    """
    Render form for Together AI input
    """
    with st.form("together_form"):
        if not st.session_state.get("api_key"):
            together_input_key = st.text_input(
                "Together AI API Key",
                type="password",
                help="Get your API key from https://api.together.xyz",
            )
        else:
            together_input_key = st.session_state.get("api_key")

        topic_text = st.text_area(
            "Book Topic",
            placeholder="Enter the topic or subject of your book",
            help="Describe what you want your book to be about",
        )

        additional_instructions = st.text_area(
            "Additional Instructions (Optional)",
            placeholder="Enter any additional instructions for the AI",
            help="Optional instructions to guide the AI in generating your book",
        )

        models = {
            "Llama 3.3 70B Turbo (Recommended)": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Llama 3.1 8B Turbo (Fast)": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "Llama 3.1 70B Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        }

        selected_model = st.selectbox(
            "Select Model",
            options=list(models.keys()),
            help="Choose the AI model to use for generation"
        )

        submitted = st.form_submit_button(
            button_text,
            disabled=button_disabled,
            on_click=on_submit if on_submit else None,
        )

        return (
            submitted,
            together_input_key,
            topic_text,
            additional_instructions,
            models[selected_model]
        )
