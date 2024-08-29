from typing import Any, List, Tuple

from openai import OpenAI

client = OpenAI()

DEFAULT_SYSTEM_PROMPT = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."

GPT_4o_MINI = 'gpt-4o-mini'
GPT_4o_AUG = 'gpt-4o-2024-08-06'

TEMPERATURE = 0.0
DEFAULT_COMPLETE = GPT_4o_MINI
DEFAULT_RESPONSE_FORMAT = { "type": "json_object" }

def completion(system_prompt: str, user_prompt: str, model: str = DEFAULT_COMPLETE, response_format: dict = DEFAULT_RESPONSE_FORMAT) -> Tuple[str, Any]:
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            response_format=response_format,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as e:
        # Handle the exception here
        print(f"An error occurred: {str(e)}")
        response = None

    if response is not None:
        return response.choices[0].message.content, response.usage
    else:
        return None, None # type: ignore
    
def get_cost(usage: dict) -> str:
    if DEFAULT_COMPLETE == GPT_4o_MINI:
        token_input_cost = .15 / 1000000 # TODO: legacy; update to Usage
        token_output_cost = .6 / 1000000 # TODO: legacy; update to Usage
    else:
        return '-1.0'
    
    input_cost = usage['prompt_tokens'] * token_input_cost
    output_cost = usage['completion_tokens'] * token_output_cost
    return f"{input_cost + output_cost:.3f}"