"""
LLM with tool calling via Groq. Answers general questions directly, or
calls get_order_status when the user asks about an order -- including
Hinglish phrasing ("mera order kab aayega").

Requires GROQ_API_KEY in the environment (see .env.example).
"""
import json
import os

from groq import Groq

from agent.tools import AVAILABLE_FUNCTIONS, TOOLS_SCHEMA

MODEL_NAME = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant for an e-commerce app. "
    "Answer general questions directly. If the user asks about an order "
    "(status, delivery date, tracking), use the get_order_status tool -- "
    "don't guess. The user may write in English, Hindi, or Hinglish "
    "(code-switched Hindi/English); reply naturally in whichever style "
    "they used. Keep replies short and conversational, suitable for being "
    "read aloud by a voice assistant."
)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _decide_and_maybe_call_tools(user_text: str):
    """Runs the tool-decision completion (never streamed -- a tool name and
    its JSON arguments only become valid once complete, so there's nothing
    useful to stream mid-decision). Returns (direct_reply, messages): if no
    tool was needed, direct_reply is the final answer and messages is None;
    if a tool was called, direct_reply is None and messages is the message
    list (with the assistant's tool-call turn and the tool's result
    appended) ready for a final completion call."""
    client = _get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
    )
    message = response.choices[0].message

    if not message.tool_calls:
        return message.content, None

    messages.append(message)
    for tool_call in message.tool_calls:
        function = AVAILABLE_FUNCTIONS[tool_call.function.name]
        args = json.loads(tool_call.function.arguments)
        result = function(**args)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })
    return None, messages


def respond(user_text: str) -> str:
    direct_reply, messages = _decide_and_maybe_call_tools(user_text)
    if direct_reply is not None:
        return direct_reply

    client = _get_client()
    final_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )
    return final_response.choices[0].message.content


def respond_stream(user_text: str):
    """Same behavior as respond(), but the final natural-language reply is
    yielded as text chunks (generator) instead of returned all at once.
    The tool-decision call is never streamed (see
    _decide_and_maybe_call_tools) -- if no tool was needed, this yields the
    one already-complete direct reply as a single chunk rather than a true
    token stream, since that reply came from a non-streamed call by
    construction. Only tool-augmented replies get real token-by-token
    streaming under this design; acceptable since Phase 6/8 measured the
    tool-decision call itself as already fast (~0.5s).

    Only yields delta.content. MODEL_NAME (openai/gpt-oss-120b) is a
    reasoning model that also streams its internal chain-of-thought via a
    separate delta.reasoning field (channel='analysis', verified by
    inspecting a real streamed response) -- that is never yielded here, to
    avoid TTS ever speaking the model's internal reasoning aloud."""
    direct_reply, messages = _decide_and_maybe_call_tools(user_text)
    if direct_reply is not None:
        yield direct_reply
        return

    client = _get_client()
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content
