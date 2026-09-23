"""
Phase 1 — the core agent loop (Ollama / local model version).

We switched from Gemini's free-tier API to Ollama running locally, because
the free tier was taking 30-60+ seconds per request (server-side throttling,
confirmed via a raw curl test that bypassed our code entirely). A local
model on this machine's i9-13900H answers in 1-4 seconds, with no network
call, no API key, and no rate limit — the tradeoff is you're bounded by your
own hardware instead of Google's.

The bigger difference from the Gemini version: Gemini's SDK had "automatic
function calling" — hand it a Python function, it inspects the function and
calls it FOR you when needed. Ollama's client doesn't do that. Instead:

  1. You send your message plus a list of tool *schemas* (JSON descriptions
     of what each tool does and what arguments it takes — we write these by
     hand below, instead of Gemini's trick of reading them off docstrings).
  2. The model can either answer directly, OR reply with a "tool_calls"
     field saying "call this function with these arguments."
  3. If it asks for a tool call, WE run the actual Python function
     ourselves, and send the result back as a new message.
  4. We call the model again with that result added to the conversation,
     and it writes the final answer.

This is more steps to write, but it's the standard pattern almost every
tool-calling system uses under the hood (OpenAI, Anthropic, Ollama) —
Gemini's auto-calling was the unusual convenience, not the norm. Now you've
seen both.

Unlike the Gemini version, we also have to manage conversation history
ourselves: `messages` is a plain list we keep appending to. Gemini's
`client.chats.create()` object did this invisibly; here it's explicit.
"""

import ollama

from tools.basic_tools import calculate, get_current_datetime

# qwen2.5:3b-instruct — chosen after llama3.1:8b measured 8-13s per reply on
# this hardware vs qwen2.5's 1-2s, and llama3.1 was also falsely triggering
# tool calls on plain greetings. Must be a model that supports tool
# calling; plain "llama3:latest" does NOT reliably support it.
# Model stays loaded in RAM for ~5 min of inactivity by default, then
# unloads — the first message after a gap pays a ~6s reload cost.
MODEL = "qwen2.5:3b-instruct"

# The actual Python functions the model is allowed to trigger, keyed by the
# same name used in TOOLS_SCHEMA below. When the model asks for a tool call,
# we look up the function here and run it.
AVAILABLE_FUNCTIONS = {
    "get_current_datetime": get_current_datetime,
    "calculate": calculate,
}

# Ollama (like OpenAI, and most tool-calling APIs) wants each tool described
# as a JSON Schema block: a name, a description the model reads to decide
# WHEN to call it, and a list of parameters with their types. This is the
# manual equivalent of what Gemini's SDK generated automatically from our
# docstrings — same information, just written out explicitly.
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": (
                "Returns the current date and time. Use this whenever the "
                "user asks what day, date, or time it is right now."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluates a basic arithmetic expression and returns the "
                "result. Use this for any maths the user asks for, e.g. "
                "'what's 42 * 17' or 'what's 15% of 340'. Supports "
                "+, -, *, /, ** and parentheses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A plain arithmetic expression as a string, e.g. '42 * 17'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

SYSTEM_INSTRUCTION = (
    "You are a helpful personal assistant. You have tools available, but "
    "most messages do not need one. Only call a tool when the user's "
    "message explicitly requires current data (the real date/time) or a "
    "calculation you cannot do reliably in your head. Greetings, opinions, "
    "and general questions get a direct answer with NO tool call. When in "
    "doubt, answer directly. Be concise."
)


def _run_tool_call(tool_call) -> str:
    """Executes one tool call the model asked for, and returns the result
    as a string (tool results always go back to the model as text)."""
    fn_name = tool_call["function"]["name"]
    fn_args = tool_call["function"]["arguments"]  # already a dict
    fn = AVAILABLE_FUNCTIONS.get(fn_name)
    if fn is None:
        return f"Error: unknown tool '{fn_name}'"
    try:
        return str(fn(**fn_args))
    except Exception as exc:  # noqa: BLE001 — hand the model a readable
        # error instead of crashing the whole loop over one bad call.
        return f"Error running '{fn_name}': {exc}"


def main() -> None:
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

    print(f"Jarvis (Phase 1, Ollama/{MODEL}) — type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            print(
                f"Jarvis: (couldn't reach Ollama — is it running? "
                f"Try 'ollama serve' in another terminal. Error: {exc})\n"
            )
            messages.pop()  # don't keep a user turn that never got a reply
            continue

        reply = response["message"]

        if reply.get("tool_calls"):
            messages.append(reply)  # record that the model asked for a tool
            for tool_call in reply["tool_calls"]:
                result = _run_tool_call(tool_call)
                print(f"  [tool call: {tool_call['function']['name']}"
                      f"({tool_call['function']['arguments']}) -> {result}]")
                messages.append({
                    "role": "tool",
                    "content": result,
                    "name": tool_call["function"]["name"],
                })

            final_response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS_SCHEMA)
            final_reply = final_response["message"]
            print(f"Jarvis: {final_reply['content']}\n")
            messages.append(final_reply)
        else:
            print(f"Jarvis: {reply['content']}\n")
            messages.append(reply)


if __name__ == "__main__":
    main()
