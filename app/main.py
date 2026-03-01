import argparse
import os
import json
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    default="https://openrouter.ai/api/v1"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", required=True)
    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # ✅ conversation history
    messages = [
        {"role": "user", "content": args.p}
    ]

    # ✅ Advertise tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path", "content"],
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path of the file to write to"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    }
                }
            }
        }
    ]

    # ✅ AGENT LOOP
    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=tools
        )

        if not chat.choices:
            raise RuntimeError("no choices in response")

        message = chat.choices[0].message

        assistant_message = {
            "role": "assistant",
            "content": message.content,
        }

        if message.tool_calls:
            assistant_message["tool_calls"] = message.tool_calls

        messages.append(assistant_message)

        # ✅ TOOL EXECUTION
        if message.tool_calls:
            for tool_call in message.tool_calls:
                name = tool_call.function.name.lower()
                args_json = json.loads(tool_call.function.arguments)

                result = ""

                # ---------- READ ----------
                if name == "read":
                    file_path = args_json["file_path"]

                    with open(file_path, "r") as f:
                        result = f.read()

                # ---------- WRITE ----------
                elif name == "write":
                    file_path = args_json["file_path"]
                    content = args_json["content"]

                    with open(file_path, "w") as f:
                        f.write(content)

                    result = "File written successfully"

                # ✅ send tool result back
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            continue

        # ✅ FINAL ANSWER
        print(message.content)
        break


if __name__ == "__main__":
    main()