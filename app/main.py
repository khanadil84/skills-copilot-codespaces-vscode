import argparse
import os
import json
import subprocess

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

    # ✅ conversation memory
    messages = [
        {"role": "user", "content": args.p}
    ]

    # ✅ Tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"}
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
                        "file_path": {"type": "string"},
                        "content": {"type": "string"}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute"
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
            raise RuntimeError("no choices")

        message = chat.choices[0].message

        assistant_message = {
            "role": "assistant",
            "content": message.content
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

                # -------- READ --------
                if name == "read":
                    with open(args_json["file_path"], "r") as f:
                        result = f.read()

                # -------- WRITE --------
                elif name == "write":
                    with open(args_json["file_path"], "w") as f:
                        f.write(args_json["content"])
                    result = "File written successfully"

                # -------- BASH --------
                elif name == "bash":
                    command = args_json["command"]

                    completed = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True
                    )

                    result = completed.stdout + completed.stderr

                # ✅ return tool result to model
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            continue

        # ✅ FINAL RESPONSE
        print(message.content)
        break


if __name__ == "__main__":
    main()