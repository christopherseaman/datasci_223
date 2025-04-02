import json
from openai import OpenAI
client = OpenAI()


system_content = """
You are here to generaate pros and cons for and idea that will submitted by a user.
"""


messages =[{
        "role" : "system",
        "content": system_content,
    },
    {
      "role": "user",
      "content": "Build in ice cream store in the desert."
    } 
  ] 

def reasoning_pro_or_con(reason, pro_con):
    if pro_con == "pro":
        print(f"Pro: {reason}")
    else:
        print(f"Con: {reason}")


tools = [
    {
        "type": "function",
        "function": {
            "name": "reasoning_pro_or_con",
            "description": "Get reasoning for an idea either a pro or a con",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "the reason for the pro or con",
                    },
                    "pro_con": {"type": "string", "enum": ["pro", "con"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=messages,
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  tools=tools,
  tool_choice="auto"
)

response_message = response.choices[0].message

tool_calls = response_message.tool_calls

if tool_calls:
    available_functions = {
        "reasoning_pro_or_con": reasoning_pro_or_con,
    }  
    messages.append(response_message)  


    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(
            reason=function_args.get("reason"),
            pro_con=function_args.get("pro_con"),
        )



