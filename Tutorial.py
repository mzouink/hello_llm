# %% [markdown]
# # Open AI API Tutorial
# First you need to create a tocken key from openai.com
# (don't forget to put some money in your account)
# ```python
# GPT_API_KEY = "dummy key"
# ```
# ### install dependencies for today's tutorial
# 1- OpenAI - pydantic- autogen
# ```bash
# pip install openai
# pip install pydantic==2.7.1
# pip install pyautogen==0.2.25
# pip install chess
# ```
# %%
# GPT_API_KEY = "dummy key"
from config import GPT_API_KEY
import os

os.environ["OPENAI_API_KEY"] = GPT_API_KEY
# %%
from openai import OpenAI

client = OpenAI(api_key=GPT_API_KEY)
# %% [markdown]
# # Part 1: Create a chat - LLM
# basic call to the openai API
# %%
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "What do you know about Janelia ? ",
        },
    ],
)
# %%
completion
# %%
completion.model_dump()
# %%
completion.choices[0].message.content
# %%
# %% [markdown]
# ## Temperature Parameter
# The `temperature` parameter influences the creativity or randomness of ChatGPT’s responses.
# A higher temperature value (e.g., 1) makes the output more random, while a lower value (e.g., 0.2) makes it more focused and deterministic.
# In the following example, we set the temperature to 1 to increase the randomness of the response.
# %%
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "What does Janelia do ? ",
        },
    ],
    temperature=1,
    max_tokens=10,
)
completion.choices[0].message.content

# %% [markdown]
# ## Image Analysis Task
# In this task, we will provide an image to the ChatGPT model and ask it to describe what it sees in the image.
# ![Janelia Image](https://www.janelia.org/sites/default/files/styles/epsa_625x415/public/Janelia%20Archives/Nessie-HEader-Image.jpg)
# %%
# input image
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What you see in this image ?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.janelia.org/sites/default/files/styles/epsa_625x415/public/Janelia%20Archives/Nessie-HEader-Image.jpg"
                    },
                },
            ],
        },
    ],
)
completion.choices[0].message.content
# %% [markdown]
# ## Generate Image using DALL-E
# %%
# generate image
response = client.images.generate(
    model="dall-e-3",
    prompt="a painting image of Janelia Research Campus",
    size="1024x1024",
    quality="standard",
    n=1,
)

from IPython.display import Image

Image(url=response.data[0].url)
# %%
# %% [markdown]
# # Part 2: Create an Agent
# ![Agent Design](https://langfuse.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fagent-design.b449b28a.png&w=750&q=75)
# ![Agent Design](/Users/zouinkhim/Desktop/learn/multi_agents/agents.png)
# %%
# # Part 2: Create an Agent
JANELIA_ASSISTANT_PROMPT = "You are a Janelia assistant bot. Your mission is to return the department name of the user's query. \n\
    The user will ask you about the department name of a specific person. You should return the department name of that person."
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": JANELIA_ASSISTANT_PROMPT,
        },
        {
            "role": "user",
            "content": "what is the department of Marwan Zouinkhi ?",
        },
    ],
)
completion.choices[0].message.content
# %% [markdown]
# # Structure output using Pydantic
# still beta: use ```client.beta.chat.completions.parse``` instead of ```client.chat.completions.create```
# %%

from pydantic import BaseModel, Field


class DepartmentSchema(BaseModel):
    department: str = Field(..., description="The department name of the person")


messages = [
    {
        "role": "system",
        "content": JANELIA_ASSISTANT_PROMPT,
    },
    {
        "role": "user",
        "content": "what is the department of Marwan Zouinkhi ?",
    },
]
completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    response_format=DepartmentSchema,
)
completion.choices[0].message.parsed
# %% [markdown]
# limit the output using Enum
# %%
from enum import Enum


class JaneliaDepartment(Enum):
    Scicomp = "Scientific Computing"
    jet = "Janelia Experimental Technology"
    cellmap = "Cell Map"
    flylight = "Fly Light"
    lab = "Lab"


class DepartmentSchemaV2(BaseModel):
    department: JaneliaDepartment = Field(
        ..., description="The department name of the person"
    )


completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    response_format=DepartmentSchemaV2,
)

completion.choices[0].message.parsed
# %% [markdown]
# # Function calling
# Give a list of tools to the model and let it call the function
#
# PS: the model will not call the function, you need to implement the routing function
#
# and the model will return ```tool_calls``` with the arguments to pass to the function
# %%
# Function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_department",
            "description": "Get Janelia department for a given name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the employee.",
                    }
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        "strict": True,
    }
]
completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    response_format=DepartmentSchemaV2,
    tools=tools,
)
completion.model_dump()

# %%
completion.choices[0].message.tool_calls[0].function

# %%
import json

args = completion.choices[0].message.tool_calls[0].function.arguments
args = json.loads(args)
args


# %%
def get_department(name: str) -> JaneliaDepartment:
    return JaneliaDepartment.Scicomp


get_department(**args)


# %%
#  you need to implement routing function
def execute_tool(tool_call):
    if tool_call.function.name == "get_department":
        return get_department(**json.loads(tool_call.function.arguments))
    return None


execute_tool(completion.choices[0].message.tool_calls[0])
# %%
# %% [markdown]
# # Agents Framework
# You can do everything with the API, but you can also use the Agent Framework for easier implementation of complex scenarios.
# %% [markdown]
# Available frameworks are a a lot but i only tried:
# - **Langchain**:
#
# ![Langchain](https://miro.medium.com/v2/resize:fit:622/1*MVJZLfszGGNiJ-UFK4U31A.png)
#
# Less code to write but more complex to understand
# - **Autogen**:
#
# ![autogen](https://opengraph.githubassets.com/58f3a6e72d42f10caab43913550255b3529597c447cfb0a8960bf906fbd2da99/microsoft/autogen)
#
# More code to write but more simple to understand, most mature framework
# - **PydanticAI**:
#
# ![PydanticAI](https://ai.pydantic.dev/img/pydantic-ai-dark.svg#only-dark)
#
# New project with a lot of potential, everything is a BaseModel functions/tools/input/output but still in beta
#
# ## coupled with logfire for logging
# ![logfire](https://pydantic.dev/assets/logfire/tabbed-code-blocks/database/right-code-block.png)
# - **OpenAI Swarm**:
#
# ![Swarm](https://github.com/openai/swarm/raw/main/assets/logo.png)
#
# Still in experimental phase but very promising
#

# %%
# %%
from config import GPT_API_KEY

llm_config = {
    "config_list": [
        {"model": "gpt-4", "api_key": GPT_API_KEY},
    ]
}
# %% [markdown]
# # Types Agents
# ![Types Agents](/Users/zouinkhim/Desktop/learn/multi_agents/types_agents.png)
# - **ConversableAgent**: Parent class, can be used to create any type of agent.
# - **GroupChatManager**: No Human input, can manage multiple agents in a group chat.
# - **UserProxyAgent**: Interface with Human as input or output. Can be used to create a chatbot.
# - **AssistantAgent**: Expert system, communicate with other agents to solve a specific problem.
# PS: User proxy can keep iterating with the user with questions until all the needed information are gathered.
# %%
from autogen import ConversableAgent

agent = ConversableAgent(
    name="chatbot",
    llm_config=llm_config,
    human_input_mode="NEVER",  # NEVER, TERMINATE, ALWAYS, never prompt for human input
)
# %%
reply = agent.generate_reply(
    messages=[{"content": "Tell me a joke about Janelia.", "role": "user"}]
)
reply

# %%
# human_input_mode (str): whether to ask for human inputs every time a message is received.
#                 Possible values are "ALWAYS", "TERMINATE", "NEVER".
#                 (1) When "ALWAYS", the agent prompts for human input every time a message is received.
#                     Under this mode, the conversation stops when the human input is "exit",
#                     or when is_termination_msg is True and there is no human input.
#                 (2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or
#                     the number of auto reply reaches the max_consecutive_auto_reply.
#                 (3) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops
#                     when the number of auto reply reaches the max_consecutive_auto_reply or when is_termination_msg is True.

# %%

# Agents conversation
marwan = ConversableAgent(
    name="Marwan",
    system_message="Your name is Marwan, You are Software engineer at Janelian and you are in a stand-up comedian and you are performing in a show about Janelia. Your goal is to keep the jokes rolling. by making fun of the other person's jokes. just a short joke.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)
joe = ConversableAgent(
    name="Joe",
    system_message="Your name is Joe and you are an engineer at Google and stand-up comedian. Your goal is to keep the jokes rolling. by making fun Janelia and of the other person's jokes. just a short joke. Start the next joke from the punchline of the previous joke.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)
chat_result = joe.initiate_chat(
    recipient=marwan,
    message="I'm Joe. Marwan, let's keep the jokes rolling.",
    max_turns=0,
)

# %% [markdown]
# # Discription:
#  description is field of ConversableAgent (and all subclasses), used by GroupChat / other agents when choosing which agents should speak next.
# # is_termination_msg:
# is_termination_msg is a function that takes a message and returns True if the message is a termination message, and False otherwise.
# %%
# added is_termination_msg to end the conversation, agent decided to end the conversation (can use a function for it)
# and description for the agent, what other agent should know about this agent
marwan = ConversableAgent(
    name="Marwan",
    system_message="Your name is Marwan, You are Software engineer at Janelian and you are in a stand-up comedian and you are performing in a show about Janelia. Your goal is to keep the jokes rolling. by making fun of the other person's jokes. just a short joke."
    "When you're ready to end the conversation, say 'I gotta go'.",
    llm_config=llm_config,
    description="Marwan is a software engineer from Tunisia, He couldn't finish his PhD and he is now working in Janelia and CellMap Team",
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "I gotta go" in msg["content"],
)

joe = ConversableAgent(
    name="joe",
    system_message="Your name is Joe and you are an engineer at Google and stand-up comedian. Your goal is to keep the jokes rolling. by making fun Janelia and of the other person's jokes. just a short joke. Start the next joke from the punchline of the previous joke."
    "When you're ready to end the conversation, say 'I gotta go'.",
    llm_config=llm_config,
    description="Joe is a software enigneer from USA, He is working in Google and he is a stand-up comedian",
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "I gotta go" in msg["content"]
    or "Goodbye" in msg["content"],
)
chat_result = joe.initiate_chat(
    recipient=marwan, message="I'm Joe. Marwan, let's keep the jokes rolling."
)


# %%
# memory

marwan.send(message="What's last joke we talked about?", recipient=joe)


# %% [markdown]
# # Chess Game
# %%
import chess
import chess.svg
from typing_extensions import Annotated

# %%
board = chess.Board()
display(
    chess.svg.board(
        board,
        size=200,
    )
)


# %%
def get_legal_moves() -> Annotated[str, "A list of legal moves in UCI format"]:
    return "Possible moves are: " + ",".join([str(move) for move in board.legal_moves])


get_legal_moves()

# %%
made_move = False


def make_move(
    move: Annotated[str, "A move in UCI format."]
) -> Annotated[str, "Result of the move."]:
    move = chess.Move.from_uci(move)
    board.push_uci(str(move))
    global made_move
    made_move = True

    # Display the board.
    display(
        chess.svg.board(
            board,
            arrows=[(move.from_square, move.to_square)],
            fill={move.from_square: "gray"},
            size=200,
        )
    )
    return f"Moved {move}"


def check_made_move(move):
    global made_move
    if made_move:
        made_move = False
        return True
    else:
        return False


# %% [markdown]
# Create agents:
# ![Chess Agents](/Users/zouinkhim/Desktop/learn/multi_agents/Chess_Agents.png)


# %%
from autogen import ConversableAgent

player_white = ConversableAgent(
    name="Player White",
    system_message="You are a chess player and you play as white. "
    "First call get_legal_moves(), to get a list of legal moves. "
    "Then call make_move(move) to make a move.",
    llm_config=llm_config,
)
player_black = ConversableAgent(
    name="Player Black",
    system_message="You are a chess player and you play as black. "
    "First call get_legal_moves(), to get a list of legal moves. "
    "Then call make_move(move) to make a move.",
    llm_config=llm_config,
)

# board proxy agent will keep track of the board state and make moves on the board, keep asking for moves until a move is made
board_proxy = ConversableAgent(
    name="Board Proxy",
    llm_config=False,
    is_termination_msg=check_made_move,
    default_auto_reply="Please make a move.",
    human_input_mode="NEVER",
)

# %%
# Register the tools, each agent will have access to the tools

from autogen import register_function

for caller in [player_white, player_black]:
    register_function(
        get_legal_moves,
        caller=caller,
        executor=board_proxy,
        name="get_legal_moves",
        description="Get legal moves.",
    )

    register_function(
        make_move,
        caller=caller,
        executor=board_proxy,
        name="make_move",
        description="Call this tool to make a move.",
    )


# %%
player_black.llm_config["tools"]
# %%
player_white.llm_config["tools"]


# ## Register the nested chats
#
# Each player agent will have a nested chat with the board proxy agent to
# make moves on the chess board.

# %% [markdown]
# Define the nested chats,
# when player white receives a message from player black, it will get a message from the board proxy agent.
# and keep getting messages from the board proxy agent until a move is made.
# ![Nested Chat](https://microsoft.github.io/autogen/0.2/assets/images/nested_chat_1-0ed771e6f77cdab64fd58bc3d956fe6e.png)
# %%
player_white.register_nested_chats(
    trigger=player_black,
    chat_queue=[
        {
            "sender": board_proxy,
            "recipient": player_white,
            "summary_method": "last_msg",
        }
    ],
)

player_black.register_nested_chats(
    trigger=player_white,
    chat_queue=[
        {
            "sender": board_proxy,
            "recipient": player_black,
            "summary_method": "last_msg",
        }
    ],
)

# %%
# Start the game

board = chess.Board()

chat_result = player_black.initiate_chat(
    player_white,
    message="Let's play chess! Your move.",
    max_turns=2,
)

# %%
# to keep it running
while not board.is_game_over():
    chat_result = player_black.initiate_chat(
        player_white, message="Keep playing! Your move."
    )

# %% [markdown]
# How to create a complex group chat with multiple agents
# ```python
# groupchat = autogen.GroupChat(
#     agents=[user_proxy, engineer, writer, executor, planner],
#     messages=[],
#     max_round=10,
#     allowed_or_disallowed_speaker_transitions={
#         user_proxy: [engineer, writer, executor, planner],
#         engineer: [user_proxy, executor],
#         writer: [user_proxy, planner],
#         executor: [user_proxy, engineer, planner],
#         planner: [user_proxy, engineer, writer],
#     },
#     speaker_transitions_type="allowed",
# )```
# %%

# %% [markdown]
# Resources:
# - [OpenAI API](https://beta.openai.com/docs/)
# - AI Agentic Design Patterns with AutoGen course - Coursera
