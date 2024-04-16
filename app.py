#When is France's national holiday? Do you know the date for that? I just got back from driving all day, so I'm going to be getting some work done FYI. Oh and I wanted to mention, remind me later that my anniversary is coming up. I should buy some flowers for that. Hey also, email Joe and tell him that I want to get his department cleaned up ASAP. Do you know when he was last in the office? I really need to start hitting the gym more often, I've been letting myself go and eating all kinds of garbage.

# I'm thinking about going to Big Bear because I miss my aunt sandy and uncle jack who died. Separately from that I'm going up to Tulsa soon, and I want to visit Meghan when I go up there, so I need to make sure I text her tomorrow. Someday I really want to have my goals tracked, as well as my projects and tasks, and have them linked, and even keep track of dependencies so I can prioritize better. Would be nice to be able to automate all this to whatever degree possible. Could even automate calendar stuff, whenever due date is known. Someday I'd like to buy an F-150 but it's not a huge priority. But I do have a goal that I want to visit my daughter sometime soon. Oh and at work, the company wants me to take over the CI pipeline, so I've been studying up on that stuff. Remember for the CI pipeline the process is to get it building on my local computer first, then make sure the Github Actions work on my local computer, before the final step of actually getting it running up on Github.


#"chat_history": "Human: I hate the bank and their damned surprise recurring charges! They always hit me at the worst time. I've got Rick's wedding coming up on October 23rd, and I have to buy some outfits before my flight. I don't want my debit card to fail when I'm at the mall.\nAI: I'll check Amazon for outfits matching the description you provided.\n",
#"human_message": "ANd I DEFINITELY do need to buy those this week. Oh and BTW, the damned bank hit me with ANOTHER one of their surprises last night, and it caused my account to overdraft, and then my amazon order failed as a result. I want to get some kind of process into place to track those damned things. I should be able to just upload my bank statements and then a chatbot agent should figure out the rest and keep track of them. Also, I need to send my sister a birthday present. Oh hey, BTW, remember that dinosaur conversation we had last week? Apparently they found a new fossilized dinosaur and you could still see its feathers! My kid loves dinosaurs.",

import json
import os
import re
import sys
import asyncio
from pprint import pprint

from enum import Enum, auto

from langchain.chains import LLMMathChain
from langchain.agents import initialize_agent, Tool, AgentType as LangchainAgentType, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

from langchain_openai import ChatOpenAI
from typing import *
from langchain.tools import BaseTool

import chainlit as cl
from chainlit.sync import run_sync

from Agent.BaseAgent import BaseAgent
from Agent.SimpleAgent import SimpleAgent
from Agent.IntentAgent import IntentAgent

from user_intent import UserIntent, UserQuestion, UserGoal, UserCommand, UserProcedure, UserChitchat

from prompts import single_step_intent_extraction
from constants import LLMModels

from dotenv import load_dotenv
load_dotenv()


#LLM_OPENAI_GPT4_TURBO_PREVIEW = "gpt-4-turbo-preview"
#LLM_OPENAI_TEXT_DAVINCI = "text-davinci-003"
#LLM_OPENAI_GPT35_TURBO = "gpt-3.5-turbo"
#LLM_OPENAI_GPT35_TURBO_0125 = "gpt-3.5-turbo-0125"
#LLM_OPENAI_GPT35_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"

class Assistant:
    def __init__(self, name):
        self.name = name

class Team:
    def __init__(self, name):
        self.name = name

def load_data(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    else:
        return {}

def save_data(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

assistants = load_data('assistants.json')
teams = load_data('teams.json')

UserIntent.LoadAllIntents()

class CommandOption(Enum):
    ASSISTANT = (auto(), "assistant")
    TEAM = (auto(), "team")
    INTENT = (auto(), "intent")
    QUESTION = (auto(), "question")
    COMMAND = (auto(), "command")
    GOAL = (auto(), "goal")
    PROCEDURE = (auto(), "procedure")
    CHITCHAT = (auto(), "chitchat")

    def __init__(self, _, command_string):
        self.command_string = command_string

    @staticmethod
    def get_command_option_by_string(command_string) -> 'Optional[CommandOption]':
        """
        Retrieves a command option based on a case-insensitive match or abbreviation of the command string.

        Parameters:
            command_string (str): The command string or its abbreviation.

        Returns:
            Optional[CommandOption]: The matching command option Enum, or None.
        """
        command_string = command_string.lower()
        for command_option in CommandOption:
            # Check if the provided command_string is a prefix of the command_option's command_string
            if command_option.command_string.startswith(command_string):
                return command_option
        return None

    @staticmethod
    def command_option_exists(command_string):
        """
        Checks if a given command option string exists in the command options.
        Parameters:
            command_string (str): The command option to check.
        Returns:
            bool: True if the command string exists, False otherwise.
        """
        command_string = command_string.lower()
        return any(command_option.command_string == command_string for command_option in CommandOption)

    @staticmethod
    def iterate_command_options():
        return [(command_option, command_option.command_string)
                for command_option in CommandOption]

    @staticmethod
    def get_command_strings():
        """
        Returns a formatted string containing all command_strings from the CommandOption enum.
        Returns:
            str: A string formatted as "[value1|value2|value3|...]" where each value is a command_string.
        """
        command_strings = [command_option.command_string for command_option in CommandOption]
        return f"[{'|'.join(command_strings)}]"

def pretty_repr(data, indent=4):
    """
    Returns a pretty-printed representation of a dictionary with `repr()`.
    
    Parameters:
        data (dict): The dictionary to pretty-print.
        indent (int): The indentation level for nested elements.
    
    Returns:
        str: A pretty-printed string.
    """
    pretty = "{\n"
    for key, value in data.items():
        pretty += " " * indent + repr(key) + ": " + repr(value) + ",\n"
    pretty = pretty.rstrip(',\n') + "\n}"
    return pretty

def processUserCommand(command) -> str:
    cmd_parts = command.split()
    cmd = cmd_parts[0]
    retval = ""

    user_id = cl.user_session.get("user_id")
    session_id = cl.user_session.get("id")

    intents = UserIntent.ListAllTypesIntentsAsString(user_id)
    questions = UserQuestion.ListIntentsAsString(user_id)
    procedures = UserProcedure.ListIntentsAsString(user_id)
    commands = UserCommand.ListIntentsAsString(user_id)
    goals = UserGoal.ListIntentsAsString(user_id)
    chitchat = UserChitchat.ListIntentsAsString(user_id)
    cmd_option_strings = CommandOption.get_command_strings()
    
    if '/help'.startswith(cmd):
        retval += ("/commands:\n"
               + "- /help -- Display this message.\n"
               + f"- /list {cmd_option_strings}\n"
               + f"- /show {cmd_option_strings} <NAME/ID>\n"
               +  "- /add [team|assistant] <NAME>\n"
               + f"- /remove {cmd_option_strings} <NAME/ID>\n")
        return retval
    elif '/list'.startswith(cmd):
        if len(cmd_parts) == 1:
            retval += f"List options: {cmd_option_strings}\n"
            return retval
        else:
            cmd_option = CommandOption.get_command_option_by_string(cmd_parts[1].lower())

            if cmd_option == CommandOption.ASSISTANT:
                retval += "Assistants:\n"
                for name in assistants.keys():
                    retval += (name + '\n')
            elif cmd_option == CommandOption.TEAM:
                retval += "Teams:\n"
                for name in teams.keys():
                    retval += (name + '\n')
            elif cmd_option == CommandOption.INTENT:
                retval += f"All Intents:\n{intents}\n"
            elif cmd_option == CommandOption.QUESTION:
                retval += f"Questions:\n{questions}\n"
            elif cmd_option == CommandOption.PROCEDURE:
                retval += f"Procedures:\n{procedures}\n"
            elif cmd_option == CommandOption.COMMAND:
                retval += f"Commands:\n{commands}\n"
            elif cmd_option == CommandOption.GOAL:
                retval += f"Goals:\n{goals}\n"
            elif cmd_option == CommandOption.CHITCHAT:
                retval += f"Chitchat:\n{chitchat}\n"
            else:
                retval += f"Unknown list option: {cmd_parts[1]}\n"
            return retval
    elif '/add'.startswith(cmd) or '/remove'.startswith(cmd) or '/show'.startswith(cmd):
        command_str = ""
        if '/add'.startswith(cmd):
            command_str = 'add'
        elif '/remove'.startswith(cmd):
            command_str = 'remove'
        elif '/show'.startswith(cmd):
            command_str = 'show'

        if len(cmd_parts) == 1:
            if command_str == 'add':
                retval += f"Options: [team|assistant] <NAME>\n"
            else: # remove or show.
                retval += f"Options: {cmd_option_strings} <NAME or ID>\n"
            return retval
        else:
            cmd_option = CommandOption.get_command_option_by_string(cmd_parts[1].lower())

            if cmd_option is None:
                retval += f"Unknown {command_str} option: {cmd_parts[1]}\n"
                return retval
            elif len(cmd_parts) < 3:
                retval += f"Error: Missing name or ID for the {cmd_parts[1]} you want to {command_str}.\n"
                return retval
            else:
                name_pattern = re.compile(r'^[A-Za-z][A-Za-z0-9]*$')

                if any(s.startswith(cmd_parts[1].lower()) for s in ("intent", "question", "procedure", "command", "goal", "chitchat")):
                    if not UserIntent.validate_abbrev_id(cmd_parts[2]):
                        retval += "Error: Invalid ID format.\n"
                        return retval
                    else:
                        intents = UserIntent.GetIntentAbbrev(cmd_parts[2])

                        if len(intents) == 0:
                            retval += f"No matching intents found for ID: {cmd_parts[2]}\n"
                            return retval
                        elif len(intents) == 1:
                            # Access the ID of the single found intent
                            retval += f"One intent found, with ID: {intents[0]['id']}\n"
                        else:
                            # Create a string that contains all IDs from the found intents, joined by commas
                            ids = ', '.join(intent['id'] for intent in intents)
                            retval += f"Multiple intents found. Here are their IDs: {ids}"
    
                        if command_str == 'show':
                            retval += "```json\n"
                            for intent in intents:
                                intent_json = pretty_repr(intent)  # Assuming pretty_repr is defined and formats a single intent
                                retval += f"{intent_json}\n"
                            retval += "```\n"
                            return retval
                        else:
                            if not UserIntent.validate_custom_id(cmd_parts[2]):
                                retval += "Failure: For DELETING intents, you must use the full ID. No abbreviations.\n"
                                return retval
                            intent = UserIntent.GetIntent(cmd_parts[2])
                            if intent is None:
                                retval += f"No matching intents found for ID: {cmd_parts[2]}\n"
                                return retval
                            retval += f"Deleting intent with ID: {cmd_parts[2]}\n"
                            UserIntent.DeleteIntent(cmd_parts[2])
                elif not name_pattern.match(cmd_parts[2]):
                    retval += "Error: Invalid name format.\n"
                else:
                    retval += processAddRemoveShowCommands(cmd_parts)
    return retval


def processAddRemoveShowCommands(cmd_parts):
    retval = ""

    dict_to_use = assistants if cmd_parts[1].lower() == 'assistant' else teams
    filename = 'assistants.json' if cmd_parts[1].lower() == 'assistant' else 'teams.json'
    name = ' '.join(cmd_parts[2:])
    
    
    instance_name = cmd_parts[1].capitalize()
    if cmd_parts[0] == '/add':
        if name.lower() in (n.lower() for n in dict_to_use.keys()):
            retval += f"{instance_name} '{name}' already exists.\n"
        else:
            dict_to_use[name] = {'Name': name}
            retval += f"{instance_name} '{name}' added.\n"
            save_data(filename, dict_to_use)
    elif cmd_parts[0] == '/remove':
        if name.lower() in (n.lower() for n in dict_to_use.keys()):
            dict_to_use.pop(name)
            retval += f"{instance_name} '{name}' removed.\n"
            save_data(filename, dict_to_use)
        else:
            retval += f"{instance_name} '{name}' not found.\n"
    elif cmd_parts[0] == '/show':
            retval += f"{instance_name} '{name}' TODO: SHOW DETAILS.\n"
        
    return retval


    

#from dotenv import load_dotenv
#load_dotenv()

#/Users/au/src/langchain/libs/experimental/langchain_experimental/agents/agent_toolkits/python
# create_python_agent


#/Users/au/src/langchain/libs/langchain/langchain/agents
# create_react_agent
# create_json_chat_agent
# create_openai_tools_agent
# create_openai_functions_agent
# create_structured_chat_agent
# create_sql_agent
# create_self_ask_with_search_agent
# create_xml_agent


class IntentExtratorTool(BaseTool):
    """Tool for extracting the user's intents from his newest message"""

    name = "intent extractor"
    description = (
        "This tool extracts all of the user's intents from his newest message, using chat history as context."
    )

    def _run(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Tool for extracting the user's intents from his most recent message, using the chat history as context."""
        run_sync(cl.Message(content=query).send())
        return "Thank you for the information, now please use a different tool or come up with your final answer."

    async def _arun(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Tool for extracting the user's intents from his most recent message, using the chat history as context."""
        await cl.Message(content=query).send()
        return "Thank you for the information, now please use a different tool or come up with your final answer."


class HumanInputChainlit(BaseTool):
    """Tool that requests clarification from the human."""

    name = "ask human to clarify"
    description = (
        "If you have a question for the human, you can ask for specific information "
        "or guidance, but only when you are stuck or are not sure what to do next. "
        "The input to this tool MUST be a very specific and pointed question for the "
        "human, and it should also include an explanation of WHY you need the information. "
        "Here are some examples of proper input for this tool:\n"
        "Can you give me her phone number? I need it in order to send her that text message.\n"
        "What's Frank's last name? There are multiple contacts named Frank and I don't want to email the wrong one.\n"
        "What time do you want me to set up your dinner reservation? Does 8 PM work? OpenTable requires a specific time to make the reservation.\n"
        "What language should this project be coded in? How about Python, or C++? I need to know this in order to select the correct engineer for the job."
#        "You can ask a human for guidance when you think you "
#        "got stuck or you are not sure what to do next. "
#        "The input should be a question for the human."
    )

    def _run(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Tool to ask the human a clarification question when necessary."""
        res = run_sync(cl.AskUserMessage(content=query).send())
        return res.content

    async def _arun(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Tool to ask the human a clarification question when necessary."""
        res = await cl.AskUserMessage(content=query).send()
        pprint(res)
        return res['output'] # works with agent=LangchainAgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION in BaseAgent.create_agent
#        return res.output # does NOT work with STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION


def createSimpleAgent():
    tools = [
        HumanInputChainlit(),
#        HumanFYIChainlit(),
    ]
    agent = BaseAgent.factory(BaseAgent.AgentType.SIMPLE, llm=None, tools=tools)
    memory = agent.GetMemory()
    cl.user_session.set("memory", memory)
    return agent

#def createIntentAgent():
#    tools = None
##    tools = [
##        HumanInputChainlit(),
##    ]
#    agent = BaseAgent.factory(BaseAgent.AgentType.INTENT, llm=None, tools=tools)
#    memory = agent.GetMemory()
#    cl.user_session.set("memory", memory)
#    return agent

# This is for the UI to allow the user to select between
# USER-FACING agents. If non-user-facing agents appear here,
# it's for testing only.
#
def createAgent(agent_enum=BaseAgent.AgentType.SIMPLE):
    if agent_enum == BaseAgent.AgentType.SIMPLE:
        return createSimpleAgent()
#    elif agent_enum == BaseAgent.AgentType.INTENT:
#        return BaseAgent.factory(BaseAgent.AgentType.INTENT)
#    elif agent_enum == BaseAgent.AgentType.CALC_QUESTION:
#        return createCalcQuestionAgent()
#    elif agent_enum == BaseAgent.AgentType.ASK_HUMAN:
#        return BaseAgent.factory(BaseAgent.AgentType.ASK_HUMAN)
#    elif agent_enum == BaseAgent.AgentType.CONVERSATIONALIST:
#        return createConversationalistAgent()
#    elif agent_enum == BaseAgent.AgentType.ASSISTANT:
#        return createAssistantAgent()
    # Add additional agent types here as elif blocks
    else:
        raise ValueError(f"Unknown agent type: {agent_enum.agent_string}")
        


#When is France's national holiday? Do you know the date for that? I just got back from driving all day, so I'm going to be getting some work done FYI. Oh and I wanted to mention, remind me later that my anniversary is coming up. I should buy some flowers for that. Hey also, email Joe and tell him that I want to get his department cleaned up ASAP. Do you know when he was last in the office? I really need to start hitting the gym more often, I've been letting myself go and eating all kinds of garbage.

# I'm thinking about going to Big Bear because I miss my aunt sandy and uncle jack who died. Separately from that I want to visit Meghan when I go to Tulsa, so I need to make sure I text her tomorrow. Someday I really want to have my goals tracked, as well as my projects and tasks, and have them linked, and even keep track of dependencies so I can prioritize better. Would be nice to be able to automate all this to whatever degree possible. Could even automate calendar stuff, whenever due date is known. Someday I'd like to buy an F-150 but it's not a huge priority. But I do have a goal that I want to visit my daughter sometime soon. Oh and at work, the company wants me to take over the CI pipeline, so I've been studying up on that stuff.


#"chat_history": "Human: I hate the bank and their damned surprise recurring charges! They always hit me at the worst time. I've got Rick's wedding coming up on October 23rd, and I have to buy some outfits before my flight. I don't want my debit card to fail when I'm at the mall.\nAI: I'll check Amazon for outfits matching the description you provided.\n",
#"human_message": "ANd I DEFINITELY do need to buy those this week. Oh and BTW, the damned bank hit me with ANOTHER one of their surprises last night, and it caused my account to overdraft, and then my amazon order failed as a result. I want to get some kind of process into place to track those damned things. I should be able to just upload my bank statements and then a chatbot agent should figure out the rest and keep track of them. Also, I need to send my sister a birthday present. Oh hey, BTW, remember that dinosaur conversation we had last week? Apparently they found a new fossilized dinosaur and you could still see its feathers! My kid loves dinosaurs.",

#All this does is separate out the individual intents from a user message,
#but it doesn't parse the individual intents. That's elsewhere.
def preprocessUserInput(newest_msg, user_id, session_id, chat_history):
    first_choice_text = ""
    
#    if is_single_sentence(newest_msg):
#        first_choice_text = f"""        {{
#            "user_id": "{user_id}",
#            "session_id": "{session_id}",
#            "extracted_intents": [
#                {{"raw_intent": "{newest_msg}"}}
#            ]
#        }}"""
#        return first_choice_text;

    system_prompt = single_step_intent_extraction.format(user_id=user_id, session_id=session_id)
    
    task_prompt = f"""    **** ACTUAL INPUT:
        {{
            "user_id": "{user_id}",
            "session_id": "{session_id}",
            "chat_history": "{chat_history}",
            "human_message": "{newest_msg}"
        }}
        
    **** EXPECTED OUTPUT:
    """
    
    full_prompt = f"{system_prompt}\n\n{task_prompt}\n\n"
    
    res = openai_client.completions.create(model=LLMModels.OPENAI_GPT35_TURBO_INSTRUCT,
        prompt=full_prompt,
        temperature=0.0,
        max_tokens=1024,
        stop="<<INTENTS_EXTRACTED>>")
    first_choice_text = res.choices[0].text
    return first_choice_text



def test_is_single_sentence():
    # Test the function with your examples
    example1 = "Remind me to call my mom this weekend."
    example2 = """        Consider this piece of code:
    
        int main()
        {
        InitializeLibrary();
        DrawObject.draw(default);
        }"""
    example3 = """        // This is the blah generator function.
        // It uses food.
        void blahGenerator()
        {
            generateBlahASAP();
            foo.blahPrepare();
        }"""
    example4 = "Make sure you stop by the chicken farm: I left my rake there. Oh and another thing: I forgot to feed the pigs, make sure you do that when you get home."
    example5 = """Hey here's that scripture we discussed:

    Fallen, fallen, is Babylon the Great.
    For she has made the whole world drink the wine of the wrath of her porneia.
    And by her Pharmakeia she has led the whole world astray."""
    print(is_single_sentence(example1))  # Expected: True
    print(is_single_sentence(example2))  # Expected: True
    print(is_single_sentence(example3))  # Expected: True
    print(is_single_sentence(example4))  # Expected: False
    print(is_single_sentence(example5))  # Expected: True


def is_single_sentence(text):
    # First, check if the text is likely code or contains a colon followed by a block of text (e.g., code or quotation).
    # This regex looks for a colon possibly followed by newlines and open braces, which often denote code blocks.
    if re.search(r":\s*[\n{]", text) or re.search(r"\n", text):
        return True
    
    # Check for periods that would denote multiple sentences, 
    # ignoring periods followed by whitespace and a lowercase letter (inline abbreviations or URLs),
    # and periods at the end of the string.
    # This regex also checks for other sentence-ending punctuation like "?" and "!".
    if re.search(r"[.!?](?=\s+[A-Z])", text):
        return False
    
    # If there are no clear markers of multiple sentences, consider it a single sentence.
    return True
    
@cl.action_callback("question_research")
async def on_action_question_research(action):
    await cl.Message(content=f"Executed {action.name}").send()
    # Optionally remove the action button from the chatbot user interface
#    await action.remove()

@cl.action_callback("question_answer")
async def on_action_question_answer(action):
    await cl.Message(content=f"Executed {action.name}").send()
    # Optionally remove the action button from the chatbot user interface
#    await action.remove()

@cl.action_callback("question_remember")
async def on_action_question_remember(action):
    await cl.Message(content=f"Executed {action.name}").send()
    # Optionally remove the action button from the chatbot user interface
#    await action.remove()

@cl.action_callback("intent_discard")
async def on_action_intent_discard(action):
    await cl.Message(content=f"Executed {action.name}").send()
    # Optionally remove the action button from the chatbot user interface
#    await action.remove()

@cl.action_callback("command_perform")
async def on_action_command_perform(action):
    await cl.Message(content=f"Executed {action.name}").send()
    # Optionally remove the action button from the chatbot user interface
#    await action.remove()

@cl.action_callback("command_research")
async def on_action_command_research(action):
    await cl.Message(content=f"Executed {action.name}").send()
    # Optionally remove the action button from the chatbot user interface
#    await action.remove()

@cl.action_callback("command_remember")
async def on_action_command_remember(action):
    await cl.Message(content=f"Executed {action.name}").send()
    # Optionally remove the action button from the chatbot user interface
#    await action.remove()

@cl.action_callback("goal_research")
async def on_action_goal_research(action):
    await cl.Message(content=f"Executed {action.name}").send()
    # Optionally remove the action button from the chatbot user interface
#    await action.remove()

@cl.action_callback("goal_remember")
async def on_action_goal_remember(action):
    await cl.Message(content=f"Executed {action.name}").send()
    # Optionally remove the action button from the chatbot user interface
#    await action.remove()

@cl.action_callback("chitchat_remember")
async def on_action_chitchat_remember(action):
    await cl.Message(content=f"Executed {action.name}").send()
    # Optionally remove the action button from the chatbot user interface
#    await action.remove()


#    # ----------------------------------------------------
#    # Create the TaskList
#    task_list = cl.TaskList()
#    task_list.status = "Running..."
#
#    # Create a task and put it in the running state
#    task1 = cl.Task(title="Processing data", status=cl.TaskStatus.RUNNING)
#    await task_list.add_task(task1)
#    # Create another task that is in the ready state
#    task2 = cl.Task(title="Performing calculations")
#    await task_list.add_task(task2)
#
#    # Optional: link a message to each task to allow task navigation in the chat history
#    message_id = await cl.Message(content="Started processing data").send()
#    task1.forId = message_id
#
#    # Update the task list in the interface
#    await task_list.send()
#
#    # Perform some action on your end
#    await cl.sleep(1)
#
#    # Update the task statuses
#    task1.status = cl.TaskStatus.DONE
#    task2.status = cl.TaskStatus.FAILED
#    task_list.status = "Failed"
#    await task_list.send()


#success_processing, processed_string = await processUserInput(message.content, sections)
async def processUserInput(user_id, session_id, memory, human_message, sections):

    current_section = 0
    current_section_header = sections[current_section]["header"]
    current_section_messages = sections[current_section]["messages"]
    output_str = ""

    def incrementSection():
        nonlocal current_section, current_section_header, current_section_messages
        current_section += 1
        if current_section < len(sections):
            # Ensure header is not None by creating a cl.Message instance if needed
            if sections[current_section]["header"] is None:
                sections[current_section]["header"] = cl.Message(content="")
            current_section_header = sections[current_section]["header"]
            current_section_messages = sections[current_section]["messages"]
        else:
            print("ERROR: No more sections to increment to.")

    chat_history = memory.buffer_as_str
    
    preprocessedJSON = preprocessUserInput(human_message, user_id, session_id, chat_history)
#    memory.chat_memory.add_user_message(human_message)

    print("PREPROCESSED:\n", preprocessedJSON)
    
#    new_questions = []
#    new_commands = []
#    new_goals = []
#    new_chitchat = []
#    
#    somethingGotImported = UserIntent.Import(preprocessedJSON)
#    
#    if not somethingGotImported:
#        return False, output_str
#        
#    new_questions = UserQuestion.GetNewestImports()
#    new_commands = UserCommand.GetNewestImports()
#    new_goals = UserGoal.GetNewestImports()
#    new_chitchat = UserChitchat.GetNewestImports()
#    
#    prior_output = False
#    output_str = "**If I understand correctly,** "
#    current_section_header.content = output_str
#    
#    if new_questions:
#        temp_str = "**you are asking:**\n"
#        current_section_header.content += temp_str
#        output_str += (temp_str + UserQuestion.ListNewestImportsAsString() + "\n\n")
#        prior_output = True
#        for intent in new_questions:
#            actions=[
#                cl.Action(name="question_research", value=f"{intent['id']}", label="🧬 Do some Research"),
#                cl.Action(name="question_answer", value=f"{intent['id']}", label="🙋 Answer this Now"),
#                cl.Action(name="question_remember", value=f"{intent['id']}", label="🐘 Remember Question"),
#                cl.Action(name="intent_discard", value=f"{intent['id']}", label="❌ Forget Question"),
#            ]
#            msg = cl.Message(content=f"{intent['rephrased_intent']}", actions=actions)
#            current_section_messages.append(msg)
#        incrementSection()
#
#    if new_commands:
#        temp_str = ("**" + ("y" if not prior_output else "And y") + "ou've directed me to track (or perform) these tasks:**\n")
#        current_section_header.content += temp_str
#        output_str += (temp_str + UserCommand.ListNewestImportsAsString() + "\n\n")
#        prior_output = True
#        for intent in new_commands:
#            actions=[
#                cl.Action(name="command_perform", value=f"{intent['id']}", label="🔄 Do this Now"),
#                cl.Action(name="command_research", value=f"{intent['id']}", label="🧬 Research this"),
#                cl.Action(name="command_remember", value=f"{intent['id']}", label="🐘 Remember Task"),
#                cl.Action(name="intent_discard", value=f"{intent['id']}", label="❌ Forget Task"),
#            ]
#            msg = cl.Message(content=f"{intent['rephrased_intent']}", actions=actions)
#            current_section_messages.append(msg)
#        incrementSection()
#
#    if new_goals:
#        temp_str = ("**" + ("you've " if not prior_output else "You also ") + "set these goals:**\n")
#        current_section_header.content += temp_str
#        output_str += (temp_str + UserGoal.ListNewestImportsAsString() + "\n\n")
#        prior_output = True
#        for intent in new_goals:
#            actions=[
#                cl.Action(name="goal_research", value=f"{intent['id']}", label="🧬 Research this"),
#                cl.Action(name="goal_remember", value=f"{intent['id']}", label="🐘 Remember Goal"),
#                cl.Action(name="intent_discard", value=f"{intent['id']}", label="❌ Forget Goal"),
#            ]
#            msg = cl.Message(content=f"{intent['rephrased_intent']}", actions=actions)
#            current_section_messages.append(msg)
#        incrementSection()
#
#    if new_chitchat:
#        temp_str = ("**" + ("you've " if not prior_output else "And you've ") + "mentioned:**\n")
#        current_section_header.content += temp_str
#        output_str += (temp_str + UserChitchat.ListNewestImportsAsString() + "\n\n")
#        prior_output = True
#        for intent in new_chitchat:
#            actions=[
#                cl.Action(name="chitchat_remember", value=f"{intent['id']}", label="🐘 Remember this"),
#                cl.Action(name="intent_discard", value=f"{intent['id']}", label="❌ Forget this"),
#            ]
#            msg = cl.Message(content=f"{intent['rephrased_intent']}", actions=actions)
#            current_section_messages.append(msg)
#
##    if new_commands:
##        output_str += ("**" + ("y" if not prior_output else "And y") +
##            "ou've directed me to track (or perform) these tasks:**\n" +
##            UserCommand.ListNewestImportsAsString() + "\n\n")
##        prior_output = True
##    if new_goals:
##        output_str += ("**" + ("you've " if not prior_output else "You also ") +
##            "set these goals:**\n" + UserGoal.ListNewestImportsAsString() + "\n\n")
##        prior_output = True
##    if new_chitchat:
##        output_str += ("**" + ("you've " if not prior_output else "And you've ") + "mentioned:**\n"
##            + UserChitchat.ListNewestImportsAsString() + "\n\n")
##        prior_output = True

    #--------------------------------------------------------------
    new_questions = ""
    new_commands = ""
    new_procedures = ""
    new_goals = ""
    new_chitchat = ""
    
    if UserIntent.Import(preprocessedJSON):
        if UserQuestion.NewestImportsCount() > 0:
            new_questions = UserQuestion.ListNewestImportsAsString()
        if UserProcedure.NewestImportsCount() > 0:
            new_procedures = UserProcedure.ListNewestImportsAsString()
        if UserCommand.NewestImportsCount() > 0:
            new_commands = UserCommand.ListNewestImportsAsString()
        if UserGoal.NewestImportsCount() > 0:
            new_goals = UserGoal.ListNewestImportsAsString()
        if UserChitchat.NewestImportsCount() > 0:
            new_chitchat = UserChitchat.ListNewestImportsAsString()
    
    prior_output = False
    output_str = "**If I understand correctly,** "
    if new_questions:
        output_str += ("**you are asking:**\n" + new_questions + "\n\n")
        prior_output = True
    if new_commands:
        output_str += ("**" + ("y" if not prior_output else "And y") +
            "ou've directed me to track (or perform) these tasks:**\n" +
            new_commands + "\n\n")
        prior_output = True
    if new_procedures:
        output_str += ("**" + ("y" if not prior_output else "And y") +
            "ou've described (or mentioned) these procedures:**\n" +
            new_procedures + "\n\n")
        prior_output = True
    if new_goals:
        output_str += ("**" + ("you've " if not prior_output else "You also ") +
            "set these goals:**\n" + new_goals + "\n\n")
        prior_output = True
    if new_chitchat:
        output_str += ("**" + ("you've " if not prior_output else "And you've ") + "mentioned:**\n"
            + new_chitchat + "\n\n")
        prior_output = True

#    # Convert the processed output to a JSON string for pretty printing
#    extracted_intents_json = json.dumps(extracted_intents, indent=4)
#    return output_str + "\n```json\n" + preprocessedJSON + "\n```\n\n"
    return True, output_str



#@cl.step
#async def tool():
#    # Simulate a running task
#    await cl.sleep(2)
#
#    return "Response from the tool!"

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin", metadata={"role": "ADMIN"})
    else:
        return None


#def getSessionInfo(user_session: cl.UserSession):
#
#        raise ValueError(f"Unknown agent type: {typeStr}")
    
class ProfileManager:
    _agent_type_to_image_url = {
        BaseAgent.AgentType.SIMPLE: "https://picsum.photos/200",
        BaseAgent.AgentType.INTENT: "https://picsum.photos/250",
#        BaseAgent.AgentType.CONVERSATIONALIST: "https://picsum.photos/300",
        # Add more mappings as necessary
    }

    @staticmethod
    def getChatProfiles():
        profiles = []
        # Use the SIMPLE agent's icon as the default if a specific AgentType's icon isn't found
        default_icon = ProfileManager._agent_type_to_image_url.get(BaseAgent.AgentType.SIMPLE)
        for agent_type in BaseAgent.AgentType.iterate_agent_types():
            agent_enum, agent_string, markdown_description = agent_type
            # Retrieve the icon URL, or use the default if not found
            icon = ProfileManager._agent_type_to_image_url.get(agent_enum, default_icon)
            profile = cl.ChatProfile(
                name=agent_string,
                markdown_description=markdown_description,
                icon=icon,
            )
            profiles.append(profile)
        return profiles

    @staticmethod
    def getAgentEnumByProfileName(name: str) -> BaseAgent.AgentType:
        return BaseAgent.AgentType.get_agent_type_by_string(name)

        
@cl.on_chat_start
async def start():
    user = cl.user_session.get("user")
    user_id = user.identifier
    cl.user_session.set("user_id", user_id)
    session_id = cl.user_session.get("id")
    chat_profile = cl.user_session.get("chat_profile")
    await cl.Message(
        content=f"starting chat with {user.identifier} using the {chat_profile} chat profile"
    ).send()
    agent_enum = ProfileManager.getAgentEnumByProfileName(chat_profile)
    agent = createAgent(agent_enum)
    agent.setUserAndSessionId(user_id, session_id)
    cl.user_session.set("agent", agent)

@cl.set_chat_profiles
async def setup_chat_profiles():
    return ProfileManager.getChatProfiles()
#   return [
#       cl.ChatProfile(
#           name="GPT-3.5",
#           markdown_description="The underlying LLM model is **GPT-3.5**.",
#           icon="https://picsum.photos/200",
#       ),
#       cl.ChatProfile(
#           name="GPT-4",
#           markdown_description="The underlying LLM model is **GPT-4**.",
#           icon="https://picsum.photos/250",
#       ),
#   ]

@cl.on_message
async def main(message: cl.Message):
    res = []
    raw_user_input = message.content
    user_id = cl.user_session.get("user_id")
    session_id = cl.user_session.get("id")
    memory = cl.user_session.get("memory")
    
#    intermediate_output = ""
#    
#    # There will be at LEAST one, which we'll use as the main placeholder.
#    # but then if more than one intent type is found, these placeholders will
#    # start to fill out for those groups.
    top_placeholder = cl.Message(content="")
    sections = [
        {"header": top_placeholder, "messages": []},
        {"header": None, "messages": []},
        {"header": None, "messages": []},
        {"header": None, "messages": []},
    ]
    await top_placeholder.send()
#
##    await cl.sleep(1)
    
    if raw_user_input.startswith('/'):
        top_placeholder.content = processUserCommand(raw_user_input)
        await top_placeholder.update()
    else:
#        success_processing, processed_string = await processUserInput(user_id, session_id, memory, raw_user_input, sections)
        
#        messages = []
#        
#        for section in sections:
#            header = section["header"]
#            if header is not None:
##                print(f"Header: {header.content}")
#                messages.append(header)
#                for msg in section["messages"]:
##                    print(f"    Message: {msg.content}")
#                    messages.append(msg)
#
#        if messages:
#            for msg in messages:
#                await msg.send()
#        else:
#            top_placeholder.content = processed_string
#            await top_placeholder.update()



#        top_placeholder.content = processed_string   #temporary for testing.
#        await top_placeholder.update()
#


#        agent = cl.user_session.get("conversation_agent")  # type: AgentExecutor
#
#        final_output = await agent.arun(
#            processed_string, callbacks=[cl.AsyncLangchainCallbackHandler()])



#    text_content = "Hello, this is a text element."
#    elements = [
#        cl.Text(name="simple_text", content=text_content, display="inline")
#    ]
#    await cl.Message(
#        content=final_output
##        elements=elements
#        ).send()




#        intermediate_output = await cl.AskActionMessage(
#                content="Should I analyze this more deeply?",
#                actions=[
#                    cl.Action(name="analyze", value="analyze", label="✅ Analyze more deeply"),
#                    cl.Action(name="nah", value="nah", label="❌ Nah, just give me your gut reaction"),
#                ],
#            ).send()
#
#        if intermediate_output and intermediate_output.get("value") == "analyze":
#            await cl.Message(
#                content="Analyze more deeply.",
#            ).send()

        current_agent = cl.user_session.get("agent")  # type: AgentExecutor
        cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
        current_agent.setUIAsyncCallback(cb)

        extracted_intents = await IntentExtractorTool(raw_user_input)
        
        memory.chat_memory.add_user_message(raw_user_input) # may be redundant. Not sure how to order these right since the first
        memory.chat_memory.add_ai_message(extracted_intents) # user message is probably added by the framework AFTER this call.
        top_placeholder.content = extracted_intents
        await top_placeholder.update()
        # ------------------------------------------------
        input_for_agent = "Please continue."
        res = await current_agent.ainvoke({"input": input_for_agent})
        await cl.Message(content=res["output"]).send()

@cl.step
async def IntentExtractorTool(message_content):
    user = cl.user_session.get("user")
    user_id = user.identifier
    session_id = cl.user_session.get("id")
    memory = cl.user_session.get("memory")
    
    intent_agent = BaseAgent.factory(BaseAgent.AgentType.INTENT)
    intent_agent.setUserAndSessionId(user_id, session_id)
    intent_agent.setExternalChatHistoryStr(memory.buffer_as_str)
    
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
    intent_agent.setUIAsyncCallback(cb)
    
    res = await intent_agent.ainvoke({"input": message_content})
    return res["output"]


