# Copyright 2024 Chris Odom.

from .BaseAgent import BaseAgent
from constants import LLMModels
from langchain import hub
from langchain.agents import AgentExecutor, create_json_chat_agent, Tool
from langchain.chains import LLMMathChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
import json
import textwrap
from typing import Any, Dict, List, Optional, Callable

import chainlit as cl
from chainlit.sync import run_sync

from user_intent import UserIntent, UserQuestion, UserProcedure, UserGoal, UserCommand, UserChitchat

from pprint import pprint
from prompts import single_step_intent_extraction, intent_chat_json


from dotenv import load_dotenv
load_dotenv()

import os

from openai import OpenAI
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


class IntentAgent(BaseAgent):
    def __init__(self, llm=None, tools=None):
        super().__init__(llm)
        self.executor_ = None
        self.create_tools(tools)
        self.create_memory()
        self.langchain_agent = self.create_agent()
        self.user_import_displays_ = {}
        self.external_chat_history_str_ = None

    def setExternalChatHistoryStr(self, external_chat_history=None):
        self.external_chat_history_str_ = external_chat_history
        
    def setRecentImportUserDisplayText(self, user_id, display_text):
        """
        Set the display text for a given user ID.
        
        :param user_id: str, the unique identifier for the user
        :param display_text: str, the text to be displayed for the user
        """
        self.user_import_displays_[user_id] = display_text

    def getRecentImportUserDisplayText(self, user_id):
        """
        Get the display text for a given user ID.
        
        :param user_id: str, the unique identifier for the user
        :return: str or None, the display text for the user if exists, else None
        """
        return self.user_import_displays_.get(user_id, "")

    def setUIAsyncCallback(self, cb):
        super().setUIAsyncCallback(cb)
        self.runnable_config_ = RunnableConfig(callbacks=[cb])

    def getRecentImportsUserText(self):
        return self.most_recent_imports_user_text_
        
    @classmethod
    def create_llm(cls):
        return ChatOpenAI(temperature=0, streaming=True, model_name=LLMModels.LLM_DEFAULT)

    def create_memory(self):
        self.memory_ = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        pass

    def create_tools(self, tools):
        llm = self.GetLLM(createIfNone=True)
            
        # The tool most likely to be passed in as a parameter is the
        # "Ask Human" tool. So we put this LAST, so that the agent doesn't
        # just default to asking the human before trying more appropriate
        # tools first.
        if tools is not None:
            self.tools_.extend(tools)

        # This is where we create any internally-created tools. Whereas the ones
        # that were passed in, might be ChainlitHumanFeedback or whatever and created externally.
        # (That's because maybe the cl object had to be available when that tool was constructed).
        # But here we'll construct every tool that this AgentType would normally construct
        # internally that is custom to itself.
        
        llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
        
        my_tools = [
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="useful for when you need to answer questions about math",
                coroutine=llm_math_chain.arun,
            ),
            TavilySearchResults(max_results=1), # for web searching.
        ]
        self.tools_.extend(my_tools)
        
    
    def create_agent(self):
        llm = self.GetLLM(createIfNone=True)
        prompt = hub.pull("fellowtraveler/intent-chat-json-cot")
        tools = self.GetTools()
        memory = self.GetMemory()
        
        agent = create_json_chat_agent(llm, tools, prompt)
        self.executor_ = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory)
#                                       return_intermediate_steps=True,
        return agent

    async def execute_ainvoke(self, input: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        res = None
        memory = self.GetMemory()
        user_input_str:str = input["input"]
        success_processing, processed_string = await self.processUserInput(human_message=user_input_str)
        if not success_processing:
            return {
                'input': user_input_str,
#                'chat_history': memory.load_memory_variables({}),
                'output': processed_string
            }
        else:
#            print(f"--- User input string:\n{user_input_str}\n")
#            print(f"--- Processed intents:\n{processed_string}\n")
#            user_input_str = processed_string
#            input["input"] = ("Just send this string back as the final answer:\n" + processed_string)
#            res = await self.executor_.ainvoke(input=input, config=self.runnable_config_, **kwargs)
#            pprint(res)
            return {
                'input': user_input_str,
#                'chat_history': memory.load_memory_variables({}),
                'output': processed_string
            }


    #    processUserInput()
    #    - Returns bool based purely on whether any INTENTS were successfully IMPORTED.
    #    IOW, there is still possibly an OUTPUT DISPLAY STRING returned, even though no
    #    intents were necessarily imported. (Though normally an intent would always be
    #    imported, given that the default intent is "chitchat.")
    async def processUserInput(self, human_message):
        output_str = ""
        intents_were_imported = False
        user_id, session_id = self.getUserAndSessionId()
        preprocessedJSON = self.preprocessUserInput(human_message)
        if not preprocessedJSON:
            print("IntentAgent: Failed to preprocess user input: ", human_message)
            return False, f"IntentAgent: Failed to preprocess user input: {human_message}"
        print("IntentAgent: PREPROCESSED user input:\n", preprocessedJSON)
        #--------------------------------------------------------------
        new_questions = ""
        new_commands = ""
        new_procedures = ""
        new_goals = ""
        new_chitchat = ""
        
        if not UserIntent.Import(preprocessedJSON):
            print("IntentAgent: UserIntent.Import(preprocessedJSON) failed to import any intents from the above preprocessed json.")
            output_str = f"IntentAgent: UserIntent.Import failed to import intents from preprocessedJSON:\n```json\n{preprocessedJSON}\n```\n"
            return False, output_str
        else:
            if UserQuestion.NewestImportsCount(user_id) > 0:
                new_questions = UserQuestion.ListNewestImportsAsString(user_id)
            if UserCommand.NewestImportsCount(user_id) > 0:
                new_commands = UserCommand.ListNewestImportsAsString(user_id)
            if UserProcedure.NewestImportsCount(user_id) > 0:
                new_procedures = UserProcedure.ListNewestImportsAsString(user_id)
            if UserGoal.NewestImportsCount(user_id) > 0:
                new_goals = UserGoal.ListNewestImportsAsString(user_id)
            if UserChitchat.NewestImportsCount(user_id) > 0:
                new_chitchat = UserChitchat.ListNewestImportsAsString(user_id)
        
            def formatIntentsForUser():
                nonlocal new_questions, new_commands, new_procedures, new_goals, new_chitchat, intents_were_imported
                return_str = "**If I understand correctly,** "
                if new_questions:
                    return_str += ("**you are asking:**\n" + new_questions + "\n\n")
                    intents_were_imported = True
                if new_commands:
                    return_str += ("**" + ("y" if not intents_were_imported else "And y") +
                        "ou've directed me to track (or perform) these tasks:**\n" +
                        new_commands + "\n\n")
                    intents_were_imported = True
                if new_procedures:
                    return_str += ("**" + ("y" if not intents_were_imported else "And y") +
                        "ou've described (or mentioned) these procedures:**\n" +
                        new_procedures + "\n\n")
                    intents_were_imported = True
                if new_goals:
                    return_str += ("**" + ("you've " if not intents_were_imported else "You also ") +
                        "set these goals:**\n" + new_goals + "\n\n")
                    intents_were_imported = True
                if new_chitchat:
                    return_str += ("**" + ("you've " if not intents_were_imported else "And you've ") + "mentioned:**\n"
                        + new_chitchat + "\n\n")
                    intents_were_imported = True
                return return_str

            def formatIntentsForAgent():
                nonlocal new_questions, new_commands, new_procedures, new_goals, new_chitchat, intents_were_imported
                return_str = "The user's distinct extracted intents are listed below; each one starts on its own new line with a '--' prefix (2 dashes) followed by the intent description, and then a parenthetical containing its intent_type and intent_id:\n"
                
                def append_intent(section_name, data):
                    nonlocal return_str, intents_were_imported
                    if data:
                        return_str += f"{section_name}:\n{data}\n\n"
                        intents_were_imported = True

                append_intent("QUESTIONS", new_questions)
                append_intent("COMMANDS", new_commands)
                append_intent("PROCEDURES", new_procedures)
                append_intent("GOALS", new_goals)
                append_intent("CHITCHAT", new_chitchat)
                
#                if new_questions:
#                    return_str += ("QUESTIONS:\n" + new_questions + "\n\n")
#                    intents_were_imported = True
#                if new_commands:
#                    return_str += ("COMMANDS:\n" + new_commands + "\n\n")
#                    intents_were_imported = True
#                if new_procedures:
#                    return_str += ("PROCEDURES:\n" + new_procedures + "\n\n")
#                    intents_were_imported = True
#                if new_goals:
#                    return_str += ("GOALS:\n" + new_goals + "\n\n")
#                    intents_were_imported = True
#                if new_chitchat:
#                    return_str += ("CHITCHAT:\n" + new_chitchat + "\n\n")
#                    intents_were_imported = True

                return return_str

            output_str = formatIntentsForUser()
            self.setRecentImportUserDisplayText(user_id, output_str)
        return intents_were_imported, output_str


    def preprocessUserInput(self, newest_msg):
        user_id, session_id = self.getUserAndSessionId()
        memory = self.GetMemory()
        chat_history = self.external_chat_history_str_ if self.external_chat_history_str_ is not None else memory.buffer_as_str

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
                
        res = None
        try:
            res = openai_client.chat.completions.create(
                model=LLMModels.OPENAI_GPT35_TURBO_0125,
                messages=[
                    {"role": "system", "content": f"{system_prompt}" },
                    {"role": "user", "content": f"{task_prompt}" }
                ],
                max_tokens=1500,  # Adjust based on how much completion you need
                temperature=0.1,  # Adjust creativity
                response_format={ "type": "json_object" },
                stop=["<<INTENTS_EXTRACTED>>"]
            )
            first_choice_text = res.choices[0].message.content
            return first_choice_text
            
        except Exception as e:
            print(f"An error occurred: {e}")
            
        return ""

