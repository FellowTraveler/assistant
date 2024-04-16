"""
# Base callback handler interface
class BaseCallbackHandler:
    def handle_output(self, output):
        raise NotImplementedError

# Asynchronous callback handler
class AsyncCallbackHandler(BaseCallbackHandler):
    async def handle_output(self, output):
        # Implementation for handling output asynchronously
        pass

# Synchronous callback handler
class SyncCallbackHandler(BaseCallbackHandler):
    def handle_output(self, output):
        # Implementation for handling output synchronously
        pass

# COALA class example
class SomeCOALAClass:
    def __init__(self, callback_handler: BaseCallbackHandler):
        self.callback_handler = callback_handler

    async def do_something_async(self):
        # Example async operation
        output = await some_async_operation()
        await self.callback_handler.handle_output(output)

    def do_something_sync(self):
        # Example sync operation
        output = some_sync_operation()
        self.callback_handler.handle_output(output)
"""

from abc import ABC, abstractmethod
import asyncio
from typing import List, Callable

from langchain import hub
from langchain.agents import initialize_agent, Tool, AgentType as LangchainAgentType, AgentExecutor
from langchain.chains import LLMMathChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.tools import BaseTool
from enum import Enum, auto
from constants import LLMModels
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, create_react_agent, create_json_chat_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

    # SimpleAgent should have simple memory and 1 tool (calculator) and thus
    # be able to maintain a consistent conversation, and be able to do math
    # without hallucinating the answer. This should probably be a default tool
    # on many agents -- and all user-facing agents -- due to its near-universal
    # value.
    # It should continually update its memory stream and should repeat a summary
    # of the user's comment back to the user before answering. This is important
    # to show that the agent is really listening!
    #
    # Conversational Agent should have all this also, but should also have the
    # ability to use the Ask-Human agent whenever it is unsure about what you mean.
    # It should assemble this info first:
    #   - Its summarization of the user's statement.
    #   - Its explanation of what it's still uncertain about in that user statement.
    #   - A description of the exact info that it really needs to get from the human.
    #   - The chat history (or relevant portions) for context.
    # Conversationalist should be able to get an answer back from the Ask-Human
    # agent, who may perform several turns with the user before it's satisfied with
    # the "final answer." Then Conversationalist should update its memory stream
    # with the original user input, rephrased to remove the previously uncertain
    # bits, taking the Ask-Human answers into account during that process.
    # This is a clarification step that may eventually include DB lookups.
    # It's about clarification, disambiguation, and possibly separation of Raw
    # Intents, but that would require JSON and so I may add a "RawIntentExtractor"
    # for that part. We may also have the Conversationalist use IntentExtractor
    # as described below.
    #
    # The AssistantAgent is like the Conversationalist but it additionally does
    # an IntentExtractor if not already done, at minimum to provide preliminary
    # classification in the set [question, command, goal, chitchat/info], along
    # with contextual information related to each intent.
    # AssistantAgent will also have to compare to existing intentions stored
    # previous in the local DB, and merge/update/delete those records appropriately,
    # as well as perform additional Ask-Human steps to clarify/disambiguate/prioritize
    # those intentions.
    # At some point in Conversational or Assistant agents, it will also be necessary
    # to perform entity/relationship extraction and/or clarification/disambiguation
    # along with local DB lookup/update in the local entity and/or graph database.
    # (We'll get to that).


class BaseAgent(ABC):
    def __init__(self, llm=None):
        self.is_running_ = False
        self.lock_ = asyncio.Lock()
        self.llm_ = llm
        self.tools_ = []
#        if tools is not None:  # Note: we'll do this in create_tools.
#            self.tools.extend(tools) # (Which the subclass constructor will call).
        self.chat_history_ = None
        self.memory_ = None
        self.executor_ = None
        
    class AgentType(Enum):
        SIMPLE = (auto(), True, "SimpleAgent", "This is a **Simple Agent**.")
        CALC_QUESTION = (auto(), True, "CalculationQuestionAgent", "This is a **Calculation Question Agent**.")
        ASK_HUMAN = (auto(), True, "AskHuman", "This is an **'Ask Human' Agent** who is used by other agents whenever they need an answer from the human user.")
        CONVERSATIONALIST = (auto(), True, "Conversationalist", "This is a **Conversation Agent** who values you and cares about your conversation together.")
        ASSISTANT = (auto(), True, "Assistant", "This is a **Faithful Assistant** who cares about your intentions, priorities, and needs above all else.")
        INTENT_EXTRACTOR = (auto(), True, "IntentExtractor", "**Intent Extractor** used for analyzing input and extracting and tagging/classifying ALL of the intentions found in it.")

        def __init__(self, _, userFacing, agent_string, markdown_description):
            self.user_facing = userFacing
            self.agent_string = agent_string
            self.markdown_description = markdown_description

        @staticmethod
        def get_agent_type_by_string(agent_string):
            for agent_type in BaseAgent.AgentType:
                if agent_type.agent_string == agent_string:
                    return agent_type
            raise ValueError(f"No AgentType found for string: {agent_string}")

        @staticmethod
        def iterate_agent_types(userFacingOnly=True):
            return [(agent_type, agent_type.agent_string, agent_type.markdown_description)
                    for agent_type in BaseAgent.AgentType
                    if not userFacingOnly or agent_type.user_facing]

    @staticmethod
    def factory(agent_type, llm=None, tools=None):
        if agent_type == BaseAgent.AgentType.SIMPLE:
            return SimpleAgent(llm, tools)
        elif agent_type == BaseAgent.AgentType.CALC_QUESTION:
            return CalcQuestionAgent(llm, tools)
        elif agent_type == BaseAgent.AgentType.ASK_HUMAN:
            return AskHumanAgent(llm, tools)
        elif agent_type == BaseAgent.AgentType.CONVERSATIONALIST:
            return ConversationalistAgent(llm, tools)
        elif agent_type == BaseAgent.AgentType.ASSISTANT:
            return AssistantAgent(llm, tools)
        elif agent_type == BaseAgent.AgentType.INTENT_EXTRACTOR:
            return IntentExtractorAgent(llm, tools)
        else:
            raise ValueError("Unknown AgentType")

    async def perceive_environment(self, environment_data):
        # This might be where you set up initial conditions or input for the agent's execution cycle
        self.environment_data = environment_data

    @abstractmethod
    async def execute_cycle(self, content: str, callbacks: List[Callable]):
        """
        The core logic of the agent's operation.
        """
        pass

    @classmethod
    def create_llm(cls): # This is the default implementation, subclasses should override.
        return ChatOpenAI(temperature=0, streaming=True, model_name=LLMModels.LLM_DEFAULT)

    @abstractmethod
    def create_agent(self):
        """
        Method to initialize and return the internal agent implementation.
        """
        pass

    @abstractmethod
    def create_tools(self, tools):
        """
        Method for subclass to create custom tools.
        Note that these tools are specific to subclass and will be in ADDITION
        to any other tools that were passed in on construction.
        (Which are passed in here and are added to the internally-created ones.)
        """
        pass

    def create_memory(self):
        """
        Method for subclass to create its memory store (if applicable).
        """
        pass

    def GetLLM(self, createIfNone=False):
        if self.llm_ is None and createIfNone:
            self.llm_ = self.create_llm()
        if self.llm_ is None:
            raise Exception("Agent: failed in GetLLM.")
        return self.llm_

    def GetTools(self):
        if self.tools_ is None:
            raise Exception("BaseAgent: failed in GetTools.")
        return self.tools_

    def GetMemory(self):
        if self.memory_ is None:
            raise Exception("BaseAgent: failed in GetMemory.")
        return self.memory_

    def GetChatHistory(self):
        if self.chat_history_ is None:
            raise Exception("BaseAgent: failed in GetChatHistory.")
        return self.chat_history_

    async def arun(self, content: str, callbacks: List[Callable]):
        async with self.lock_:
            if self.is_running_:
                raise Exception("Agent is already running.")
            self.is_running_ = True

        try:
            result = await self.execute_cycle(content, callbacks)
        finally:
            async with self.lock_:
                self.is_running_ = False
        return result


#class SimpleAgent(BaseAgent):
#    def __init__(self, llm=None, tools=None):
#        super().__init__(llm)
#        self.create_tools(tools)
#        self.langchain_agent = self.create_agent()
#
#    @classmethod
#    def create_llm(cls):
#        return ChatOpenAI(temperature=0, streaming=True, model_name=LLMModels.LLM_DEFAULT)
#
#    def create_tools(self, tools):
#        llm = self.GetLLM(createIfNone=True)
#
#        if tools is not None:
#            self.tools_.extend(tools)
#            
#        # This is where we create any internally-created tools. Whereas the ones
#        # that were passed in, might be ChainlitHumanFeedback or whatever and created externally.
#        # (That's because maybe the cl object had to be available when that tool was constructed).
#        # But here we'll construct every tool that this AgentType would normally construct
#        # internally that is custom to itself.
#        
#        llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
#        
#        my_tools = [
#            Tool(
#                name="Calculator",
#                func=llm_math_chain.run,
#                description="useful for when you need to answer questions about math",
#                coroutine=llm_math_chain.arun,
#            ),
#        ]
#        self.tools_.extend(my_tools)
#
#    def create_agent(self):
#        llm = self.GetLLM(createIfNone=True)
#        tools = self.GetTools()
#        agent = initialize_agent(tools, llm, agent=LangchainAgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
#        return agent
#
#    async def execute_cycle(self, content: str, callbacks: List[Callable]):
#        res = await self.langchain_agent.arun(content, callbacks=callbacks)
#        return res



# SimpleAgent should have simple memory and 1 tool (calculator) and thus
# be able to maintain a consistent conversation, and be able to do math
# without hallucinating the answer. Calculator should probably be a default tool
# on many agents -- and all user-facing agents -- due to its near-universal
# value.
# It should continually update its memory stream and should repeat a summary
# of the user's comment back to the user before answering. This is important
# to show that the agent is really listening!

#async def on_chat_start():
#    model = ChatOpenAI(streaming=True)
#    prompt = ChatPromptTemplate.from_messages(
#        [
#            (
#                "system",
#                "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
#            ),
#            ("human", "{question}"),
#        ]
#    )
#    runnable = prompt | model | StrOutputParser()
#    cl.user_session.set("runnable", runnable)
    

class SimpleAgent(BaseAgent):
    def __init__(self, llm=None, tools=None):
        super().__init__(llm)
        self.create_tools(tools)
        self.create_memory()
        self.langchain_agent = self.create_agent()

    @classmethod
    def create_llm(cls):
        return ChatOpenAI(temperature=0, streaming=True, model_name=LLMModels.LLM_DEFAULT)

    def create_memory(self):
        self.chat_history_ = MessagesPlaceholder(variable_name="chat_history")
        self.memory_ = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def create_tools(self, tools):
        llm = self.GetLLM(createIfNone=True)

        if tools is not None:
            self.tools_.extend(tools)
            
        # This is where we create any internally-created tools. Whereas the ones
        # that were passed in, might be ChainlitHumanFeedback or whatever and created externally.
        # (That's because maybe the cl object had to be available when that tool was constructed).
        # But here we'll construct every tool that this AgentType would normally construct
        # internally that is custom to itself.
        
        llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
        
        my_tools = [
            TavilySearchResults(max_results=1), # for web searching.
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="useful for when you need to answer questions about math",
                coroutine=llm_math_chain.arun,
            ),
        ]
        self.tools_.extend(my_tools)
    

    def create_agent(self):
        llm = self.GetLLM(createIfNone=True)
        prompt = hub.pull("hwchase17/react-chat-json")
        tools = self.GetTools()
        memory = self.GetMemory()
        chat_history = self.GetChatHistory()
        
#        agent = initialize_agent(
#            tools,
#            llm,
#            agent=LangchainAgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, # works with return res['output'] in HumanInputChainlit
##            agent=LangchainAgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#            verbose=True,
#            agent_kwargs={
#                "memory_prompts": [chat_history],
#                "input_variables": ["input", "agent_scratchpad", "chat_history"]
#            },
#            memory=memory,
#        )

        agent = create_json_chat_agent(llm, tools, prompt)
        self.executor_ = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        return agent

    async def execute_cycle(self, content: str, callbacks: List[Callable]):
#        res = await self.executor_.ainvoke({"input": f"{content}"}, callbacks=callbacks)
        res = await self.executor_.ainvoke(
        {
            "input": f"{content}"},
            "chat_history": [
                HumanMessage(content="hi! my name is bob"),
                AIMessage(content="Hello Bob! How can I assist you today?"),
            ],
        }, callbacks=callbacks)
        return res
        
class CalcQuestionAgent(BaseAgent):
    def __init__(self, llm=None, tools=None):
        super().__init__(llm)
        self.create_tools(tools)
        self.langchain_agent = self.create_agent()

    @classmethod
    def create_llm(cls):
        return ChatOpenAI(temperature=0, streaming=True, model_name=LLMModels.OPENAI_GPT4_TURBO_PREVIEW)
    
    def create_tools(self, tools):
        llm = self.GetLLM(createIfNone=True)

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
        ]
        self.tools_.extend(my_tools)

    def create_agent(self):
        llm = self.GetLLM(createIfNone=True)
        tools = self.GetTools()
        agent = initialize_agent(tools, llm, agent=LangchainAgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        return agent

    async def execute_cycle(self, content: str, callbacks: List[Callable]):
        res = await self.langchain_agent.arun(content, callbacks=callbacks)
        return res

class AskHumanAgent(BaseAgent):
    def __init__(self, llm=None, tools=None):
        super().__init__(llm)
        self.create_tools(tools)
        self.create_memory()
        self.langchain_agent = self.create_agent()

    @classmethod
    def create_llm(cls):
        return ChatOpenAI(temperature=0, streaming=True, model_name=LLMModels.LLM_DEFAULT)

    def create_memory(self):
        self.chat_history_ = MessagesPlaceholder(variable_name="chat_history")
        self.memory_ = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # the "ask human" tool is passed in since it's specific to the UI framework.
    def create_tools(self, tools):
        llm = self.GetLLM(createIfNone=True)

        if tools is not None:
            self.tools_.extend(tools)
            
        # This is where we create any internally-created tools. Whereas the ones
        # that were passed in, might be ChainlitHumanFeedback or whatever and created externally.
        # (That's because maybe the cl object had to be available when that tool was constructed).
        # But here we'll construct every tool that this AgentType would normally construct
        # internally that is custom to itself.
        
        # No calculator for this agent. Left this here for reference in case
        # other tools need to be added.
#        llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
#        
#        my_tools = [
#            Tool(
#                name="Calculator",
#                func=llm_math_chain.run,
#                description="useful for when you need to answer questions about math",
#                coroutine=llm_math_chain.arun,
#            ),
#        ]
#        self.tools_.extend(my_tools)
    
    def create_agent(self):
        llm = self.GetLLM(createIfNone=True)
        tools = self.GetTools()
        memory = self.GetMemory()
        chat_history = self.GetChatHistory()
        
        agent = initialize_agent(
            tools,
            llm,
#            agent=LangchainAgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, # works with return res['output'] in HumanInputChainlit
            agent=LangchainAgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs={
                "memory_prompts": [chat_history],
                "input_variables": ["input", "agent_scratchpad", "chat_history"]
            },
            memory=memory,
        )
        return agent

    async def execute_cycle(self, content: str, callbacks: List[Callable]):
        res = await self.langchain_agent.arun(content, callbacks=callbacks)
        return res

class ConversationalistAgent(BaseAgent):
    def __init__(self, llm=None, tools=None):
        super().__init__(llm)
        self.create_tools(tools)
        self.create_memory()
        self.langchain_agent = self.create_agent()

    @classmethod
    def create_llm(cls):
        return ChatOpenAI(temperature=0, streaming=True, model_name=LLMModels.LLM_DEFAULT)

    def create_memory(self):
        self.chat_history_ = MessagesPlaceholder(variable_name="chat_history")
        self.memory_ = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def create_tools(self, tools):
        llm = self.GetLLM(createIfNone=True)

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
        ]
        self.tools_.extend(my_tools)
    
    def create_agent(self):
        llm = self.GetLLM(createIfNone=True)
        tools = self.GetTools()
        memory = self.GetMemory()
        chat_history = self.GetChatHistory()
        
        agent = initialize_agent(
            tools,
            llm,
#            agent=LangchainAgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, # works with return res['output'] in HumanInputChainlit
            agent=LangchainAgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs={
                "memory_prompts": [chat_history],
                "input_variables": ["input", "agent_scratchpad", "chat_history"]
            },
            memory=memory,
        )
        return agent

    async def execute_cycle(self, content: str, callbacks: List[Callable]):
        res = await self.langchain_agent.arun(content, callbacks=callbacks)
        return res

class AssistantAgent(BaseAgent):
    def __init__(self, llm=None, tools=None):
        super().__init__(llm)
        self.create_tools(tools)
        self.create_memory()
        self.langchain_agent = self.create_agent()

    @classmethod
    def create_llm(cls):
        return ChatOpenAI(temperature=0, streaming=True, model_name=LLMModels.LLM_DEFAULT)

    def create_memory(self):
        self.chat_history_ = MessagesPlaceholder(variable_name="chat_history")
        self.memory_ = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def create_tools(self, tools):
        llm = self.GetLLM(createIfNone=True)

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
        ]
        self.tools_.extend(my_tools)
    
    def create_agent(self):
        llm = self.GetLLM(createIfNone=True)
        tools = self.GetTools()
        memory = self.GetMemory()
        chat_history = self.GetChatHistory()
        
        agent = initialize_agent(
            tools,
            llm,
#            agent=LangchainAgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, # works with return res['output'] in HumanInputChainlit
            agent=LangchainAgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs={
                "memory_prompts": [chat_history],
                "input_variables": ["input", "agent_scratchpad", "chat_history"]
            },
            memory=memory,
        )
        return agent

    async def execute_cycle(self, content: str, callbacks: List[Callable]):
        res = await self.langchain_agent.arun(content, callbacks=callbacks)
        return res

class IntentExtractorAgent(BaseAgent):
    def __init__(self, llm=None, tools=None):
        super().__init__(llm)
        self.create_tools(tools)
        self.create_memory()
        self.langchain_agent = self.create_agent()

    @classmethod
    def create_llm(cls):
        return ChatOpenAI(temperature=0, streaming=True, model_name=LLMModels.LLM_DEFAULT)

    def create_memory(self):
        self.chat_history_ = MessagesPlaceholder(variable_name="chat_history")
        self.memory_ = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def create_tools(self, tools):
        llm = self.GetLLM(createIfNone=True)

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
        ]
        self.tools_.extend(my_tools)
    
    def create_agent(self):
        llm = self.GetLLM(createIfNone=True)
        tools = self.GetTools()
        memory = self.GetMemory()
        chat_history = self.GetChatHistory()
        
        agent = initialize_agent(
            tools,
            llm,
#            agent=LangchainAgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, # works with return res['output'] in HumanInputChainlit
            agent=LangchainAgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs={
                "memory_prompts": [chat_history],
                "input_variables": ["input", "agent_scratchpad", "chat_history"]
            },
            memory=memory,
        )
        return agent

    async def execute_cycle(self, content: str, callbacks: List[Callable]):
        res = await self.langchain_agent.arun(content, callbacks=callbacks)
        return res

