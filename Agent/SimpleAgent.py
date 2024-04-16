# Copyright 2024 Chris Odom.

from .BaseAgent import BaseAgent
from constants import LLMModels
from langchain import hub
from langchain.agents import AgentExecutor, create_json_chat_agent, Tool
from langchain.chains import LLMMathChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from typing import Any, Dict, List, Optional, Callable

class SimpleAgent(BaseAgent):
    def __init__(self, llm=None, tools=None):
        super().__init__(llm)
        self.executor_ = None
        self.create_tools(tools)
        self.create_memory()
        self.langchain_agent = self.create_agent()

    def setUIAsyncCallback(self, cb):
        super().setUIAsyncCallback(cb)
        self.runnable_config_ = RunnableConfig(callbacks=[cb])

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
        prompt = hub.pull("hwchase17/react-chat-json")
        tools = self.GetTools()
        memory = self.GetMemory()
        
        agent = create_json_chat_agent(llm, tools, prompt)
        self.executor_ = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory)
#           return_intermediate_steps=True,
        return agent

    async def execute_ainvoke(self, input: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        res = await self.executor_.ainvoke(input=input, config=self.runnable_config_, **kwargs)
        return res

