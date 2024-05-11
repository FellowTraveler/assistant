import asyncio
import importlib

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable, Tuple

from langchain_core.runnables.config import RunnableConfig

class BaseAgent(ABC):
    def __init__(self, llm=None, tools=None):
        self.is_running_ = False
        self.lock_ = asyncio.Lock()
        self.llm_ = llm
        self.tools_ = []
#        if tools is not None:  # Note: we'll do this in create_tools.
#            self.tools.extend(tools) # (Which the subclass constructor will call).
        self.memory_ = None
        self.ui_async_callback_ = None
        self.runnable_config_ = None
        self.user_id_ = None
        self.session_id_ = None
    
    def getUserAndSessionId(self) -> Tuple[Optional[str], Optional[str]]:
        return (self.user_id_, self.session_id_)

    def setUserAndSessionId(self, user_id, session_id):
        self.user_id_ = user_id
        self.session_id_ = session_id

    def setUIAsyncCallback(self, cb):
        self.ui_async_callback_ = cb

    class AgentType(Enum):
        SIMPLE = (auto(), True, "SimpleAgent", "This is a **Simple Agent**.")
        INTENT = (auto(), False, "IntentAgent", "This is an **Intent Extractor Agent**.")

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
        """
        Factory method to create agents dynamically based on the agent_type.
        """
        try:
            # Dynamically import the agent class based on the agent_string attribute of the enum
            module = importlib.import_module("." + agent_type.agent_string, package="Agent")
            agent_class = getattr(module, agent_type.agent_string)
            return agent_class(llm, tools)
        except (AttributeError, ImportError) as e:
            raise ImportError(f"Could not import the agent type {agent_type.name}: {e}") from e

    @classmethod
    def create_llm(cls):
        """
        Method to create the llm. Subclass must provide.
        """
        raise Exception("Missing create_llm implementation.")

    @abstractmethod
    def create_agent(self):
        """
        Method to initialize and return the internal agent implementation.
        """
        raise Exception("Missing create_llm implementation.")

    @abstractmethod
    def create_tools(self, tools):
        """
        Method for subclass to create custom tools.
        Note that these tools are specific to subclass and will be in ADDITION
        to any other tools that were passed in on construction.
        (Which are passed in here and are added to the internally-created ones.)
        """
        pass

    # This method not declared as abstract since memory is optional.
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

    async def execute_ainvoke(self, input: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        The core logic of the agent's operation.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    async def ainvoke(self, input: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        result = None
        async with self.lock_:
            if self.is_running_:
                raise Exception("Agent is already running.")
            self.is_running_ = True

        try:
            result = await self.execute_ainvoke(input=input, **kwargs)
        finally:
            async with self.lock_:
                self.is_running_ = False
        return result

