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
        pass

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