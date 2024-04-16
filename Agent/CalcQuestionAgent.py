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