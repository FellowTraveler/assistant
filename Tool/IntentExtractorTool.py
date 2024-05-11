#
#@cl.step
#async def IntentExtractorTool(message_content):
#    user = cl.user_session.get("user")
#    user_id = user.identifier
#    session_id = cl.user_session.get("id")
#    memory = cl.user_session.get("memory")
#
#    intent_agent = BaseAgent.factory(BaseAgent.AgentType.INTENT)
#    intent_agent.setUserAndSessionId(user_id, session_id)
#    intent_agent.setExternalChatHistoryStr(memory.buffer_as_str)
#
#    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
#    intent_agent.setUIAsyncCallback(cb)
#
#    res = await intent_agent.ainvoke({"input": message_content})
#    return res["output"]

from ..Utility.Assertion import InitializationError, type_check, none_check
import logging
logger = logging.getLogger(__name__)
from prompts import single_step_intent_extraction
from constants import LLMModels
from user_intent import UserIntent, UserQuestion, UserProcedure, UserGoal, UserCommand, UserChitchat
from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
from typing import Optional
from langchain.tools import BaseTool

# ---------------------------------------------------------------------
@none_check
class IntentExtractorTool(BaseTool):
    """Tool for extracting the user's intents from his newest message"""

    name = "intent_extractor"
    description = (
        "Tool for extracting all of the user's intents from the user's newest message, using chat history as additional context."
    )

    none_check_ = ['user_id_', 'session_id_']

    def __init__(self):
        super().__init__()
        self.user_id_ = None
        self.session_id_ = None
        self.external_chat_history_str_ = None
        self.user_import_displays_ = {}
        self.success_extracting_ = False

    def setExternalChatHistoryStr(self, external_chat_history=None):
        self.external_chat_history_str_ = external_chat_history

    def setUserAndSessionId(self, user_id, session_id):
        self.user_id_ = user_id
        self.session_id_ = session_id

    def getUserAndSessionId(self) -> Tuple[Optional[str], Optional[str]]:
        return (self.user_id_, self.session_id_)

    def getSuccessExtracting(self):
        return self.success_extracting_
        
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

    def _run(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Tool for extracting all of the user's intents from his most recent message, using chat history as additional context."""
        success_extracting, output_string = await self.processUserInput(human_message=query)
        self.success_extracting_ = success_extracting
        return output_string

    async def _arun(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Tool for extracting all of the user's intents from his most recent message, using chat history as additional context."""
        success_extracting, output_string = await self.processUserInput(human_message=query)
        self.success_extracting_ = success_extracting
        return output_string
]
    @none_check(keyword_types={'newest_msg'})
    @type_check(keyword_types={'newest_msg': str})
    def preprocessUserInput(self, newest_msg:str) -> str:
        user_id, session_id = self.getUserAndSessionId()
        chat_history = self.external_chat_history_str_ if self.external_chat_history_str_ is not None else ""

        first_choice_text = ""
        
#       if is_single_sentence(newest_msg):
#           first_choice_text = f"""        {{
#               "user_id": "{user_id}",
#               "session_id": "{session_id}",
#               "extracted_intents": [
#                   {{"raw_intent": "{newest_msg}"}}
#               ]
#           }}"""
#           return first_choice_text;

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
            logger.error(f"preprocessUserInput: An error occurred: {e}")
            
        return ""

    #--------------------------------------------------------------
    #    processUserInput()
    #    - Returns bool based purely on whether any INTENTS were successfully IMPORTED.
    #    IOW, there is still possibly an OUTPUT DISPLAY STRING returned, even though no
    #    intents were necessarily imported. (Though normally an intent would always be
    #    imported, given that the default intent is "chitchat.")
    @none_check(keyword_types={'human_message'})
    @type_check(keyword_types={'human_message': str})
    async def processUserInput(self, human_message):
        output_str = ""
        intents_were_imported = False
        user_id, session_id = self.getUserAndSessionId()
        preprocessedJSON = self.preprocessUserInput(newest_msg=human_message)
        if not preprocessedJSON:
            logger.error("IntentAgent: Failed to preprocess user input: ", human_message)
            return False, f"IntentAgent: Failed to preprocess user input: {human_message}"
        print("IntentAgent: PREPROCESSED user input:\n", preprocessedJSON)
        #--------------------------------------------------------------
        new_questions = ""
        new_commands = ""
        new_procedures = ""
        new_goals = ""
        new_chitchat = ""
        
        if not UserIntent.Import(preprocessedJSON):
            logger.error("IntentAgent: UserIntent.Import(preprocessedJSON) failed to import any intents from the above preprocessed json.")
            output_str = f"IntentAgent: UserIntent.Import failed to import intents from preprocessedJSON:\n```json\n{preprocessedJSON}\n```\n"
            return False, output_str
        else:
            def updateStrings(IntentType):
                nonlocal user_id
                if IntentType.NewestImportsCount(user_id) > 0:
                    return IntentType.ListNewestImportsAsString(user_id)
                return ""

            new_questions = updateStrings(UserQuestion)
            new_commands = updateStrings(UserCommand)
            new_procedures = updateStrings(UserProcedure)
            new_goals = updateStrings(UserGoal)
            new_chitchat = updateStrings(UserChitchat)

#            if UserQuestion.NewestImportsCount(user_id) > 0:
#                new_questions = UserQuestion.ListNewestImportsAsString(user_id)
#            if UserCommand.NewestImportsCount(user_id) > 0:
#                new_commands = UserCommand.ListNewestImportsAsString(user_id)
#            if UserProcedure.NewestImportsCount(user_id) > 0:
#                new_procedures = UserProcedure.ListNewestImportsAsString(user_id)
#            if UserGoal.NewestImportsCount(user_id) > 0:
#                new_goals = UserGoal.ListNewestImportsAsString(user_id)
#            if UserChitchat.NewestImportsCount(user_id) > 0:
#                new_chitchat = UserChitchat.ListNewestImportsAsString(user_id)
        
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
                
                return return_str

            output_str = formatIntentsForUser()
            self.setRecentImportUserDisplayText(user_id, output_str)
        return intents_were_imported, output_str

# ---------------------------------------------------------------------
# Unit tests go here.
if __name__ == "__main__":
    user_id = "test_user"
    session_id = "test_session"
    chat_history_str = "blah blah blah blah blah"
    
    tool = IntentExtractorTool()

    print("This first try SHOULD throw an exception.")
    print("First try...\n")
    try:
        tool.verifySpecificMembersNotNone()
    except InitializationError as e:
        print(f"Initialization error: {e}")
    
    print("This second try, however, should succeed WITHOUT throwing an exception.")
    print("Second try...\n")
    tool.setUserAndSessionId(user_id, session_id)
    try:
        tool.verifySpecificMembersNotNone()
    except InitializationError as e:
        print(f"Initialization error: {e}")
