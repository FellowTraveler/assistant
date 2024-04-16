import json
import uuid
import random
import string
import re
from datetime import datetime
from dateutil import parser
from typing import List, Dict

def is_similar(str1, str2, threshold=0.8):
    """
    Check if two strings are similar based on simple containment or overlap.

    Args:
        str1 (str): First string.
        str2 (str): Second string.
        threshold (float): Threshold for determining similarity based on overlap.

    Returns:
        bool: True if strings are considered similar, False otherwise.
    """
    str1, str2 = str1.lower(), str2.lower()
    # NOTE: I commented this out because what if one string is 10 bytes long and
    # is contained inside the other string at 10,000 bytes long? It shouldn't
    # return True in that case, should it? And we'd definitely want to preserve
    # the longer one in that case, not the shorter one.
#    if str1 in str2 or str2 in str1:
#        return True
    overlap = len(set(str1.split()) & set(str2.split())) / max(len(str1.split()), len(str2.split()))
    return overlap >= threshold

def are_similar(first, second, threshold=0.75):
    return is_similar(first['rephrased_intent'], second["rephrased_intent"], threshold)


class UserIntent:
    subclasses = []  # Registry of subclasses
    intents = []  # This is overridden in each subclass
    newest_imports = [] # This is overridden in each subclass
    intent_type = 'unknown'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        UserIntent.subclasses.append(cls)

    @staticmethod
    def validate_custom_id(custom_id):
        # Regular expression pattern to match the custom ID format
        # Starts with an alphabetical character, followed by 4 alphanumeric characters
        pattern = re.compile(f'^[{string.ascii_letters}][{string.ascii_letters}{string.digits}]{{4}}$')
        # Use the fullmatch method to check if the entire string matches the pattern
        return bool(pattern.fullmatch(custom_id))
        
    @staticmethod
    def validate_abbrev_id(abbrev_id):
        """
        Validates an abbreviated ID format to ensure it could be a valid start for full IDs.
        A valid abbreviated ID can be the initial letter alone or the initial letter followed by up to four alphanumeric characters.

        Parameters:
            abbrev_id (str): The abbreviated ID to validate.

        Returns:
            bool: True if the ID is a valid start for a full custom ID, False otherwise.

        Criteria:
            - Starts with an alphabetical character.
            - Followed by up to four alphanumeric characters.
            - Total length: 1 to 5 characters.
        """
        # Regular expression pattern to match a valid abbreviated custom ID format
        pattern = re.compile(f'^[{string.ascii_letters}][{string.ascii_letters}{string.digits}]{{0,4}}$')
        # Use the fullmatch method to check if the entire string matches the pattern
        return bool(pattern.fullmatch(abbrev_id))

    @classmethod
    def _generate_id(cls):
        # Start with a random alphabetical character
        start_char = random.choice(string.ascii_letters)
        # Generate a unique identifier using uuid4, remove dashes, and take a slice
        unique_part = str(uuid.uuid4()).replace("-", "")[:4]  # Adjust length as needed
        # Combine and return
        return start_char + unique_part

    @classmethod
    def _current_datetime(cls):
        return datetime.now()
#        return datetime.now().isoformat()

    @classmethod
    def Import(cls, json_data: str):
        thereWereImports = False
        for subclass in cls.subclasses:
            if subclass.import_intents(json_data):
                thereWereImports = True
        if thereWereImports:
            print("Imported intents.")
        return thereWereImports

    def __init__(self, rephrased_intent: str, relevant_info: str):
        self.rephrased_intent = rephrased_intent
        self.relevant_info = relevant_info

    @classmethod
    def ListIntentsAsString(cls, user_id: str, session_id=None):
        # Filter intents first by user_id and then by session_id if it is provided
        if session_id is None:
            relevant_intents = [intent for intent in cls.intents if intent["user_id"] == user_id]
        else:
            relevant_intents = [intent for intent in cls.intents if intent["user_id"] == user_id and intent["session_id"] == session_id]
        
        # Generate a formatted string for each relevant intent
        return "\n".join(f"-- {intent['rephrased_intent']} *({intent['intent_type']} ID {intent['id']})*" for intent in relevant_intents)

    @classmethod
    def ListAllTypesIntentsAsString(cls, user_id: str, session_id=None):
        result_str = ""
        for subclass in cls.subclasses:
            count = subclass.IntentsCount(user_id=user_id, session_id=session_id)
            if count > 0:
                intents_str = subclass.ListIntentsAsString(user_id=user_id, session_id=session_id)
                result_str += f"*{subclass.intent_type}:*\n"
                result_str += intents_str
                result_str += "\n\n"
        return result_str


#    @classmethod
#    def ListNewestImportsAsString(cls):
#        output_lines = []
#        for intent in cls.newest_imports:
#            # Add the intent line
#            output_lines.append(f"-- {intent['rephrased_intent']} *({intent['intent_type']} ID {intent['id']})*")
#        return "\n".join(output_lines)

    @classmethod
    def ListNewestImportsAsString(cls, user_id: str, session_id=None):
        # Filter newest_imports first by user_id and then by session_id if it is provided
        if session_id is None:
            relevant_intents = [intent for intent in cls.newest_imports if intent["user_id"] == user_id]
        else:
            relevant_intents = [intent for intent in cls.newest_imports if intent["user_id"] == user_id and intent["session_id"] == session_id]
        
        # Generate a formatted string for each relevant intent
        return "\n".join(f"-- {intent['rephrased_intent']} *({intent['intent_type']} ID {intent['id']})*" for intent in relevant_intents)

    @classmethod
    def ListAllTypesNewestImportsAsString(cls, user_id: str, session_id=None):
        result_str = ""
        for subclass in cls.subclasses:
            count = subclass.NewestImportsCount(user_id=user_id, session_id=session_id)
            if count > 0:
                intents_str = subclass.ListNewestImportsAsString(user_id=user_id, session_id=session_id)
                result_str += f"*{subclass.intent_type}:*\n"
                result_str += intents_str
                result_str += "\n\n"
        return result_str

    @classmethod
    def NewestImportsCount(cls, user_id: str, session_id=None):
#       return len(cls.newest_imports)
        if session_id is None:
            relevant_intents = [intent for intent in cls.newest_imports if intent["user_id"] == user_id]
        else:
            relevant_intents = [intent for intent in cls.newest_imports if intent["user_id"] == user_id and intent["session_id"] == session_id]
        return len(relevant_intents)
        
    @classmethod
    def IntentsCount(cls, user_id: str, session_id=None):
        if session_id is None:
            relevant_intents = [intent for intent in cls.intents if intent["user_id"] == user_id]
        else:
            relevant_intents = [intent for intent in cls.intents if intent["user_id"] == user_id and intent["session_id"] == session_id]
        return len(relevant_intents)

#    @classmethod
#    def GetIntents(cls):
#        # Directly return the list of intent dictionaries
#        return cls.intents

    @classmethod
    def GetIntents(cls, user_id: str, session_id=None):
        """
        Retrieves intents filtered by user_id and optionally by session_id.

        Parameters:
            user_id (str): The user ID to filter the intents.
            session_id (Optional[str]): The session ID to further filter the intents, optional.

        Returns:
            list: A list of relevant intent dictionaries based on the given filters.
        """
        relevant_intents = []
        if session_id is None:
            relevant_intents = [intent for intent in cls.intents if intent["user_id"] == user_id]
        else:
            relevant_intents = [intent for intent in cls.intents if intent["user_id"] == user_id and intent["session_id"] == session_id]
        return relevant_intents

    @classmethod
    def GetAllTypesIntents(cls, user_id: str, session_id=None):
        relevant_intents = []
        for subclass in cls.subclasses:
            relevant_intents.extend(subclass.GetIntents(user_id=user_id, session_id=session_id))
        return relevant_intents

#    @classmethod
#    def GetNewestImports(cls):
#        # Directly return the list of intent dictionaries
#        return cls.newest_imports

    @classmethod
    def GetNewestImports(cls, user_id: str, session_id=None):
        """
        Retrieves the newest imports filtered by user_id and optionally by session_id.

        Parameters:
            user_id (str): The user ID to filter the imports.
            session_id (Optional[str]): The session ID to further filter the imports, optional.

        Returns:
            list: A list of relevant newest import dictionaries based on the given filters.
        """
        relevant_imports = []
        if session_id is None:
            relevant_imports = [import_ for import_ in cls.newest_imports if import_["user_id"] == user_id]
        else:
            relevant_imports = [import_ for import_ in cls.newest_imports if import_["user_id"] == user_id and import_["session_id"] == session_id]
        return relevant_imports

    @classmethod
    def GetAllTypesNewestImports(cls, user_id: str, session_id=None):
        relevant_intents = []
        for subclass in cls.subclasses:
            relevant_intents.extend(subclass.GetNewestImports(user_id=user_id, session_id=session_id))
        return relevant_intents

    @classmethod
    def GetIntent(cls, intent_id):
        for subclass in cls.subclasses:
            intent = subclass.get_intent(intent_id)
            if intent is not None:
                return intent
        return None

    @classmethod
    def GetIntentAbbrev(cls, abbrev_intent_id: str) -> List[Dict[str, str]]:
        """
        Retrieves all matching intents by abbreviated intent_id across all subclasses.
        
        Parameters:
            abbrev_intent_id (str): The abbreviated ID of the intent(s) to retrieve.
        
        Returns:
            List[Dict[str, str]]: A list of all intent objects across subclasses that match the abbreviated ID.
        """
        all_intents = []
        for subclass in cls.subclasses:
            intents = subclass.get_intent_abbrev(abbrev_intent_id)
            if intents:
                all_intents.extend(intents)
        return all_intents

    @classmethod
    def DeleteIntent(cls, intent_id):
        for subclass in cls.subclasses:
            if subclass.delete_intent(intent_id):
                return True
        return False

    @classmethod
    def get_intent_abbrev(cls, abbrev_intent_id: str) -> List[Dict[str, str]]:
        """
        Retrieves all matching intents by abbreviated intent_id.
        
        Parameters:
            abbrev_intent_id (str): The abbreviated ID of the intent(s) to retrieve.
        
        Returns:
            List[Dict[str, str]]: A list of intent objects that match the abbreviated ID, if found; otherwise, an empty list.
        """
        # Collect all intents where the id starts with the abbreviated intent_id
        matching_intents = [
            intent for intent in cls.intents if intent["id"].startswith(abbrev_intent_id)
        ]
        return matching_intents
        
    @classmethod
    def get_intent(cls, intent_id):
        """
        Retrieves a single intent by its ID.
        
        Parameters:
            intent_id (int): The ID of the intent to retrieve.
        
        Returns:
            dict: The intent dictionary if found, otherwise None.
        """
        # Use next() to return the first matching intent or None if no match is found
        return next((intent for intent in cls.intents if intent["id"] == intent_id), None)

    @classmethod
    def delete_intent(cls, intent_id):
        """
        Deletes an intent from the intents and newest_imports lists based on the given intent_id.

        Parameters:
            intent_id (int): The ID of the intent to delete.

        Returns:
            bool: True if an intent was deleted, False otherwise.
        """
        original_intents_count = len(cls.intents)
        original_imports_count = len(cls.newest_imports)

        # Filter out the intent to delete
        cls.intents = [intent for intent in cls.intents if intent["id"] != intent_id]
        cls.newest_imports = [intent for intent in cls.newest_imports if intent["id"] != intent_id]

        # Determine if any intents were deleted
        intents_deleted = len(cls.intents) != original_intents_count
        imports_deleted = len(cls.newest_imports) != original_imports_count

        # Save changes to file
        cls.SaveToIntentsFile(f"intents_{cls.intent_type}.json")

        # Return True if any deletions occurred, otherwise False
        return intents_deleted or imports_deleted

    @classmethod
    def import_intents(cls, json_data: str):
        thereWereImports = False
        cls.newest_imports.clear()
        
        data = json.loads(json_data)
        for intent in data["extracted_intents"]:
            if intent["intent_type"] == cls.intent_type:
                intent["user_id"] = data.get("user_id")
                intent["session_id"] = data.get("session_id")
                if cls._add_intent(intent):
                    thereWereImports = True

        if thereWereImports:
            cls.SaveToIntentsFile(f"intents_{cls.intent_type}.json")
        return thereWereImports

    @classmethod
    def _add_intent(cls, intent_data):
        existing_intent = next((i for i in cls.intents if are_similar(i, intent_data)), None)

        thereWereChanges = False
        
        if "suggested_intent_type" in intent_data:
            print(f'Log: suggested_intent_type is {intent_data.get("suggested_intent_type")}')

        if existing_intent:
            # Assume no changes initially
            # WRONG. If there's an existing one then we're at least updating its
            # session_id and last modified date. Why? Because if the user JUST
            # gave us a new intent that matches one we've had before, that doesn't
            # change the fact that the one that JUST came in is still "new". And
            # the fact that older copies exist just reinforces its importance, if
            # anything. We don't want to leave it looking older and less relevant
            # (when its not...) so we HAVE to update it at least to reflect its
            # immediacy and also to make sure that it's labeled with the current
            # session_id and not some older one, because it IS important to the user
            # in THIS current session that's happening right now, which matters,
            # compared to some other session that happened in the past.
            # Based on that we might want to even store a history of session IDs
            # related to this intent because there might be context in those that's
            # relevant to this session. In fact, the things that we'd want to know
            # are probably the things that a person would have "learned" over various
            # episodes in his life. So by finding useful context from some previous
            # episode, and by having that info improve the robot's current session ID
            # to solve or attain whatever it's trying to do, shows that there is
            # valuable knowledge to be gained by noticing when the same or similar
            # intents keep popping up over different session IDs. Their very existence
            # is an indicator that there are lessons to be learned from them which
            # can improve the outcome of future sessions. Future episodes. By definition
            # when you have a solution to a problem or need that repeatedly occurs
            # that solution has a value multiplier effect. And why should the robot
            # need to learn the same lesson over and over again, running into the
            # same situation in one episode that it's already have to deal with or
            # learn to solve in previous episodes?
            #
            # THEREFORE WHEN THESE PATTERNS ARE OBSERVED, THEY MUST BE PRESERVED
            # AS GENERAL KNOWLEDGE OR BELIEF, WHETHER EMPIRICAL OR AXIOMATIC,
            # OR CALL IT A GUT FEELING IF YOU WANT. Part of this is that whenever
            # lessons are particularly painful or pleasurable, based on some heuristic,
            # the robot must remember them more importantly than other learnings/beliefs.
            # No one can ever know the full truth, all we can do is make assumptions
            # about how things actualy work, and then learn lessons from our experiences
            # which cause us to revise our theories/assumptions/beliefs/prejudices
            # and accordingly our future decisions. Sometimes those experiences are
            # just our observations over time, accompanied by deep thought. But also
            # some experiences cause us greater benefit or greater pain, they are
            # more visceral than other experiences. Their outcomes are more important
            # to us. Their outcomes force stronger, more automatic reactions. When
            # a person is gripped by fear for their life, powerful emotions are
            # activated, and each emotion tends to trigger specific behaviors. The
            # powerful FEELING of the motion is precisely what motivates those
            # behaviors, just the same as the powerful feeling of being burned
            # causes a corresponding powerful urge to move away from the source of
            # heat as quickly as possible. Moving away from being burned is MORE
            # IMPORTANT than moving away from many other things. Its immediacy
            # and criticality, the great price you will pay if you ignore the risk
            # of that outcome, in terms of suffering and other very real costs,
            # must be factored in, yet a person doesn't logically think this through
            # when they are being burned. No, the pain is so awful that it motivates
            # the correct behaviors with the fastest possible reaction time, in fact
            # it basically forces those behaviors so that they happen as quickly as
            # possible, much faster than some deep logical throught process could
            # achieve.
            # And in the future you will remember that burn much more vividly than
            # other memories, and it will influence your future behavior around fires
            # much more powerfully than the influence that various other learnings
            # might exercise over you and your future behavior.
            # The robot, when it has an experience that causes some GREAT LOSS or some
            # WINDFALL GAIN, must be able to learn a lesson from that so it doesn't
            # have to repeatedly learn it over and over again forever. It must
            # incorporate that lesson so that it never has to learn it again.
            # That is what pleasure and pain are, to humans. The robot doesn't have
            # to actually SUFFER real pain, of course, but rather, you must set
            # the priority level of the pain much higher. "Painful" things simply
            # must be prioritized higher. They must be avoided more, learned about
            # more, remembered more, etc. And emotions are a similar thing, except
            # they trigger based on potential damage (percentage chance of damage)
            # and on potential gain (percentage chance of gain). Of course it takes
            # intelligence to predict these kinds of percentage chances, which is just
            # another reason that it's important to be able to learn lessons about
            # their true risk / potential impact. And is also a good reason why it's
            # important, for certain things that have a greater impact, to have
            # emotional (automated) responses that force important behaviors without
            # relying on learning. More on instinct. For even animals have powerful
            # emotional reactions to certain things, and whether or not they understand
            # or learn about those things intellectually, those reactions still
            # benefit them regarding the potential impacts they face.
            # Of course what I want for my robot is both, the same as I have: the ability
            # to learn lessons from experiences, but also quicker, more automated responses
            # in times of great risk, where the potential outcome could have greater
            # impact and where time is of the essence.
            # The emotions also do not trigger with perfect accuracy. Rather, they have
            # evolved based on Pareto distributions and percentage chances of impact.
            # The triggers they are based on may not provide perfect information, but
            # they were simply more effective over evolutionary time, versus not having
            # those triggers.
            # One benefit with a robot is simply that it is theoretically able, upon
            # learning a lesson intellectually, to then adjust its emotional (more
            # automated) responses to react more accurately and effectively by
            # triggering on better cues it has learned, or by automating better behaviors
            # than the ones we received from the accidents of evolution. Of course,
            # our emotions are actually extremely well-tuned over millenia, but
            # my basic point remains the same: We can only tune our built-in emotions
            # as a species, via natural selection. But a robot could individually
            # refine its emotions to make them better tuned over time by adjusting their
            # reactions to its learnings.
                            
            existing_intent["session_id"] = intent_data["session_id"]
            existing_intent["modified_datetime"] = cls._current_datetime()
            thereWereChanges = True

            new_info = intent_data["relevant_info"]
            old_info = existing_intent["relevant_info"]
            # Check if the new information is different and update accordingly
            if new_info != old_info:
                if is_similar(new_info, old_info):
                    # If one info is a substring of the other, keep the more informative one
                    existing_intent["relevant_info"] = max(new_info, old_info, key=len)
                else:
                    # Concatenate new information if it's different and not a substring
                    existing_intent["relevant_info"] += (" / " + new_info)
                
            if existing_intent not in cls.newest_imports:
                cls.newest_imports.append(existing_intent)
        else:
            # Add a new intent since no existing match was found
            intent_id = cls._generate_id()
            creation_datetime = cls._current_datetime()
            new_intent = {
                "id": intent_id,
                "user_id": intent_data.get("user_id"),
                "session_id": intent_data.get("session_id"),
                "raw_intent": intent_data.get("raw_intent"),
                "rephrased_intent": intent_data.get("rephrased_intent"),
                "relevant_info": intent_data.get("relevant_info"),
                "intent_type": intent_data.get("intent_type"),
                "creation_datetime": creation_datetime,
                "modified_datetime": creation_datetime
            }
            cls.intents.append(new_intent)
            cls.newest_imports.append(new_intent)
            thereWereChanges = True

        return thereWereChanges

    @classmethod
    def SaveAllIntents(cls):
        for subclass in cls.subclasses:
            subclass.SaveToIntentsFile()
        print("Saved all intents.")

    @classmethod
    def SaveToIntentsFile(cls, filename=None):
        if filename is None:
            filename = f"intents_{cls.intent_type}.json"
        # Convert datetime objects to strings for JSON serialization
        intents_to_save = []
        for intent in cls.intents:
            modified_intent = intent.copy()
            modified_intent["creation_datetime"] = modified_intent["creation_datetime"].isoformat()
            modified_intent["modified_datetime"] = modified_intent["modified_datetime"].isoformat()
            intents_to_save.append(modified_intent)
            
        with open(filename, 'w') as f:
            json.dump(intents_to_save, f, indent=4)

    @classmethod
    def LoadAllIntents(cls):
        for subclass in cls.subclasses:
            subclass.LoadFromIntentsFile(f"intents_{subclass.intent_type}.json")
        print("Loaded all intents.")

    @classmethod
    def LoadFromIntentsFile(cls, filename):
        loaded_intents = None
        try:
            with open(filename, 'r') as f:
                loaded_intents = json.load(f)
                # Convert datetime strings back to datetime objects
                for intent in loaded_intents:
                    intent["creation_datetime"] = parser.parse(intent["creation_datetime"])
                    intent["modified_datetime"] = parser.parse(intent["modified_datetime"])

                cls.intents = loaded_intents

        except FileNotFoundError:
            # Log message or initialize to empty list if file not found
            cls.intents = []
        except json.JSONDecodeError:
            # Handle empty or invalid JSON file
            cls.intents = []
        
class UserQuestion(UserIntent):
    intents = []  # Separate list for each subclass
    newest_imports = [] # Separate list for each subclass
    intent_type = 'question'

class UserProcedure(UserIntent):
    intents = []  # Separate list for each subclass
    newest_imports = [] # Separate list for each subclass
    intent_type = 'procedure'

class UserGoal(UserIntent):
    intents = []
    newest_imports = [] # Separate list for each subclass
    intent_type = 'goal'
        
class UserCommand(UserIntent):
    intents = []  # Separate list for each subclass
    newest_imports = [] # Separate list for each subclass
    intent_type = 'command'

class UserChitchat(UserIntent):
    intents = []
    newest_imports = [] # Separate list for each subclass
    intent_type = 'chitchat'

