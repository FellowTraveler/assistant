import logging
logger = logging.getLogger(__name__)

class InitializationError(Exception):
    """Exception raised for errors in the initialization of an object."""
# ---------------------------------------------------------------------
#def none_check(cls):
#    """Decorator used by any class to ensure certain members are initialized before invocation.
#    The decorated class must declare the class member 'none_check_', which must contain the array of names of self.member variables that must be initialized BEFORE some critical invocation occurs.
#    And then BEFORE calling that critical function (the invocation, sometime after construction), the programmer must call 'verifySpecificMembersNotNone' first, to make sure the right members are initialized BEFORE calling that critical function."""
#
#    if not hasattr(cls, 'none_check_'):
#        logger.error(f"Missing required class property: 'none_check_' in class {cls.__name__}")
#        raise InitializationError(f"Missing required class property: 'none_check_' in class {cls.__name__}")
#
#    def verifySpecificMembersNotNone(self):
#        missing_fields = [var for var in self.__class__.none_check_ if getattr(self, var, None) is None]
#        if missing_fields:
#            missing_fields_str = ", ".join(missing_fields)
#            logger.error(f"Missing initialization for fields: {missing_fields_str} in class {self.__class__.__name__}")
#            raise InitializationError(f"Missing initialization for fields: {missing_fields_str} in class {self.__class__.__name__}")
#    cls.verifySpecificMembersNotNone = verifySpecificMembersNotNone
#    return cls

def none_check(positional_types=None, keyword_types=None):
    if positional_types is None:
        positional_types = []
    if keyword_types is None:
        keyword_types = []

    def decorator(obj):
        if inspect.isclass(obj):
            cls = obj
            if not hasattr(cls, 'none_check_'):
                logger.error(f"Missing required class property: 'none_check_' in class {cls.__name__}")
                raise InitializationError(f"Missing required class property: 'none_check_' in class {cls.__name__}")

            def verifySpecificMembersNotNone(self):
                missing_fields = [var for var in cls.none_check_ if getattr(self, var, None) is None]
                if missing_fields:
                    missing_fields_str = ", ".join(missing_fields)
                    logger.error(f"Missing initialization for fields: {missing_fields_str} in class {cls.__name__}")
                    raise InitializationError(f"Missing initialization for fields: {missing_fields_str} in class {cls.__name__}")

            cls.verifySpecificMembersNotNone = verifySpecificMembersNotNone
            return cls
        else:
            func = obj
            @wraps(func)
            def wrapper(*args, **kwargs):
                offset = 1 if 'self' in inspect.signature(func).parameters or 'cls' in inspect.signature(func).parameters else 0
                for index in positional_types:
                    if (index + offset) >= len(args) or args[index + offset] is None:
                        raise ValueError(f"Argument at position {index} must not be None")

                for key in keyword_types:
                    if key not in kwargs or kwargs[key] is None:
                        raise ValueError(f"Keyword argument '{key}' must not be None")

                return func(*args, **kwargs)
            return wrapper

    return decorator
# ---------------------------------------------------------------------
from functools import wraps
import inspect

#Keeping this only temporarily until I verify the code below it works.
#def type_check(positional_types=None, keyword_types=None):
#    def decorator(func):
#        @wraps(func)
#        def wrapper(*args, **kwargs):
#            if positional_types:
#                for pos, type_ in positional_types.items():
#                    if not isinstance(args[pos], type_):
#                        logger.error(f"Argument {pos} must be {type_.__name__}, got {type(args[pos]).__name__} instead.")
#                        raise TypeError(f"Argument {pos} must be {type_.__name__}, got {type(args[pos]).__name__} instead.")
#            if keyword_types:
#                for key, type_ in keyword_types.items():
#                    if key in kwargs and not isinstance(kwargs[key], type_):
#                        logger.error(f"Argument '{key}' must be {type_.__name__}, got {type(kwargs[key]).__name__} instead.")
#                        raise TypeError(f"Argument '{key}' must be {type_.__name__}, got {type(kwargs[key]).__name__} instead.")
#            return func(*args, **kwargs)
#        return wrapper
#    return decorator
# ---------------------------------------------------------------------
def type_check(positional_types=None, keyword_types=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if it's a function or a method within a class (class method, instance method)
            if inspect.isfunction(func) or inspect.ismethod(func):
                func_signature = inspect.signature(func)
                first_param = next(iter(func_signature.parameters))
                
                if first_param in ['self', 'cls'] and not inspect.isfunction(func):
                    # Adjust positional checks for 'self' or 'cls'
                    adjusted_positional_types = {k + 1: v for k, v in (positional_types or {}).items()}
                else:
                    # No adjustment needed for static methods or standalone functions
                    adjusted_positional_types = positional_types
            else:
                adjusted_positional_types = positional_types

            if adjusted_positional_types:
                for pos, type_ in adjusted_positional_types.items():
                    if not isinstance(args[pos], type_):
                        logger.error(f"Argument {pos-1 if pos > 0 else pos} must be {type_.__name__}, got {type(args[pos]).__name__} instead.")
                        raise TypeError(f"Argument {pos-1 if pos > 0 else pos} must be {type_.__name__}, got {type(args[pos]).__name__} instead.")
            if keyword_types:
                for key, type_ in keyword_types.items():
                    if key in kwargs and not isinstance(kwargs[key], type_):
                        logger.error(f"Argument '{key}' must be {type_.__name__}, got {type(kwargs[key]).__name__} instead.")
                        raise TypeError(f"Argument '{key}' must be {type_.__name__}, got {type(kwargs[key]).__name__} instead.")
            return func(*args, **kwargs)
        return wrapper
    return decorator
