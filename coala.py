
from abc import ABC, abstractmethod

class BaseClass(ABC):
    @abstractmethod
    def method_a(self):
        pass

    @abstractmethod
    def method_b(self):
        pass

class InterfaceOne(BaseClass):
    def method_a(self):
        # Specific implementation for InterfaceOne
        print("InterfaceOne method A")

    def method_b(self):
        # Calls BaseClass method_b if needed, or provide its own implementation
        super().method_b()

class InterfaceTwo(BaseClass):
    def method_a(self):
        # Calls BaseClass method_a if needed, or provide its own implementation
        super().method_a()
    
    def method_b(self):
        # Specific implementation for InterfaceTwo
        print("InterfaceTwo method B")

class ImplementingClass(BaseClass):
    def __init__(self):
        self.one = InterfaceOne()
        self.two = InterfaceTwo()
        
    def method_a(self):
        # Delegate to InterfaceOne
        self.one.method_a()

    def method_b(self):
        # Delegate to InterfaceTwo
        self.two.method_b()

# You can now use ImplementingClass as BaseClass
def process_base(base: BaseClass):
    base.method_a()
    base.method_b()

obj = ImplementingClass()
process_base(obj)  # obj is treated as an instance of BaseClass
