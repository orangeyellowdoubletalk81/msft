from abc import *
from typing import TypeVar, Generic

from modules.events.event_data import EventTypes, EventData

EventData_T = TypeVar("EventData_T", bound=EventData)


class EventHandler(Generic[EventData_T], metaclass=ABCMeta):
    @abstractmethod
    def handle_event(self, event_data: EventData_T):
        pass


class Event(Generic[EventData_T]):
    def __init__(self, event_type: EventTypes, handlers: list[EventHandler]):
        self.__event_type = event_type
        self.__handlers = handlers
    
    def add_handler(self, handler: EventHandler):
        self.__handlers.append(handler)

    def add_handlers(self, handlers: list[EventHandler]):
        for handler in handlers:
            self.add_handler(handler)
    
    def remove_handler(self, handler: EventHandler):
        try:
            self.__handlers.remove(handler)
        except ValueError:
            raise ValueError("없는 handler를 삭제하려고 했습니다.")

    def remove_handlers(self, handlers: list[EventHandler]):
        for handler in handlers:
            self.remove_handler(handler)
    
    def invoke(self, event_data: EventData_T):
        event_data.event_type = self.__event_type
        for handler in self.__handlers:
            handler.handle_event(event_data)

    @property
    def event_type(self):
        return self.__event_type
