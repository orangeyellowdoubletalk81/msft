from abc import *

from modules.events.event import EventHandler, EventData_T

class Tracker(EventHandler[EventData_T]):
    def __init__(self, rank):
        self.rank = rank
        self._current_value = None
    
    @property
    def current_value(self) -> dict:
        return self._current_value
