import torch
import torch.distributed as dist

from modules.trackers.tracker import Tracker
from modules.events.event_data import EventTypes, GradientUpdateEndEventData

class LRTracker(Tracker[GradientUpdateEndEventData]):
    def __init__(self, rank):
        super().__init__(rank)

    def handle_event(self, event_data: GradientUpdateEndEventData):
        event_type = event_data.event_type
        if event_type is EventTypes.GRADIENT_UPDATE_END:
            self._on_gradient_update(event_data)
            return

    def _on_gradient_update(self, event_data: GradientUpdateEndEventData):
        scheduler = event_data.scheduler
        lr = scheduler.get_last_lr()[0]
        if lr is None:
            lr = -100
        self._current_value = {
            "learning_rate": lr
        }
