import torch
import torch.distributed as dist

from dataclasses import asdict

from modules.trackers.tracker import Tracker
from modules.events.event_data import EventTypes, BatchStartEventData

class SettingsTracker(Tracker[BatchStartEventData]):
    def __init__(self, rank):
        super().__init__(rank)

    def handle_event(self, event_data: BatchStartEventData):
        event_type = event_data.event_type
        if event_type is EventTypes.BATCH_START:
            self._on_batch_start(event_data)
            return

    def _on_batch_start(self, event_data: BatchStartEventData):
        self._current_value = {
            "train_settings": asdict(event_data.train_settings)
        }
