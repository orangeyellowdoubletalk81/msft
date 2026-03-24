import torch
import torch.distributed as dist

from typing import Union
from collections import Counter

from modules.trackers.tracker import Tracker
from modules.events.event_data import EventTypes, QuantumStartEventData, QuantumEndEventData


class DatasetCountTracker(Tracker[QuantumStartEventData]):
    def __init__(self, rank):
        super().__init__(rank)
        self.__dataset_counter = Counter()

    
    def handle_event(self, event_data: QuantumStartEventData):
        event_type = event_data.event_type
        if event_type is EventTypes.QUANTUM_START:
            self._on_quantum_start(event_data)
            return
    
    def _on_quantum_start(self, event_data: QuantumStartEventData):
        self._current_value = {
            "category_counts": {
                "train": event_data.dataloader.train_subdataset_count,
                "test": event_data.dataloader.test_subdataset_count
            }
        }
