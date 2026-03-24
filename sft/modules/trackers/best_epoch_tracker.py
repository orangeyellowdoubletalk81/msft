import torch
import torch.distributed as dist

from modules.trackers.tracker import Tracker
from modules.events.event_data import EventTypes, QuantumEndEventData


class BestEpochTracker(Tracker[QuantumEndEventData]):
    def __init__(self, rank):
        super().__init__(rank)

    
    def handle_event(self, event_data: QuantumEndEventData):
        event_type = event_data.event_type
        if event_type is EventTypes.QUANTUM_END:
            self._on_quantum_end(event_data)
            return
    
    def _on_quantum_end(self, event_data: QuantumEndEventData):
        self._current_value = {
            "best_accuracy": {
                "epoch": event_data.best_epoch,
                "accuracy": event_data.best_accuracy,
                "subdataset_accuracy": event_data.subdataset_accuracy
            }
        }
