import torch
import torch.distributed as dist

from typing import Union
from collections import Counter

from modules.trackers.tracker import Tracker
from modules.events.event_data import EventTypes, GradientUpdateEndEventData, BatchStartEventData

class CategoryCountTracker(Tracker[Union[GradientUpdateEndEventData, BatchStartEventData]]):
    def __init__(self, rank):
        super().__init__(rank)
        self.__category_counter = Counter()

    def handle_event(self, event_data: Union[GradientUpdateEndEventData, BatchStartEventData]):
        event_type = event_data.event_type
        if event_type is EventTypes.GRADIENT_UPDATE_END:
            self._on_gradient_update(event_data)
            return
        
        if event_type is EventTypes.BATCH_START:
            self._on_batch_start(event_data)
            return

    def _on_gradient_update(self, event_data: GradientUpdateEndEventData):
        world_size = event_data.world_size
        category_counts = self._gather(world_size)
        self._current_value = {
            "category_counts": category_counts
        }
        self.__category_counter = Counter()

    def _on_batch_start(self, event_data: BatchStartEventData):
        categories = event_data.batch_categories
        self.__category_counter.update(categories)

    def _gather(self, world_size):
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, dict(self.__category_counter))

        merged = Counter()
        for d in gathered:
            merged.update(d or {})
        
        return dict(merged)
