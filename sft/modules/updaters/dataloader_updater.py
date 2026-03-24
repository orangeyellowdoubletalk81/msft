import torch.distributed as dist

from utils.dataloader_helper import DataLoader

from modules.events.event import EventHandler, EventData_T
from modules.events.event_data import EventTypes, GradientUpdateEndEventData, EvalEndEventData, QuantumStartEventData

class DataLoaderUpdater(EventHandler[EventData_T]):
    def __init__(self, rank, dataloader, filter_function):
        self._rank = rank
        self._original_dataloader = dataloader
        self._dataloader = dataloader
        self._filter_function = filter_function
        self._batch_updated_flag = False
        self._is_updated = False

    def _call_filter_function(self, examples, event_data):
        result = self._filter_function(examples, event_data)
        if all(result):
            self._batch_updated_flag = False
        else:
            self._batch_updated_flag = True
        
        if self._batch_updated_flag:
            self._is_updated = True
        return result

    def gather_flags(self, world_size):
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, self._is_updated)
        self._is_updated = any(gathered)
        if self._rank == 0:
            if self._is_updated:
                print("The dataset has been udpated!")
            else:
                print("The dataset is not updated...")
    
    def reset_flags(self):
        self._batch_updated_flag = False
        self._is_updated = False

    @property
    def is_updated(self):
        return self._is_updated

    @property
    def dataloader(self):
        return self._dataloader

    @property
    def original_dataloader(self):
        return self._original_dataloader


class DataLoaderUpdaterOnGradientUpdate(DataLoaderUpdater[GradientUpdateEndEventData]):
    def __init__(self, rank, dataloader, filter_function):
        super().__init__(rank, dataloader, filter_function)
    
    def handle_event(self, event_data: GradientUpdateEndEventData):
        event_type = event_data.event_type
        if event_type is EventTypes.GRADIENT_UPDATE_END:
            self._on_gradient_update(event_data)
            return
    
    def _on_gradient_update(self, event_data: GradientUpdateEndEventData):
        self.reset_flags()
        self._dataloader = DataLoader.from_dataloader(self._original_dataloader)
        self._dataloader.filter_dataset(self._rank, event_data.world_size, lambda examples: self._call_filter_function(examples, event_data))
        self.gather_flags(event_data.world_size)


class DataLoaderUpdaterOnEvaluationEnd(DataLoaderUpdater[EvalEndEventData]):
    def __init__(self, rank, dataloader, filter_function):
        super().__init__(rank, dataloader, filter_function)
    
    def handle_event(self, event_data: GradientUpdateEndEventData):
        event_type = event_data.event_type
        if event_type is EventTypes.EVAL_END:
            self._on_evaluation(event_data)
            return
    
    def _on_evaluation(self, event_data: GradientUpdateEndEventData):
        self.reset_flags()
        self._dataloader = DataLoader.from_dataloader(self._original_dataloader)
        self._dataloader.filter_dataset(self._rank, event_data.world_size, lambda examples: self._call_filter_function(examples, event_data))
        self.gather_flags(event_data.world_size)


class DataLoaderUpdaterOnQuantumStart(DataLoaderUpdater[QuantumStartEventData]):
    def __init__(self, rank, dataloader, filter_function):
        super().__init__(rank, dataloader, filter_function)
    
    def handle_event(self, event_data: QuantumStartEventData):
        event_type = event_data.event_type
        if event_type is EventTypes.QUANTUM_START:
            self._on_quantum_start(event_data)
            return
    
    def _on_quantum_start(self, event_data: QuantumStartEventData):
        self.reset_flags()
        self._dataloader = DataLoader.from_dataloader(self._original_dataloader)
        self._dataloader.filter_dataset(self._rank, event_data.world_size, lambda examples: self._call_filter_function(examples, event_data))
        self.gather_flags(event_data.world_size)


def test_filter_function(examples, event_data):
    return [category == "allenai/ai2_arc/ARC-Easy" for category in examples["category"]]
