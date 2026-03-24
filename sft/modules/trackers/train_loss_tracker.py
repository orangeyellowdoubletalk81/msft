import torch
import torch.distributed as dist

from typing import Union

from modules.trackers.tracker import Tracker
from modules.events.event_data import EventTypes, TrainLossLogEventData, LossBackwardEndEventData

class TrainLossTracker(Tracker[Union[TrainLossLogEventData, LossBackwardEndEventData]]):
    def __init__(self, rank):
        super().__init__(rank)
        self.__loss_tensor = torch.zeros(2).to(self.rank)

    def handle_event(self, event_data: Union[TrainLossLogEventData, LossBackwardEndEventData]):
        event_type = event_data.event_type
        if event_type is EventTypes.TRAIN_LOSS_LOG:
            self._on_train_loss_log(event_data)
            return
        
        if event_type is EventTypes.LOSS_BACKWARD_END:
            self._on_lossbackward(event_data)

    def _on_train_loss_log(self, event_data: TrainLossLogEventData):
        progress_bar = event_data.progress_bar
        avg_loss = self._compute_avg_loss()
        self._current_value = {
            "loss": avg_loss
        }
        if self.rank == 0:
            progress_bar.set_postfix({"Loss": avg_loss})
        self.__loss_tensor = torch.zeros(2).to(self.rank)

    def _on_lossbackward(self, event_data: LossBackwardEndEventData):
        self.__loss_tensor[0] += event_data.loss_value
        self.__loss_tensor[1] += event_data.batch_size

    def _compute_avg_loss(self):
        loss_tensor = self.__loss_tensor
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return (loss_tensor[0] / loss_tensor[1]).item()
