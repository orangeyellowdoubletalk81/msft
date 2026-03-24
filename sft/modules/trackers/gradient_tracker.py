import torch
import torch.distributed as dist

from modules.trackers.tracker import Tracker
from modules.events.event_data import EventTypes, GradientUpdateEndEventData

class GradientNormTracker(Tracker[GradientUpdateEndEventData]):
    def __init__(self, model, rank):
        super().__init__(rank)
        self.__model = model

    def handle_event(self, event_data: GradientUpdateEndEventData):
        event_type = event_data.event_type
        if event_type is EventTypes.GRADIENT_UPDATE_END:
            self._on_gradient_update(event_data)
            return

    def _on_gradient_update(self, event_data: GradientUpdateEndEventData):
        grad_norm = self._compute_grad_norm()
        self._current_value = {
            "gradient_norm": grad_norm
        }

    def _compute_grad_norm(self):
        grad_norm_tensor = self._compute_sharded_grad_norm_squared()
        dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.SUM)
        grad_norm = pow(grad_norm_tensor.item(), 0.5)
        return grad_norm

    def _compute_sharded_grad_norm_squared(self):
        total = 0
        for p in self.__model.parameters():
            if p.grad is not None:
                g = p.grad.data

                param_norm = g.norm(2)
                total += pow(param_norm.item(), 2)
        return torch.tensor(total, device=self.rank)


class PreviousGradientNormTracker(Tracker[GradientUpdateEndEventData]):
    def __init__(self, rank, gradient_norm_tracker: GradientNormTracker):
        super().__init__(rank)
        self.__tracker = gradient_norm_tracker

    def handle_event(self, event_data: GradientUpdateEndEventData):
        event_type = event_data.event_type
        if event_type is EventTypes.GRADIENT_UPDATE_END:
            self._on_gradient_update(event_data)
            return

    def _on_gradient_update(self, event_data: GradientUpdateEndEventData):
        if self.__tracker.current_value is None:
            prev_grad_norm = 0
        else:
            prev_grad_norm = self.__tracker.current_value["gradient_norm"]
        self._current_value = {
            "prev_gradient_norm": prev_grad_norm
        }


class ConsecutiveBatchInnerProductTracker(Tracker[GradientUpdateEndEventData]):
    def __init__(self, model, rank):
        super().__init__(rank)
        self.__model = model
        self._previous_gradient = dict()

    def handle_event(self, event_data: GradientUpdateEndEventData):
        event_type = event_data.event_type
        if event_type is EventTypes.GRADIENT_UPDATE_END:
            self._on_gradient_update(event_data)
            return

    def _on_gradient_update(self, event_data: GradientUpdateEndEventData):
        inner_product = self._compute_inner_product()
        self._current_value = {
            "inner_product": inner_product
        }

    def _compute_inner_product(self):
        inner_product_tensor = self._compute_sharded_inner_product()
        dist.all_reduce(inner_product_tensor, op=dist.ReduceOp.SUM)
        inner_product = inner_product_tensor.item()
        return inner_product

    def _compute_sharded_inner_product(self):
        inner_product = 0
        for name, p in self.__model.named_parameters():
            if p.grad is not None:
                g = p.grad.data
                if name in self._previous_gradient.keys():
                    prev_g = self._previous_gradient[name]
                    inner_product += torch.sum(g * prev_g)
                self._previous_gradient[name] = g
                    
        return torch.tensor(inner_product, device=self.rank)


class ConsecutiveBatchCosineSimilarityTracker(Tracker[GradientUpdateEndEventData]):
    def __init__(self, rank, prev_tracker: PreviousGradientNormTracker, curr_tracker: GradientNormTracker, inn_tracker: ConsecutiveBatchInnerProductTracker):
        super().__init__(rank)
        self._prev_tracker = prev_tracker
        self._curr_tracker = curr_tracker
        self._inn_tracker = inn_tracker

    def handle_event(self, event_data: GradientUpdateEndEventData):
        event_type = event_data.event_type
        if event_type is EventTypes.GRADIENT_UPDATE_END:
            self._on_gradient_update(event_data)
            return

    def _on_gradient_update(self, event_data: GradientUpdateEndEventData):
        if self._prev_tracker.current_value["prev_gradient_norm"] == 0:
            cosine_sim = -100
        elif self._curr_tracker.current_value["gradient_norm"] == 0:
            consine_sim = -200
        else:
            cosine_sim = self._inn_tracker.current_value["inner_product"] / self._prev_tracker.current_value["prev_gradient_norm"] / self._curr_tracker.current_value["gradient_norm"]
        self._current_value = {
            "cosine_similarity": cosine_sim
        }
