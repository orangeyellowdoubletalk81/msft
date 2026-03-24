from tqdm import tqdm
from enum import Enum, auto
from dataclasses import dataclass, field

from settings.configs import TrainSettings
from utils.dataloader_helper import DataLoader

class EventTypes(Enum):
    BATCH_START = auto()
    GRADIENT_UPDATE_END = auto()
    LOSS_BACKWARD_END = auto()
    TRAIN_LOSS_LOG = auto()
    EVAL_END = auto()
    BATCH_END = auto()
    QUANTUM_START = auto()
    QUANTUM_END = auto()


@dataclass
class EventData:
    event_type: EventTypes = field(init=False)
    epoch: int
    world_size: int


@dataclass
class EpochEventData(EventData):
    step: float


@dataclass
class BatchStartEventData(EpochEventData):
    batch_categories: list[str]
    train_settings: TrainSettings


@dataclass
class GradientUpdateEndEventData(EpochEventData):
    scheduler: object


@dataclass
class LossBackwardEndEventData(EpochEventData):
    loss_value: float
    batch_size: int


@dataclass
class TrainLossLogEventData(EpochEventData):
    progress_bar: tqdm


@dataclass
class EvalEndEventData(EpochEventData):
    pass


@dataclass
class BatchEndEventData(EpochEventData):
    pass


@dataclass
class QuantumStartEventData(EventData):
    dataloader: DataLoader


@dataclass
class QuantumEndEventData(EventData):
    best_epoch: float
    best_accuracy: float
    subdataset_accuracy: dict
