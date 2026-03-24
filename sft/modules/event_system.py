from modules.events.event import *
from modules.events.event_data import *
from modules.loggers.logger import Logger

from modules.trackers.lr_tracker import *
from modules.trackers.settings_tracker import *
from modules.trackers.gradient_tracker import *
from modules.trackers.train_loss_tracker import TrainLossTracker
from modules.trackers.best_epoch_tracker import BestEpochTracker
from modules.trackers.dataset_count_tracker import DatasetCountTracker
from modules.trackers.category_count_tracker import CategoryCountTracker
from modules.updaters.dataloader_updater import DataLoaderUpdaterOnQuantumStart

BATCH_START_EVENT = Event[BatchEndEventData](EventTypes.BATCH_START, [])
GRADIENT_UPDATE_END_EVENT = Event[GradientUpdateEndEventData](EventTypes.GRADIENT_UPDATE_END, [])
LOSS_BACKWARD_END_EVENT = Event[LossBackwardEndEventData](EventTypes.LOSS_BACKWARD_END, [])
TRAIN_LOSS_LOG_EVENT = Event[TrainLossLogEventData](EventTypes.TRAIN_LOSS_LOG, [])
EVAL_END_EVENT = Event[EvalEndEventData](EventTypes.EVAL_END, [])
BATCH_END_EVENT = Event[BatchEndEventData](EventTypes.BATCH_END, [])

QUANTUM_START_EVENT = Event[QuantumStartEventData](EventTypes.QUANTUM_START, [])
QUANTUM_END_EVENT = Event[QuantumEndEventData](EventTypes.QUANTUM_END, [])


def setup_event_system(log_and_save_settings, model, rank):
    train_loss_tracker = TrainLossTracker(rank)
    category_count_tracker = CategoryCountTracker(rank)
    learning_rate_tracker = LRTracker(rank)
    gradient_norm_tracker = GradientNormTracker(model, rank)
    
    gradient_update_logger = Logger(rank, log_and_save_settings.gradient_log_path, [category_count_tracker, learning_rate_tracker, gradient_norm_tracker]) 
    train_loss_logger = Logger(rank, log_and_save_settings.train_loss_log_path, [train_loss_tracker])
    
    BATCH_START_EVENT.add_handlers([category_count_tracker])
    GRADIENT_UPDATE_END_EVENT.add_handlers([category_count_tracker, learning_rate_tracker, gradient_norm_tracker])
    LOSS_BACKWARD_END_EVENT.add_handlers([train_loss_tracker])
    TRAIN_LOSS_LOG_EVENT.add_handlers([train_loss_tracker])

    GRADIENT_UPDATE_END_EVENT.add_handlers([gradient_update_logger])
    TRAIN_LOSS_LOG_EVENT.add_handlers([train_loss_logger])


def setup_dataloader_updater(rank, dataloader, filter_function):
    dataloader_updater = DataLoaderUpdaterOnQuantumStart(rank, dataloader, filter_function)
    QUANTUM_START_EVENT.add_handlers([dataloader_updater])
    return dataloader_updater


def setup_event_system_after_dataloader_registration(log_and_save_settings, model, rank):
    current_dataset_count_tracker = DatasetCountTracker(rank)
    best_epoch_tracker = BestEpochTracker(rank)

    quantum_start_logger = Logger(rank, log_and_save_settings.quantum_start_log_path, [current_dataset_count_tracker], log_steps=False)
    quantum_end_logger = Logger(rank, log_and_save_settings.quantum_end_log_path, [best_epoch_tracker], log_steps=False)

    QUANTUM_START_EVENT.add_handlers([current_dataset_count_tracker])
    QUANTUM_END_EVENT.add_handlers([best_epoch_tracker])

    QUANTUM_START_EVENT.add_handlers([quantum_start_logger])
    QUANTUM_END_EVENT.add_handlers([quantum_end_logger])
