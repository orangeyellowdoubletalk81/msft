import os
import json

from modules.trackers.tracker import Tracker
from modules.events.event import EventHandler
from modules.events.event_data import EventData

class Logger(EventHandler[EventData]):
    def __init__(self, rank, output_file, associated_trackers: list[Tracker], log_steps: bool=True):
        self.rank = rank
        self.output_file = output_file
        self.output_ext = os.path.splitext(output_file)[-1]
        self.trackers = associated_trackers
        self.log_steps = log_steps

    def handle_event(self, event_data: EventData):
        if self.rank != 0:
            return

        if self.log_steps:
            result_dict = {
                "epoch": event_data.epoch,
                "step": event_data.step
            }
        else:
            result_dict = {
                "epoch": event_data.epoch
            }
        for tracker in self.trackers:
            try:
                result_dict.update(tracker.current_value)
            except:
                if tracker.current_value is None:
                    raise AssertionError(f"{tracker}가 제때 최신화 되지 못했습니다.")
                
                if not isinstance(tracker.current_value, dict):
                    raise TypeError(f"{tracker}의 current_value가 dictionary가 아닙니다!")
        
        self._flush(result_dict)
    
    def _flush(self, result_dict: dict):
        if self.output_ext == ".json":
            self._json_log(**result_dict)
            return
        
        raise NotImplementedError("해당 확장자 logger는 구현되지 않았습니다.")
        
    def _json_log(self, **kwargs):
        if os.path.exists(self.output_file):
            with open(self.output_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(kwargs)
        with open(self.output_file, "w") as f:
            json.dump(logs, f, indent=4)
