from modules.updaters.dataloader_updater import *

def single_category_filter(examples, event_data):
    return [category == event_data.filter_category for category in examples["category"]]


def full_pass_filter_function(examples, event_data):
    return [True] * len(examples)
