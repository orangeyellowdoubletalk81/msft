import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import gc
import os
import json
import argparse

from utils.vllm_evaluator import vllm_evaluate, persist_checkpoint
from utils.dataloader_helper import DataLoader
from utils.utils import setup, update_accuracy_log, CUDATimer

from modules.event_system import *
from modules.events.event_data import *
from modules.updaters.dataloader_updater import test_filter_function
from modules.updaters.filter_functions import full_pass_filter_function
from modules.updaters.checkpoint_updater import CheckpointUpdater

from settings.configs import *

def _step_scheduler_and_optimizer(optimizer, scheduler):
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()


def _settings_before_evaluation(optimizer):
    torch.cuda.empty_cache()
    gc.collect()


def _is_grad_update_step(batch_idx, grad_accum_period, train_loader_size):
    curr_step = batch_idx + 1
    return curr_step % grad_accum_period == 0 or curr_step == train_loader_size


def _is_logging_step(batch_idx, log_divisor, train_loader_size, dataloader_updater):
    curr_step = batch_idx + 1

    if curr_step == train_loader_size:
        return True

    log_interval = max(1, round(train_loader_size / log_divisor))
    return dataloader_updater.is_updated or curr_step % log_interval == 0


def _is_save_step(max_epoch, epoch, batch_idx, save_divisor, train_loader_size, dataloader_updater):
    curr_step = batch_idx + 1

    if dataloader_updater.is_updated:
        return True

    if save_divisor == 0:
        return curr_step == train_loader_size and epoch == max_epoch

    if curr_step == train_loader_size:
        return True

    unit = int(round(train_loader_size / save_divisor))
    return curr_step % unit == 0


def _is_eval_step(batch_idx, eval_divisor, train_loader_size, dataloader_updater):
    curr_step = batch_idx + 1

    if dataloader_updater.is_updated or curr_step == train_loader_size:
        return True

    unit = int(train_loader_size / eval_divisor)
    if unit == 0:
        return False
    if curr_step % unit == 0:
        remaining = train_loader_size - curr_step
        return remaining >= unit
    return False


def _is_epoch_early_stop_condition(dataloader_updater):
    return dataloader_updater.is_updated


def train(train_settings, test_settings, model_settings, log_and_save_settings, model, rank, world_size, dataloader, dataloader_updater, tokenizer, optimizer, scheduler, epoch, dataset_settings=None, updater=None):
    dist.barrier()
    model.train()
    dataloader.sampler_train.set_epoch(epoch)

    progress_bar = dataloader.get_train_progress_bar(rank, epoch)

    for batch_idx, data in progress_bar:
        BATCH_START_EVENT.invoke(
            BatchStartEventData(
                epoch=epoch,
                step=batch_idx,
                world_size=world_size,
                batch_categories=data["category"],
                train_settings=train_settings
            )
        )

        input_ids = data["input_ids"].to(rank)
        attention_mask = data["attention_mask"].to(rank)
        labels = data["labels"].to(rank)

        output = model.forward(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = output.loss / train_settings.grad_accumulation
        loss.backward()

        loss_value_for_log  = loss.item() * train_settings.grad_accumulation
        batch_size_for_log  = len(input_ids)

        LOSS_BACKWARD_END_EVENT.invoke(
            LossBackwardEndEventData(
                epoch=epoch,
                step=batch_idx,
                world_size=world_size,
                loss_value=loss_value_for_log,
                batch_size=batch_size_for_log
            )
        )

        if _is_grad_update_step(batch_idx, train_settings.grad_accumulation, dataloader.train_size):
            GRADIENT_UPDATE_END_EVENT.invoke(
                GradientUpdateEndEventData(
                    epoch=epoch,
                    step=batch_idx,
                    world_size=world_size,
                    scheduler=scheduler
                )
            )
            dataloader.sampler_train.set_epoch(epoch)
            _step_scheduler_and_optimizer(optimizer, scheduler)
        
        float_epoch = (batch_idx + 1) / dataloader.train_size + (epoch - 1)

        if _is_logging_step(batch_idx, log_and_save_settings.log_divisor, dataloader.train_size, dataloader_updater):
            TRAIN_LOSS_LOG_EVENT.invoke(
                TrainLossLogEventData(
                    epoch=epoch,
                    step=batch_idx,
                    world_size=world_size,
                    progress_bar=progress_bar
                )
            )
            dist.barrier()

        if _is_eval_step(batch_idx, test_settings.eval_divisor, dataloader.train_size, dataloader_updater):
            _settings_before_evaluation(optimizer)

            eval_results = vllm_evaluate(
                model, tokenizer, rank, world_size, model_settings, test_settings,
                log_and_save_settings, dataloader, epoch, float_epoch,
                dataset_settings=dataset_settings
            )
            should_save = False
            if log_and_save_settings.adaptive_checkpointing:
                should_save = True
            elif _is_save_step(train_settings.epochs, epoch, batch_idx, log_and_save_settings.save_divisor, dataloader.train_size, dataloader_updater):
                should_save = True

            if should_save:
                dest_dir = log_and_save_settings.get_hf_ckpt_save_dir(float_epoch)
                persist_checkpoint(log_and_save_settings.eval_tmp_ckpt_dir, dest_dir, rank)

            if rank == 0 and eval_results is not None:
                print("--- Real-time Evaluation Result ---")
                print(eval_results)
                print("------------------------------------")

                accuracy_log_path = log_and_save_settings.accuracy_log_path
                update_accuracy_log(accuracy_log_path, eval_results, float_epoch)

                if log_and_save_settings.adaptive_checkpointing:
                    current_ckpt_path = log_and_save_settings.get_hf_ckpt_save_dir(float_epoch)
                    updater.update(eval_results, current_ckpt_path)

            EVAL_END_EVENT.invoke(
                EvalEndEventData(
                    epoch=epoch,
                    step=batch_idx,
                    world_size=world_size
                )
            )
            dist.barrier()
            model.train()
        BATCH_END_EVENT.invoke(
            BatchEndEventData(
                epoch=epoch,
                step=batch_idx,
                world_size=world_size
            )
        )

        if _is_epoch_early_stop_condition(dataloader_updater):
            break
        


def fsdp_main(rank, dataset_settings: DatasetSettings, train_settings: TrainSettings, test_settings: TestSettings, model_settings: ModelSettings, log_and_save_settings: LogAndSaveSettings, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    if log_and_save_settings.resume_ckpt_dir:
        model = model_settings.get_model_with_hf_checkpoint(log_and_save_settings.resume_ckpt_dir, rank)
    else:
        model = model_settings.get_fsdp_wrapped_model(rank)

    tokenizer = model_settings.get_tokenizer()

    dataloader = DataLoader(
        dataset_settings, train_settings, test_settings, log_and_save_settings,
        tokenizer, rank, world_size
    )

    print(f"DEBUG: excluding... {args.exclude_cats}")

    if args.exclude_cats:
        exclude_list = [cat.strip() for cat in args.exclude_cats.split(',')]
        print(f"INFO: RANK {rank} | Excluding categories from training set: {exclude_list}")

        def filter_function(examples):
            keep_mask = [
                category not in exclude_list 
                for category in examples['category']
            ]
            return keep_mask

        dataloader.filter_dataset(rank, world_size, filter_function)


    setup_event_system(log_and_save_settings, model, rank)
    dataloader_updater = setup_dataloader_updater(rank, dataloader, full_pass_filter_function)

    timer = CUDATimer()

    optimizer = train_settings.get_optimizer(model)
    scheduler = train_settings.get_scheduler(optimizer, train_settings.epochs * dataloader.train_size)

    timer.record_start()

    updater = None
    if rank == 0 and log_and_save_settings.adaptive_checkpointing:
        updater = CheckpointUpdater(
            log_directory=log_and_save_settings.log_directory,
            excluded_categories=exclude_list if args.exclude_cats else None,
            global_best_acc=args.global_best_acc,
            global_best_ckpt=args.global_best_ckpt,
        )

    initial_eval = vllm_evaluate(
        model, tokenizer, rank, world_size, model_settings, test_settings,
        log_and_save_settings, dataloader, 0, 0,
        dataset_settings=dataset_settings
    )
    if rank == 0 and initial_eval is not None:
        accuracy_log_path = log_and_save_settings.accuracy_log_path
        update_accuracy_log(accuracy_log_path, initial_eval, 0.0)

    if log_and_save_settings.adaptive_checkpointing:
        if rank == 0:
            print(f"Saving initial checkpoint (Epoch 0.00)")
        ckpt_path_0 = log_and_save_settings.get_hf_ckpt_save_dir(0.00)
        persist_checkpoint(log_and_save_settings.eval_tmp_ckpt_dir, ckpt_path_0, rank)
        if rank == 0 and updater is not None and initial_eval is not None:
            updater.update(initial_eval, ckpt_path_0)

    for epoch in range(1, train_settings.epochs + 1):
        train(train_settings, test_settings, model_settings, log_and_save_settings, model, rank, world_size, dataloader, dataloader_updater, tokenizer, optimizer, scheduler, epoch, dataset_settings=dataset_settings, updater=updater)
        dataloader_updater.reset_flags()

    timer.record_end()
    timer.print_elapsed_time(rank)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FSDP Finetune')
    parser.add_argument("--config", type=str, required=True, help="Config file location")
    parser.add_argument('--seed', type=int, required=True,help='random seed')

    parser.add_argument('--exp-name', type=str, required=True, help="Unique name for this experiment run/stage")
    parser.add_argument('--exclude-cats', type=str, default="", help="comma-separated list of categories to exclude")
    parser.add_argument('--resume-from', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--gpus', type=str, default=None, help="Comma-separated GPU IDs (e.g. '0,1'). Auto-detects all GPUs if not specified.")
    parser.add_argument('--master-port', type=str, default=None, help="Master port for distributed training")
    parser.add_argument('--global-best-acc', type=float, default=-1.0, help="Global best overall accuracy from prior stages")
    parser.add_argument('--global-best-ckpt', type=str, default=None, help="Path to global best checkpoint from prior stages")

    args = parser.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.master_port:
        os.environ["MASTER_PORT"] = args.master_port

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    params = json.load(open(args.config))

    train_batch_settings = TrainBatchSettings(
        batch_size=params["batch_size"],
        grad_accumulation=params["grad_accumulation"]
    )
    train_settings = TrainSettings(
        epochs=params["epochs"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        warm_up_ratio=params["warm_up_ratio"],
        scheduler_name=params["scheduler"],
        optimizer_name=params["optimizer"],
        batch_settings=train_batch_settings,
    )

    test_batch_settings = TestBatchSettings(
        batch_size=params["test_batch_size"]
    )
    test_settings = TestSettings(
        eval_divisor=params["eval_divisor"],
        gen_length=params["gen_length"],
        is_zeroshot=params["is_zeroshot"],
        batch_settings=test_batch_settings,
    )

    model_settings = ModelSettings(
        model_name=params["model_name"],
        precision_name=params["precision"]
    )

    log_and_save_settings = LogAndSaveSettings(
        exp_name=f"{args.exp_name}_{args.seed}",
        log_dir=params["log_save"],
        ckpt_dir=params["ckpt_save"],
        log_divisor=params["log_divisor"],
        save_divisor=params["save_divisor"],
        adaptive_checkpointing=params.get("adaptive_checkpointing", False),
        resume_ckpt_dir=args.resume_from
    )

    dataset_settings = DatasetSettings(
        dataset_file=params["dataset_file"],
        fiveshot_prompt_file=params["fewshot_file"],
        max_train_length=params["max_length_train"],
        max_test_length=params["max_length_test"],
    )

    log_config(args, dataset_settings, train_settings, test_settings, model_settings, log_and_save_settings)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(dataset_settings, train_settings, test_settings, model_settings, log_and_save_settings, WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)