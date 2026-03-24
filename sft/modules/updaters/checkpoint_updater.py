import os
import json
import shutil


class CheckpointUpdater:
    def __init__(self, log_directory: str, excluded_categories: list = None,
                 global_best_acc: float = -1.0, global_best_ckpt: str = None):
        self.state_file = os.path.join(log_directory, "checkpoint_updater_state.json")
        self.excluded_categories = set(excluded_categories) if excluded_categories else set()
        self.global_best_acc = global_best_acc
        self.global_best_ckpt = global_best_ckpt
        self.best_checkpoints = self._load_state()
        print(f"CheckpointUpdater initialized. Loaded {len(self.best_checkpoints)} best records from state file.")
        if global_best_ckpt:
            print(f"  Global best from prior stages: acc={global_best_acc}, ckpt={global_best_ckpt}")

    def _load_state(self) -> dict:
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save_state(self):
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.best_checkpoints, f, indent=4)

    def _delete_checkpoint(self, path: str):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Pruned checkpoint: {os.path.basename(path)}")
            elif os.path.isfile(path):
                os.remove(path)
                print(f"Pruned checkpoint: {os.path.basename(path)}")
        except OSError as e:
            print(f"WARNING: Error removing checkpoint {path}: {e}")

    def update(self, eval_results: dict, new_ckpt_path: str):
        if not eval_results:
            print("WARNING: eval_results is empty or None. Skipping update.")
            return
            
        all_metrics = {
            k: v for k, v in eval_results.get('per_category', {}).items()
            if k not in self.excluded_categories
        }
        all_metrics['overall_accuracy'] = eval_results.get('overall_accuracy', 0.0)
        
        obsolete_paths = set()
        new_path_became_champion = False
        
        for category, score in all_metrics.items():
            current_best_score = self.best_checkpoints.get(category, {}).get('score', 0.0)

            if score > current_best_score:
                if category in self.best_checkpoints:
                    obsolete_paths.add(self.best_checkpoints[category]['path'])
                
                self.best_checkpoints[category] = {"score": score, "path": new_ckpt_path}
                new_path_became_champion = True

        paths_to_keep = {info['path'] for info in self.best_checkpoints.values()}

        for path_to_delete in obsolete_paths:
            if path_to_delete and path_to_delete not in paths_to_keep:
                self._delete_checkpoint(path_to_delete)

        if not new_path_became_champion:
            if new_ckpt_path and new_ckpt_path not in paths_to_keep:
                self._delete_checkpoint(new_ckpt_path)

        overall_acc = all_metrics.get('overall_accuracy', 0.0)
        if self.global_best_ckpt and overall_acc > self.global_best_acc:
            old_ckpt = self.global_best_ckpt
            self.global_best_acc = overall_acc
            self.global_best_ckpt = None  # new best is in current stage, managed by paths_to_keep
            if old_ckpt not in paths_to_keep:
                print(f"[GlobalBest] New best overall acc={overall_acc:.4f}, deleting old global best")
                self._delete_checkpoint(old_ckpt)

        self._save_state()
