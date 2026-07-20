import random
import torch
import os
import json
import glob
from torch.utils.data import Dataset, DataLoader, TensorDataset
import dualpromptlib.utils as utils

# Dai lop tuan tu chuan cua CIC-IoT23: task 1..6 gom 6,6,6,6,5,5 lop
CICIOT23_TASK_CLASSES = [
    list(range(0, 6)), list(range(6, 12)), list(range(12, 18)),
    list(range(18, 24)), list(range(24, 29)), list(range(29, 34)),
]

# ── Remap label cho bo data 100-client ────────────────────────────────────────
# Bo 100-client giu NGUYEN label ID goc voi thu tu task phi tuan tu, mo ta trong
# `task_mapping_label_ids.json`. Code CIL gia dinh label tuan tu 0..33 theo thu tu task.
# Bo data cu (da tuan tu san) khong co file json -> khong doi gi (tuong thich nguoc).
_LABEL_LUT = None
_LABEL_LUT_READY = False


def _get_label_lut(data_path=None):
    global _LABEL_LUT, _LABEL_LUT_READY
    if _LABEL_LUT_READY:
        return _LABEL_LUT
    _LABEL_LUT_READY = True

    candidates = []
    if data_path:
        candidates += [
            os.path.join(data_path, "task_mapping_label_ids.json"),
            os.path.join(data_path, "data", "task_mapping_label_ids.json"),
        ]
    if os.path.exists("/kaggle/input"):
        candidates += sorted(glob.glob("/kaggle/input/**/task_mapping_label_ids.json", recursive=True))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "task_mapping_label_ids.json"))

    for path in candidates:
        if path and os.path.exists(path):
            with open(path, "r") as f:
                task_orders = json.load(f)
            flat = [int(c) for task in task_orders for c in task]
            if sorted(flat) != list(range(len(flat))):
                print(f"[FFSCIL] CANH BAO: {path} khong phu kin 0..N-1, bo qua remap.")
                continue
            lut = torch.full((max(flat) + 1,), -1, dtype=torch.long)
            for seq_id, orig_id in enumerate(flat):
                lut[orig_id] = seq_id
            _LABEL_LUT = lut
            print(f"[FFSCIL] Remap label goc -> tuan tu theo: {path}")
            return _LABEL_LUT

    print("[FFSCIL] Khong thay task_mapping_label_ids.json -> gia dinh label da tuan tu.")
    return None


def _remap_labels(y, data_path=None):
    lut = _get_label_lut(data_path)
    if lut is None or y is None or len(y) == 0:
        return y
    out = lut[y.long()]
    if (out < 0).any():
        bad = torch.unique(y[out < 0]).tolist()
        raise ValueError(f"[FFSCIL] Label {bad} khong co trong task_mapping_label_ids.json")
    return out

class CICIoT23PTDataset(Dataset):
    _global_test_cache = None  # Cache để tránh load file 1.9GB nhiều lần

    def __init__(self, data_path, client_id, task_id, is_train=True, fs_mode='1percent'):
        if fs_mode == 'full':
            fed_dir = "federated_data"
            cen_dir = "centralized_data"
        elif fs_mode == '10shot':
            fed_dir = "federated_data_10shot"
            cen_dir = "centralized_data_10shot"
        else:
            fed_dir = "federated_data_fewshot"
            cen_dir = "centralized_data_fewshot"

        if is_train:
            # Check multiple possible paths (them layout PHANG: *.pt ngay trong data_path,
            # vi Kaggle dataset khong giu thu muc con)
            possible_paths = [
                os.path.join(data_path, fed_dir, f"client_{client_id}_task_{task_id}.pt"),
                os.path.join(data_path, "10shot", fed_dir, f"client_{client_id}_task_{task_id}.pt"),
                os.path.join(data_path, "fewshot", fed_dir, f"client_{client_id}_task_{task_id}.pt"),
                os.path.join(data_path, f"client_{client_id}_task_{task_id}.pt"),
            ]
            file_path = next((p for p in possible_paths if os.path.exists(p)), possible_paths[0])

            if os.path.exists(file_path):
                data = torch.load(file_path, map_location='cpu', weights_only=False)
                self.x = data['x']
                self.y = _remap_labels(data['y'].long(), data_path)
            else:
                self.x = torch.empty(0)
                self.y = torch.empty(0, dtype=torch.long)
        else:
            # Dùng global_test_data.pt và lọc nhãn thuộc về task_id
            possible_test_paths = [
                os.path.join(data_path, "global_test_data.pt"),
                os.path.join(data_path, "data", "global_test_data.pt")
            ]
            test_file = next((p for p in possible_test_paths if os.path.exists(p)), possible_test_paths[0])
            
            if CICIoT23PTDataset._global_test_cache is None:
                print(f"Loading global test data from {test_file}...")
                CICIoT23PTDataset._global_test_cache = torch.load(test_file, map_location='cpu', weights_only=False)
            
            all_x = CICIoT23PTDataset._global_test_cache['x']
            all_y = _remap_labels(CICIoT23PTDataset._global_test_cache['y'].long(), data_path)

            possible_central_paths = [
                os.path.join(data_path, cen_dir, f"centralized_task_{task_id}.pt"),
                os.path.join(data_path, "10shot", cen_dir, f"centralized_task_{task_id}.pt"),
                os.path.join(data_path, "fewshot", cen_dir, f"centralized_task_{task_id}.pt")
            ]
            central_path = next((p for p in possible_central_paths if os.path.exists(p)), possible_central_paths[0])

            if os.path.exists(central_path):
                central_data = torch.load(central_path, map_location='cpu', weights_only=False)
                task_labels = _remap_labels(central_data['y'].long(), data_path).unique()
            else:
                # Bo data 100-client khong co centralized_data -> dung dai lop tuan tu chuan
                # cua CIC-IoT23 (task 1..6 = 6,6,6,6,5,5 lop) sau khi da remap.
                task_labels = torch.tensor(CICIOT23_TASK_CLASSES[task_id - 1], dtype=torch.long)
                print(f"  Task {task_id}: khong co centralized_data, dung dai lop chuan "
                      f"{task_labels.tolist()}")

            mask = torch.isin(all_y, task_labels)
            self.x = all_x[mask]
            self.y = all_y[mask]
            print(f"  Task {task_id} test set: {len(self.y)} samples.")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

# Cache val dataset theo (task_id, fs_mode) — dung chung cho MOI client
_VAL_DS_CACHE = {}


def build_continual_dataloader(args, client_id=0, specific_task=None):
    """
    Build dataloaders for class-incremental tasks.
    Optimization:
    - Train loader: Only for specific_task.
    - Val loader: For ALL tasks up to specific_task (for forgetting evaluation).
    """
    data_path = args.data_path
    num_tasks = args.num_tasks
    data_loader_list = []
    class_mask = []
    
    for t in range(1, num_tasks + 1):
        # specific_task is 0-indexed, t is 1-indexed
        is_current_task = (specific_task is not None and (t - 1) == specific_task)
        is_future_task = (specific_task is not None and (t - 1) > specific_task)
        
        if is_future_task:
            data_loader_list.append(None)
            class_mask.append([])
            continue

        fs_mode = getattr(args, 'fs_mode', '1percent')

        # 1. Load Training Data (Only if it's the current task)
        if is_current_task:
            train_ds = CICIoT23PTDataset(data_path, client_id, t, is_train=True, fs_mode=fs_mode)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, 
                shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_mem
            ) if len(train_ds) > 0 else None
        else:
            train_loader = None

        # 2. Load Validation Data (For current and past tasks)
        # QUAN TRONG: val set KHONG phu thuoc client_id (loc tu global_test_data theo task),
        # nen cache lai theo task. Neu khong, voi 100 client se dung lai 100 ban sao
        # ~2.5M mau moi ban -> het RAM (kernel bi OOM kill).
        _val_key = (t, fs_mode)
        if _val_key in _VAL_DS_CACHE:
            val_ds = _VAL_DS_CACHE[_val_key]
        else:
            val_ds = CICIoT23PTDataset(data_path, client_id, t, is_train=False, fs_mode=fs_mode)
            _VAL_DS_CACHE[_val_key] = val_ds
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, 
            shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem
        ) if len(val_ds) > 0 else None

        data_loader_list.append({'train': train_loader, 'val': val_loader})
        
        # 3. Build Class Mask (needed for model output layers)
        if len(val_ds.y) > 0:
            class_mask.append(torch.unique(val_ds.y).tolist())
        else:
            # Fallback to centralized data to get labels if test set is empty
            cen_dir = "centralized_data_10shot" if fs_mode == '10shot' else "centralized_data_fewshot"
            possible_central_paths = [
                os.path.join(data_path, cen_dir, f"centralized_task_{t}.pt"),
                os.path.join(data_path, "10shot", cen_dir, f"centralized_task_{t}.pt"),
                os.path.join(data_path, "fewshot", cen_dir, f"centralized_task_{t}.pt")
            ]
            central_path = next((p for p in possible_central_paths if os.path.exists(p)), possible_central_paths[0])
            
            if os.path.exists(central_path):
                central_data = torch.load(central_path, map_location='cpu', weights_only=False)
                class_mask.append(torch.unique(central_data['y']).tolist())
            else:
                class_mask.append([])

    return [data_loader_list], class_mask