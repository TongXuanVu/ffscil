import random
import torch
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
import dualpromptlib.utils as utils

class CICIoT23PTDataset(Dataset):
    _global_test_cache = None  # Cache để tránh load file 1.9GB nhiều lần

    def __init__(self, data_path, client_id, task_id, is_train=True, fs_mode='1percent'):
        fed_dir = "federated_data_10shot" if fs_mode == '10shot' else "federated_data_fewshot"
        cen_dir = "centralized_data_10shot" if fs_mode == '10shot' else "centralized_data_fewshot"

        if is_train:
            # Check multiple possible paths
            possible_paths = [
                os.path.join(data_path, fed_dir, f"client_{client_id}_task_{task_id}.pt"),
                os.path.join(data_path, "10shot", fed_dir, f"client_{client_id}_task_{task_id}.pt") if fs_mode == '10shot' else os.path.join(data_path, "data", fed_dir, f"client_{client_id}_task_{task_id}.pt"),
                os.path.join(data_path, "1percent", fed_dir, f"client_{client_id}_task_{task_id}.pt")
            ]
            file_path = next((p for p in possible_paths if os.path.exists(p)), possible_paths[0])
            
            if os.path.exists(file_path):
                data = torch.load(file_path, map_location='cpu', weights_only=False)
                self.x = data['x']
                self.y = data['y'].long()
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
            all_y = CICIoT23PTDataset._global_test_cache['y'].long()
            
            possible_central_paths = [
                os.path.join(data_path, cen_dir, f"centralized_task_{task_id}.pt"),
                os.path.join(data_path, "10shot", cen_dir, f"centralized_task_{task_id}.pt") if fs_mode == '10shot' else os.path.join(data_path, "data", cen_dir, f"centralized_task_{task_id}.pt"),
                os.path.join(data_path, "1percent", cen_dir, f"centralized_task_{task_id}.pt")
            ]
            central_path = next((p for p in possible_central_paths if os.path.exists(p)), possible_central_paths[0])

            if os.path.exists(central_path):
                central_data = torch.load(central_path, map_location='cpu', weights_only=False)
                task_labels = torch.unique(central_data['y'])
                
                mask = torch.isin(all_y, task_labels)
                self.x = all_x[mask]
                self.y = all_y[mask]
                print(f"  Task {task_id} test set: {len(self.y)} samples.")
            else:
                self.x = torch.empty(0)
                self.y = torch.empty(0, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

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
        val_ds = CICIoT23PTDataset(data_path, client_id, t, is_train=False, fs_mode=fs_mode)
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
                os.path.join(data_path, "10shot", cen_dir, f"centralized_task_{t}.pt") if fs_mode == '10shot' else os.path.join(data_path, "data", cen_dir, f"centralized_task_{t}.pt"),
                os.path.join(data_path, "1percent", cen_dir, f"centralized_task_{t}.pt")
            ]
            central_path = next((p for p in possible_central_paths if os.path.exists(p)), possible_central_paths[0])
            
            if os.path.exists(central_path):
                central_data = torch.load(central_path, map_location='cpu', weights_only=False)
                class_mask.append(torch.unique(central_data['y']).tolist())
            else:
                class_mask.append([])

    return [data_loader_list], class_mask