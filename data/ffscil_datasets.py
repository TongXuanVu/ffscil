import random
import torch
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
import dualpromptlib.utils as utils

class CICIoT23PTDataset(Dataset):
    def __init__(self, data_path, client_id, task_id, is_train=True):
        self.data_path = data_path
        if is_train:
            filename = f"client_{client_id}_task_{task_id}.pt"
            path = os.path.join(data_path, "federated_data", filename)
        else:
            filename = f"centralized_task_{task_id}.pt"
            path = os.path.join(data_path, "centralized_data", filename)
            
        if os.path.exists(path):
            print(f"Loading {path}...")
            data_dict = torch.load(path, map_location='cpu', weights_only=False)
            self.x = data_dict['x']
            self.y = data_dict['y'].long()
        else:
            print(f"Warning: File {path} not found. Using empty tensors.")
            self.x = torch.empty(0, 33)
            self.y = torch.empty(0).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def build_continual_dataloader(args, client_id=0):
    if args.dataset == 'cic_iot23':
        data_path = args.data_path
        num_tasks = args.num_tasks
        
        # Build class mask by looking at centralized data
        class_mask = []
        for t in range(1, num_tasks + 1):
            central_path = os.path.join(data_path, "centralized_data", f"centralized_task_{t}.pt")
            if os.path.exists(central_path):
                data = torch.load(central_path, map_location='cpu', weights_only=False)
                unique_labels = torch.unique(data['y']).tolist()
                class_mask.append(unique_labels)
            else:
                # Fallback if centralized data is missing
                classes_per_task = args.nb_classes // num_tasks
                start = (t-1) * classes_per_task
                end = t * classes_per_task
                class_mask.append(list(range(start, end)))
        
        dataset_list = []
        for t in range(1, num_tasks + 1):
            # Load training data for this client and task
            train_ds = CICIoT23PTDataset(data_path, client_id, t, is_train=True)
            # Load validation data (centralized for the task)
            val_ds = CICIoT23PTDataset(data_path, client_id, t, is_train=False)
            
            t_loader = DataLoader(
                train_ds, batch_size=args.batch_size, 
                shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_mem
            ) if len(train_ds) > 0 else None
            
            v_loader = DataLoader(
                val_ds, batch_size=args.batch_size, 
                shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem
            ) if len(val_ds) > 0 else None
            
            dataset_list.append({'train': t_loader, 'val': v_loader})
            
        return dataset_list, class_mask

    # ... rest of original code (not used for IoT) ...
    dataloader = list()
    return dataloader, None