import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import os
import copy
from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from data.ffscil_datasets import build_continual_dataloader
from piplib.ffscil_xpip_engine1v import *
# import dualpromptlib.modelsX as models
import dualpromptlib.utils as utils
from dualpromptlib.cnn1d_prompt import create_cnn1d_prompt
from fed_pip_utils_nogp import * 

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(state, os.path.join(output_dir, filename))
    if is_best:
        torch.save(state, os.path.join(output_dir, 'model_best.pth'))

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Re-enforcing some args as per user request
    if args.batch_size_override:
        args.batch_size = args.batch_size_override

    args.use_g_prompt = False
    args.use_prefix_tune_for_g_prompt = False
    args.e_prompt_layer_idx = [0,1,2,3,4]
    
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    data_loaders=[]
    class_masks=[]
    models_list=[]
    original_models=[]
    models_without_ddp=[]
    optimizers=[]
    lr_schedulers=[]

    # Initial class mask build (to get test_sizes and class structure)
    _, full_class_mask = build_continual_dataloader(args, client_id=0)
    
    # Test sizes for weighted metrics (pre-compute using metadata or one-by-one)
    test_sizes = []
    for t in range(1, args.num_tasks + 1):
        # We need validation loader to get size. Let's load only validation for this check.
        # Alternatively, we can use 1000 as a dummy if not strictly needed for weighted avg,
        # but let's try to get real sizes.
        v_loader_list, _ = build_continual_dataloader(args, client_id=0, specific_task=t)
        v_loader = v_loader_list[t-1]['val']
        if v_loader is not None:
            test_sizes.append(len(v_loader.dataset))
        else:
            test_sizes.append(0)
    
    print(f"Test sizes per task: {test_sizes}")

    # Dataloaders will be populated per task
    data_loaders = [None] * args.num_clients
    class_masks = [None] * args.num_clients

    # Create Models
    if args.model == 'cnn1d_prompt':
        print(f"Creating 1D-CNN Prompt model for IoT data (input_dim={args.input_size})...")
        original_model = create_cnn1d_prompt(input_dim=args.input_size, num_classes=args.nb_classes).to(device)
        server_model = create_cnn1d_prompt(input_dim=args.input_size, num_classes=args.nb_classes).to(device)
    else:
        original_model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        ).to(device)

        server_model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            prompt_length=args.length,
            embedding_key=args.embedding_key,
            prompt_init=args.prompt_key_init,
            prompt_pool=args.prompt_pool,
            prompt_key=args.prompt_key,
            pool_size=args.size,
            top_k=args.top_k,
            batchwise_prompt=args.batchwise_prompt,
            prompt_key_init=args.prompt_key_init,
            head_type=args.head_type,
            use_prompt_mask=args.use_prompt_mask,
            use_g_prompt=args.use_g_prompt,
            g_prompt_length=args.g_prompt_length,
            g_prompt_layer_idx=args.g_prompt_layer_idx,
            use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
            use_e_prompt=args.use_e_prompt,
            e_prompt_layer_idx=args.e_prompt_layer_idx,
            use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
            same_key_value=args.same_key_value,
        ).to(device)
    print("DEBUG: DONE Creating Models")

    for i in range(args.num_clients):
        model = copy.deepcopy(server_model).to(device)
        models_list.append(model)
        original_models.append(original_model)

    # Freeze strategy
    if args.freeze:
        for i in range(args.num_clients):
            for p in original_models[i].parameters(): p.requires_grad = False
            for n, p in models_list[i].named_parameters():
                if n.startswith(tuple(args.freeze)): p.requires_grad = False

    server_model_without_ddp = server_model
    for i in range(args.num_clients):
        models_without_ddp.append(models_list[i])

    # Distributed Setup
    if args.distributed:
        server_model_without_ddp = server_model.module
        for i in range(args.num_clients):
            models_list[i] = torch.nn.parallel.DistributedDataParallel(models_list[i], device_ids=[args.gpu])
            models_without_ddp[i] = models_list[i].module

    # Optimizers
    server_optimizer = create_optimizer(args, server_model_without_ddp)
    for i in range(args.num_clients):
        optimizers.append(create_optimizer(args, models_without_ddp[i]))
        if args.sched != 'constant':
            lr_scheduler, _ = create_scheduler(args, optimizers[i])
            lr_schedulers.append(lr_scheduler)
        else:
            lr_schedulers.append(None)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Variables for training state
    start_task = 0
    start_round = 0
    all_time_round = 0
    global_prototype = None
    global_prototype_var = None
    all_global_prototype = {}
    all_global_prototype_var = {}
    fixed_FC_dict = None
    fixed_FC_dict2 = None

    # Resume Logic
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu')
            server_model_without_ddp.load_state_dict(checkpoint['state_dict'])
            server_optimizer.load_state_dict(checkpoint['optimizer'])
            start_task = checkpoint['task_id']
            start_round = checkpoint['n_round'] + 1
            all_time_round = checkpoint['all_time_round']
            all_global_prototype = checkpoint.get('all_global_prototype', {})
            all_global_prototype_var = checkpoint.get('all_global_prototype_var', {})
            fixed_FC_dict = checkpoint.get('fixed_FC_dict')
            fixed_FC_dict2 = checkpoint.get('fixed_FC_dict2')
            
            # Reset rounds if task finished
            if start_round >= args.rounds_per_task:
                start_round = 0
                start_task += 1
            
            # Sync clients
            FedDistribute(server_model, models_list, args.distributed)
            print(f"=> loaded checkpoint '{args.resume}' (task {start_task}, round {start_round})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    if args.eval:
        print("=> Running in EVAL mode")
        for t in range(start_task + 1):
            evaluate_server_global_model3(server_model, server_model_without_ddp, original_model,
                                criterion, data_loaders[0], server_optimizer, None,
                                device, None, t, test_sizes, 
                                all_global_prototype, all_global_prototype_var, args)
        return

    # Training Loop
    start_time = time.time()
    FedAvgWithHead(server_model, models_list, args.distributed)

    for task_id in range(start_task, args.num_tasks):
        # Build dataloaders ONLY for the current task for ALL clients
        for i in range(args.num_clients):
            dl, cm = build_continual_dataloader(args, client_id=i, specific_task=task_id+1)
            data_loaders[i] = dl
            class_masks[i] = cm
            
        print(f"\n{'='*20} Starting Task {task_id+1}/{args.num_tasks} {'='*20}")
        print(f"Loaded data for Task {task_id+1}")
        if task_id > 0:
            args.classes_per_task = args.fs_classes
            args.available_classes = args.available_fs_classes
            args.epochs = args.fs_epochs
            # Freeze head if moving to FS tasks
            if task_id == 1 and start_round == 0:
                for n, p in server_model.head.named_parameters(): p.requires_grad = False
                for c in range(len(models_list)):
                    for n, p in models_list[c].head.named_parameters(): p.requires_grad = False
                args.lr = args.lr_fs
                optimizers = [create_optimizer(args, m) for m in models_without_ddp]
        else:
            args.classes_per_task = args.base_classes
            args.available_classes = args.available_base_classes

        clients_participations = [0] * args.num_clients

        for n_round in range(start_round if task_id == start_task else 0, args.rounds_per_task):
            print(f"Task [{task_id+1}] Global Round : {all_time_round+1} (Local Round {n_round+1})")
            
            clients_index = random.sample(range(args.num_clients), args.local_clients)
            
            idx_notrain = [x for x in clients_index if clients_participations[x]==0]
            idx_trained = [x for x in clients_index if clients_participations[x]>0]
            
            FedDistribute(server_model,[models_list[i] for i in idx_trained],args.distributed)
            if task_id == 0:
                FedDistributeWithHead(server_model,[models_list[i] for i in idx_notrain],args.distributed)
            else:
                FedDistribute(server_model,[models_list[i] for i in idx_notrain],args.distributed)

            for i in clients_index: clients_participations[i] += 1
            
            # Filter clients that actually have data for this task
            active_clients = [i for i in clients_index if data_loaders[i][task_id]['train'] is not None]
            if not active_clients:
                print(f"Skipping Round {n_round+1} for Task {task_id+1}: No clients have data.")
                continue
                
            clients_weight = [clients_participations[i] for i in active_clients]

            # Local Training
            if task_id == 0:
                if n_round == 0:
                    clients_prototype, clients_prototype_var = train_pertask(
                        [models_list[i] for i in active_clients], [models_without_ddp[i] for i in active_clients], 
                        [original_models[i] for i in active_clients], criterion, [data_loaders[i] for i in active_clients],
                        [optimizers[i] for i in active_clients], [lr_schedulers[i] for i in active_clients],
                        device, [class_masks[i] for i in active_clients], task_id, args)
                else:
                    clients_prototype, clients_prototype_var = train_pertask_v2(
                        [models_list[i] for i in active_clients], [models_without_ddp[i] for i in active_clients], 
                        [original_models[i] for i in active_clients], criterion, [data_loaders[i] for i in active_clients],
                        [optimizers[i] for i in active_clients], [lr_schedulers[i] for i in active_clients],
                        device, [class_masks[i] for i in active_clients], task_id, global_prototype, global_prototype_var, args)
            else:
                if n_round == 0:
                    clients_prototype, clients_prototype_var = generate_prototype_only(
                        [models_list[i] for i in active_clients], [models_without_ddp[i] for i in active_clients], 
                        [original_models[i] for i in active_clients], criterion, [data_loaders[i] for i in active_clients],
                        [optimizers[i] for i in active_clients], [lr_schedulers[i] for i in active_clients],
                        device, [class_masks[i] for i in active_clients], task_id, args)
                    global_prototype, global_prototype_var = FedWeightedAvgPrototype(clients_prototype,clients_prototype_var,clients_weight,task_id,args)
                    for k in global_prototype.keys():
                        all_global_prototype[k] = global_prototype[k]
                        all_global_prototype_var[k] = global_prototype_var[k]

                clients_prototype, clients_prototype_var = train_fs_pertask_v2(
                    [models_list[i] for i in active_clients], [models_without_ddp[i] for i in active_clients], 
                    [original_models[i] for i in active_clients], criterion, [data_loaders[i] for i in active_clients],
                    [optimizers[i] for i in active_clients], [lr_schedulers[i] for i in active_clients],
                    device, [class_masks[i] for i in active_clients], task_id, global_prototype, global_prototype_var, 
                    all_global_prototype, all_global_prototype_var, args)

            # Global Aggregation
            global_prototype, global_prototype_var = FedWeightedAvgPrototype(clients_prototype,clients_prototype_var,clients_weight,task_id,args)
            for k in global_prototype.keys():
                all_global_prototype[k] = global_prototype[k]
                all_global_prototype_var[k] = global_prototype_var[k]

            if n_round < (args.rounds_per_task - 1):
                FedWeightedAvg(server_model, [models_list[i] for i in clients_index], clients_weight, args.distributed)
            else:
                if task_id == 0:
                    FedWeightedAvgWithHead(server_model, [models_list[i] for i in clients_index], clients_weight, args.distributed)
                else:
                    FedWeightedAvg(server_model, [models_list[i] for i in clients_index], clients_weight, args.distributed)

            all_time_round += 1

            # Save Checkpoint every round
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_task{task_id}_round{n_round}.pth')
            save_checkpoint({
                'task_id': task_id,
                'n_round': n_round,
                'all_time_round': all_time_round,
                'state_dict': server_model_without_ddp.state_dict(),
                'optimizer': server_optimizer.state_dict(),
                'all_global_prototype': all_global_prototype,
                'all_global_prototype_var': all_global_prototype_var,
                'fixed_FC_dict': fixed_FC_dict,
                'fixed_FC_dict2': fixed_FC_dict2,
            }, False, args.output_dir, filename=f'checkpoint_latest.pth')
            print(f"Checkpoint saved to {args.output_dir}/checkpoint_latest.pth")

        # End of Task logic
        FedDistribute(server_model, models_list, args.distributed)
        if task_id == 0:
            fixed_FC_dict = copy.deepcopy(server_model.head.state_dict())
            fixed_FC_dict2 = copy.deepcopy(server_model_without_ddp.head.state_dict())
            server_optimizer = create_optimizer(args, server_model_without_ddp)
        else:
            server_model.head.load_state_dict(fixed_FC_dict)
            server_model_without_ddp.head.load_state_dict(fixed_FC_dict2)

        evaluate_server_global_model3(server_model, server_model_without_ddp, original_model,
                            criterion, data_loaders[0], server_optimizer, None,
                            device, None, task_id, test_sizes, 
                            all_global_prototype, all_global_prototype_var, args)

    total_time = time.time() - start_time
    print(f"Total training time: {str(datetime.timedelta(seconds=int(total_time)))}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('FFSCIL Enhanced Training and Evaluation')
    parser.add_argument('--output_dir', default='./output_enhanced/', type=str)
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--batch_size_override', default=None, type=int, help='Override batch size (e.g. 8192)')
    parser.add_argument('--rounds_per_task_override', default=None, type=int, help='Override rounds per task (e.g. 30)')
    parser.add_argument('--eval', default=False, type=bool, help='Run in evaluation mode')

    # Parse known args to find the config name
    known_args, remaining = parser.parse_known_args()
    
    # The config name should be the first item in remaining
    if not remaining:
        print("Error: No config specified. Choose from: ffscil_cifar100_9tasks_60bases_5ways, ffscil_imagenetsubset_9tasks_60bases_5ways, ffscil_cub200_11tasks_100bases_10ways")
        sys.exit(1)
    
    config_name = remaining[0]
    subparser = parser.add_subparsers(dest='subparser_name')

    # Load appropriate config parser
    if config_name == 'ffscil_cifar100_9tasks_60bases_5ways':
        from configs.ffscil_cifar100_9tasks_60bases_5ways import get_args_parser
        config_parser = subparser.add_parser('ffscil_cifar100_9tasks_60bases_5ways')
    elif config_name == 'ffscil_imagenetsubset_9tasks_60bases_5ways':
        from configs.ffscil_imagenetsubset_9tasks_60bases_5ways import get_args_parser
        config_parser = subparser.add_parser('ffscil_imagenetsubset_9tasks_60bases_5ways')
    elif config_name == 'ffscil_cub200_11tasks_100bases_10ways':
        from configs.ffscil_cub200_11tasks_100bases_10ways import get_args_parser
        config_parser = subparser.add_parser('ffscil_cub200_11tasks_100bases_10ways')
    elif config_name == 'ffscil_cicio23_cnn1d':
        from configs.ffscil_cicio23_cnn1d import get_args_parser
        config_parser = subparser.add_parser('ffscil_cicio23_cnn1d')
    else:
        # Generic parser if name doesn't match predefined ones but we still want to try
        from configs.ffscil_cifar100_9tasks_60bases_5ways import get_args_parser
        config_parser = subparser.add_parser(config_name)
        
    get_args_parser(config_parser)
    
    args = parser.parse_args()
    if args.rounds_per_task_override:
        args.rounds_per_task = args.rounds_per_task_override
    if args.output_dir: Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
