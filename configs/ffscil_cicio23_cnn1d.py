import argparse

def get_args_parser(subparser):
    subparser.add_argument('--batch-size', default=8192, type=int, help='Batch size per device')
    subparser.add_argument('--epochs', default=1, type=int) # Base task epochs
    subparser.add_argument('--fs_epochs', default=1, type=int) # Few-shot task epochs
    subparser.add_argument('--num_tasks', default=6, type=int)
    subparser.add_argument('--nb_classes', default=36, type=int)
    subparser.add_argument('--num_clients', default=100, type=int)  # bo data '100 client'
    subparser.add_argument('--local_clients', default=100, type=int)  # so client tham gia moi round (giam de chay nhanh hon)
    subparser.add_argument('--rounds_per_task', default=30, type=int)
    subparser.add_argument('--base_classes', default=6, type=int)
    subparser.add_argument('--fs_classes', default=6, type=int)
    subparser.add_argument('--fs_shots', default=5, type=int)
    subparser.add_argument('--available_base_classes', default=6, type=int)
    subparser.add_argument('--available_fs_classes', default=6, type=int)

    # Model parameters
    subparser.add_argument('--model', default='cnn1d_prompt', type=str, help='Name of model to train')
    subparser.add_argument('--input-size', default=33, type=int, help='input size of IoT features')
    subparser.add_argument('--pretrained', action='store_true', default=False)
    subparser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    subparser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # Optimizer parameters
    subparser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    subparser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    subparser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    subparser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    subparser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    subparser.add_argument('--clip-grad', type=float, default=1.0, help='Clip gradient norm')
    subparser.add_argument('--reproducibility', action='store_false', default=True, help='switch off reproducibility set seed to None')
    subparser.add_argument('--seed', default=42, type=int)

    # Learning rate and scheduler parameters
    subparser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    subparser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
    subparser.add_argument('--lr_fs', type=float, default=0.001, metavar='LR', help='learning rate for FS tasks')
    subparser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    subparser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    subparser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    subparser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    subparser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    subparser.add_argument('--warmup-epochs', type=int, default=0, metavar='N', help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    subparser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor (default: 0.4)')
    subparser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)'),
    subparser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    subparser.add_argument('--train-interpolation', type=str, default='bicubic', help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    subparser.add_argument('--data-path', default=r'C:\FederatedLearning\FL\core\data_split', type=str, help='dataset path')
    subparser.add_argument('--dataset', default='cic_iot23', type=str, help='dataset name')
    subparser.add_argument('--output_dir', default='./output_iot_real', help='path where to save, empty for no saving')
    subparser.add_argument('--device', default='cuda', help='device to use for training / testing')
    subparser.add_argument('--num_workers', default=4, type=int)
    subparser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (faster) transfer to GPU.')
    subparser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    subparser.add_argument('--shuffle', default=False, help='shuffle the data order')
    
    # Prompt parameters
    subparser.add_argument('--size', default=10, type=int, help='prompt pool size')
    subparser.add_argument('--length', default=5, type=int, help='prompt length')
    subparser.add_argument('--top_k', default=1, type=int, help='top k prompts')
    subparser.add_argument('--batchwise_prompt', action='store_true', default=False)
    subparser.add_argument('--prompt_key', action='store_true', default=True)
    subparser.add_argument('--prompt_key_init', default='uniform', type=str)
    subparser.add_argument('--prompt_pool', action='store_true', default=True)
    subparser.add_argument('--shared_prompt_pool', default=True, type=bool)
    subparser.add_argument('--shared_prompt_key', default=False, type=bool)
    subparser.add_argument('--embedding_key', default='cls', type=str)
    subparser.add_argument('--head_type', default='prompt', type=str)
    subparser.add_argument('--use_prompt_mask', action='store_true', default=False)
    subparser.add_argument('--use_e_prompt', action='store_true', default=True)
    subparser.add_argument('--use_g_prompt', action='store_true', default=False)
    subparser.add_argument('--g_prompt_length', default=5, type=int)
    subparser.add_argument('--g_prompt_layer_idx', default=[], type=int, nargs='+')
    subparser.add_argument('--use_prefix_tune_for_g_prompt', action='store_true', default=False)
    subparser.add_argument('--e_prompt_layer_idx', default=[0, 1, 2, 3, 4], type=int, nargs='+')
    subparser.add_argument('--use_prefix_tune_for_e_prompt', action='store_true', default=False)
    subparser.add_argument('--same_key_value', action='store_true', default=False)
    subparser.add_argument('--pull_constraint', default=False)
    
    # Continual Learning parameters
    subparser.add_argument('--task_inc', action='store_true', default=True)
    subparser.add_argument('--train_mask', action='store_true', default=True)
    subparser.add_argument('--freeze', action='store_true', default=False, help='Freeze backbone')
    subparser.add_argument('--reinit_optimizer', type=bool, default=True)
    subparser.add_argument('--print_freq', type=int, default=10)
