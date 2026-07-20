# FFSCIL — Brief (Chat 6)

## Mục tiêu
Federated Few-Shot Class-Incremental Learning dùng prompt-based (DualPrompt / L2P). Baseline tham khảo.

## Dataset
Ảnh: CIFAR-100 (9 task, 60 base, 5-way), CUB-200 (11 task, 100 base, 10-way), miniImageNet/ImageNet-subset.
Config trong `configs/` và `dualpromptlib/configs/`.

## Cấu trúc
`dualpromptlib/` (DualPrompt engine: `fed_dualprompt_engine*.py`, `fed_pip_engine*.py`, ViT backbone),
`l2plib/` (L2P). Nhiều biến thể engine (2, 3, hetero).

## Entry & lệnh chạy (theo README)
Ví dụ CIFAR-100 5-shot:
```
python ttt1v_xpip_dualp_main.py ffscil_cifar100_9tasks_60bases_5ways \
  --rounds_per_task 10 --epochs 1 --fs_epochs 20 --seed 2023 --lr_fs 0.2 --fs_shots 5
```
(Entry code: `dualpromptlib/main.py` / `main2.py`; kiểm tra tên script thực tế trong repo.)

## Trạng thái
Baseline tham khảo cho hướng federated few-shot CIL (trên ảnh, không phải IDS).
Dùng để so phương pháp prompt-based với hướng adapter/prototype của AFSIC-IDS.
