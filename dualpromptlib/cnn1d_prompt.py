import torch
import torch.nn as nn
import torch.nn.functional as F

class EPrompt(nn.Module):
    def __init__(self, pool_size, prompt_length, embed_dim):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(pool_size, prompt_length, embed_dim))
        self.prompt_key = nn.Parameter(torch.randn(pool_size, embed_dim))

class CNN1D_Prompt(nn.Module):
    def __init__(self, input_dim=33, num_classes=36, embed_dim=128, pool_size=10, prompt_length=5):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # 1D-CNN Backbone
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # E-Prompt Pool for CNN
        self.e_prompt = EPrompt(pool_size, prompt_length, embed_dim)
        
        # FC Head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # [FIX] Khai báo để tương thích với Few-Shot engine của FFSCIL
        self.entered_fs_task = False

    def init_fs_head(self, available_fs_classes):
        # Không cần chuyển đổi head cho CNN1D Prompt (sử dụng forward_with_proto)
        pass
        
    def forward_features(self, x, task_id=-1, cls_features=None, train=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        
        with torch.no_grad():
            similarity = F.cosine_similarity(x.unsqueeze(1), self.e_prompt.prompt_key.unsqueeze(0), dim=-1)
            prompt_idx = torch.argmax(similarity, dim=-1)
        
        selected_prompts = self.e_prompt.prompt[prompt_idx]
        prompt_influence = selected_prompts.mean(dim=1)
        x_prompted = x + prompt_influence
        
        return {
            'pre_logits': x_prompted,
            'prompt_idx': prompt_idx.unsqueeze(1),
            'similarity': similarity
        }

    def forward_get_prelogits(self, x, task_id=-1, cls_features=None, train=False):
        res = self.forward_features(x, task_id, cls_features, train)
        return res['pre_logits']
        
    def forward_head_prelogits(self, res):
        return {'pre_logits': res['pre_logits']}

    def forward_head(self, res):
        logits = self.head(res['pre_logits'])
        return {'logits': logits}

    def forward(self, x, task_id=-1, cls_features=None, train=False):
        res = self.forward_features(x, task_id, cls_features, train)
        out = self.forward_head(res)
        out['pre_logits'] = res['pre_logits']
        out['prompt_idx'] = res['prompt_idx']
        return out

    def forward_with_proto(self, x, proto=None, task_id=-1, cls_features=None, train=False):
        res = self.forward_features(x, task_id, cls_features, train)
        logits = self.head(res['pre_logits'])
        if proto is not None:
            proto_logits = self.head(proto)
            logits = torch.cat([logits, proto_logits], dim=0)
        return {'logits': logits}

def create_cnn1d_prompt(input_dim=33, num_classes=36, **kwargs):
    model = CNN1D_Prompt(input_dim=input_dim, num_classes=num_classes)
    return model
