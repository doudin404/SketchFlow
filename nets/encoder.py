import re
import math
import unicodedata
from collections import Counter
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, CLIPTextModel,CLIPTextModelWithProjection
from transformers import CLIPProcessor, CLIPModel
import numpy as np

class PointEncoder(nn.Module):
    """
    SIREN点编码器。仅对输入点的前n=2维（如位置x,y）进行SIREN编码，后面的特征（如非位置信息）保持不变。
    输入: (B, L, F) -> 输出: (B, L, out_dim)
    实际输出维度: out_dim = siren_dim + (F-n_pos)，但 out_dim 参数指最终输出维度
    """
    def __init__(self, in_dim: int, out_dim: int = 64, n_pos: int = 2, omega: float = 30.0):
        super().__init__()
        self.n_pos = n_pos
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.omega = omega
        # SIREN编码器仅作用于前n_pos维
        self.siren_dim = max(out_dim - (in_dim - n_pos), 1)
        self.linear = nn.Linear(n_pos, self.siren_dim)
        self.init_siren()

    def init_siren(self):
        # SIREN初始化
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.n_pos, 1 / self.n_pos)
            self.linear.bias.uniform_(-1 / self.n_pos, 1 / self.n_pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        pos = x[..., :self.n_pos]  # (B, L, n_pos)
        rest = x[..., self.n_pos:] # (B, L, F-n_pos)
        siren_feat = torch.sin(self.omega * self.linear(pos))  # (B, L, siren_dim)
        if rest.shape[-1] > 0:
            out = torch.cat([siren_feat, rest], dim=-1)
        else:
            out = siren_feat
        return out
    



class TextCondSeqEncoder(nn.Module):
    """
    句子 -> (token隐藏态序列, attention_mask)
    - 返回形状: hidden_states [B, L, D], attn_mask [B, L]  (True=保留/可用)
    - 冻结参数，部署友好；需要微调时 set trainable=True
    """
    def __init__(self,
                 model_name: str = "openai/clip-vit-large-patch14",
                 max_length: int | None = None,
                 trainable: bool = False,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.model_name = model_name

        # 判断是否是本地路径
        model_path = Path(model_name)
        if model_path.exists() and model_path.is_dir():
            local = True
        else:
            local = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=local
        )

        if "clip" in model_name.lower():
            self.encoder = CLIPTextModel.from_pretrained(
                model_name,
                local_files_only=local
            )
            self.output_key = "last_hidden_state"
        else:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                local_files_only=local
            )
            self.output_key = "last_hidden_state"

        self.max_length = max_length or getattr(self.tokenizer, "model_max_length", 77)
        self.dtype = dtype
        self.encoder.eval()
        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.encoder.to(dtype=self.dtype)

    @torch.no_grad()
    def encode(self, texts: list[str]):
        """
        输入: 文本列表
        输出: hidden_states [B,L,D], attn_mask [B,L]  (bool, True=有效)
        """
        device = next(self.encoder.parameters()).device
        toks = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        self.encoder.to(device)

        out = self.encoder(**toks)
        H = out[self.output_key]                     # [B,L,D]
        attn_mask = toks.get("attention_mask", torch.ones(H.shape[:2], device=device)).bool()
        return H.to(self.dtype), attn_mask

    def forward(self, texts: list[str]):
        device = next(self.encoder.parameters()).device
        toks = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        self.encoder.to(device)
        out = self.encoder(**toks)
        H = out[self.output_key]
        attn_mask = toks.get("attention_mask", torch.ones(H.shape[:2], device=device)).bool()
        return H, attn_mask


class TextCondEncoder(nn.Module):
    """
    句子 -> 全局文本编码 (CLIP 对比空间)
    - 返回: (text_embeds [B, D], None)
    - 冻结参数默认 True；需要微调时传 trainable=True
    """
    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 max_length: Optional[int] = None,
                 trainable: bool = False,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.model_name = model_name

        model_path = Path(model_name)
        local = model_path.exists() and model_path.is_dir()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, local_files_only=local
        )

        if "clip" in model_name.lower():
            # 直接拿 CLIP 对比空间的全局向量
            self.encoder = CLIPTextModelWithProjection.from_pretrained(
                model_name, local_files_only=local
            )
            self._is_clip = True
            self.embed_dim = self.encoder.config.projection_dim  # e.g., 512 或 768
        else:
            # 非 CLIP：退化为 [CLS]/pooler 的全局表示
            self.encoder = AutoModel.from_pretrained(
                model_name, local_files_only=local
            )
            self._is_clip = False
            self.embed_dim = getattr(self.encoder.config, "hidden_size", None)

        self.max_length = max_length or getattr(self.tokenizer, "model_max_length", 77)
        self.dtype = dtype

        self.encoder.eval()
        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.encoder.to(dtype=self.dtype)

    @torch.no_grad()
    def encode(self, texts: List[str]) -> Tuple[torch.Tensor, None]:
        """
        输入: 文本列表
        输出: (text_embeds [B, D], None)
        """
        device = next(self.encoder.parameters()).device
        toks = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        self.encoder.to(device)

        out = self.encoder(**toks)

        if self._is_clip:
            # CLIP 最终投影后的全局编码（对比空间）
            embeds = out.text_embeds  # [B, D]
        else:
            # 退化方案：优先 pooler_output，否则用 CLS 向量
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                embeds = out.pooler_output  # [B, D]
            else:
                last = out.last_hidden_state  # [B, L, D]
                embeds = last[:, 0]          # CLS

        return embeds.to(self.dtype)

    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, None]:
        return self.encode(texts)
    

class ConditionEncoder:
    def __init__(self, clip_model_name: str, text_encoder: TextCondEncoder, device: str):
        # 加载CLIP模型和处理器
        self.device = device
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.text_encoder = text_encoder.to(device)

    def encode(self, text: Optional[str] = None, image: Optional[torch.Tensor] = None, second_text: Optional[str] = None, second_image: Optional[torch.Tensor] = None, B: int = 1) -> torch.Tensor:
        """
        根据传入的条件（文本或图片）生成B份条件编码。如果有第二个条件，则自动进行插值。
        
        :param text: 输入文本（可选）
        :param image: 输入图片（可选）
        :param second_text: 第二个文本条件（可选）
        :param second_image: 第二个图片条件（可选）
        :param B: 生成的样本数
        :return: 返回条件编码
        """
        cond_vec = None

        # 处理第一个条件
        if text:
            # 使用CLIP模型对文本进行编码
            inputs = self.clip_processor(text=[text] * B, return_tensors="pt", padding=True).to(self.device)
            cond_vec = self.clip_model.get_text_features(**inputs)
        elif image:
            # 使用CLIP模型对图像进行编码
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            pixel = inputs['pixel_values'].repeat(B, 1, 1, 1)  # 扩展为B个样本
            # 只取 image_embeds，保持与 CLIP 原始 512 维一致
            feats = self.clip_model.vision_model(pixel).pooler_output  # [B, 768]
            # 将 768 线性映射到官方对齐的 512 维投影空间，与 image_embeds 一致
            cond_vec = self.clip_model.visual_projection(feats)            # [B, 512]
        
        # 如果有第二个条件（文本或图片），则进行插值
        if second_text or second_image:
            # 获取第二个条件的编码
            second_cond_vec = None
            if second_text:
                inputs = self.clip_processor(text=[second_text] * B, return_tensors="pt", padding=True).to(self.device)
                second_cond_vec = self.clip_model.get_text_features(**inputs)
            elif second_image:
                inputs = self.clip_processor(images=second_image, return_tensors="pt").to(self.device)
                image_input = inputs['pixel_values'].repeat(B, 1, 1, 1)  # 扩展为B个样本
                second_cond_vec = self.clip_model.get_image_features(pixel_values=image_input)

            # 在这两个条件之间做线性插值
            alpha = torch.linspace(0, 1, B).unsqueeze(1).to(self.device)  # alpha 是插值因子
            cond_vec = (1 - alpha) * cond_vec + alpha * second_cond_vec
        
        # 如果到这里仍然没有任何条件向量，则用空文本编码一个默认向量，避免返回 None 导致出错
        if cond_vec is None:
            inputs = self.clip_processor(text=[""] * B, return_tensors="pt", padding=True).to(self.device)
            cond_vec = self.clip_model.get_text_features(**inputs)

        return cond_vec