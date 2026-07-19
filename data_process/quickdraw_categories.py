from __future__ import annotations
import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple
from tqdm import tqdm
from utils.cache import cache_filename
from data_process.quickdraw_sketch import quickdraw_cache_path
from utils.clip_manager import get_clip_manager

class QuickDrawCategoryCache:
    """
    Manages a cache for CLIP text embeddings of QuickDraw category names.
    Encodes a list of category names, saves the embeddings to a .npy file,
    and creates a dictionary mapping names to their index in the .npy file.
    """
    def __init__(self, 
                 cache_dir: Union[str, Path], 
                 device: str = "cpu", 
                 model_name: str = "ViT-B-32", 
                 pretrained: str = "openai"):
        
        self.cache_dir = Path(cache_dir)
        self.device = torch.device(device)
        self.model_name = model_name
        self.pretrained = pretrained
        
        self.emb_path, self.idx_map_path = qdraw_cat_cache_path(
            cache_dir, model_name
        )

        self.embeddings: Optional[np.ndarray] = None
        self.idx_map: Dict[str, int] = {}
        
        # Initialize CLIPManager
        self.clip_manager = get_clip_manager(
            model_name=self.model_name,
            pretrained=self.pretrained,
            device=self.device
        )

        if self.emb_path.exists() and self.idx_map_path.exists():
            print(f"Loading QuickDraw category cache from {self.cache_dir}")
            self.embeddings = np.load(self.emb_path)
            self.idx_map = torch.load(self.idx_map_path, weights_only=False)
        else:
            print("QuickDraw category cache not found.")

    def is_built(self) -> bool:
        return self.embeddings is not None and bool(self.idx_map)

    @torch.no_grad()
    def build(self, category_names: List[str], batch_size: int = 64):
        unique_names = sorted(list(set(category_names)))
        
        if self.is_built() and set(unique_names) == set(self.idx_map.keys()):
            print("Cache is already built and contains all requested names.")
            return

        print("Building QuickDraw category cache...")
        
        all_embs = []

        for i in tqdm(range(0, len(unique_names), batch_size), desc="Encoding categories"):
            batch_names = unique_names[i:i+batch_size]
            # Use a prompt template for better sketch-related embeddings
            prompts = [f"{name}" for name in batch_names]
            # The CLIPManager handles tokenization, encoding, and normalization
            text_features = self.clip_manager.encode_text(prompts)
            all_embs.append(text_features.cpu().numpy().astype(np.float16))

        self.embeddings = np.concatenate(all_embs, axis=0)
        self.idx_map = {name: i for i, name in enumerate(unique_names)}

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        pid = os.getpid()
        tmp_emb_path = self.emb_path.parent / f"{self.emb_path.stem}.{pid}.tmp.npy"
        tmp_idx_map_path = self.idx_map_path.with_suffix(f"{self.idx_map_path.suffix}.{pid}.tmp")
        
        try:
            np.save(tmp_emb_path, self.embeddings)
            torch.save(self.idx_map, tmp_idx_map_path)
            
            tmp_emb_path.rename(self.emb_path)
            tmp_idx_map_path.rename(self.idx_map_path)
        finally:
            if tmp_emb_path.exists():
                tmp_emb_path.unlink()
            if tmp_idx_map_path.exists():
                tmp_idx_map_path.unlink()

        print(f"Cache with {len(self.idx_map)} categories built and saved in {self.cache_dir}")

    def get_embedding(self, category_name: str) -> np.ndarray:
        if not self.is_built():
            raise RuntimeError("Cache is not built or loaded. Call .build() or check cache paths.")
        
        idx = self.idx_map.get(category_name)
        if idx is None:
            raise KeyError(f"Category '{category_name}' not found in cache.")
        
        return self.embeddings[idx]

    def get_embeddings_tensor(self, category_names: List[str], device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Gets embeddings for a list of names and stacks them into a tensor."""
        embs = np.stack([self.get_embedding(name) for name in category_names])
        tensor = torch.from_numpy(embs)
        if device:
            tensor = tensor.to(torch.device(device))
        return tensor


def qdraw_cat_cache_path(
    cache_dir: Union[str, Path],
    model_name: str = "ViT-B-32",

) -> Tuple[Path, Path]:
    """Generates the standard paths for QuickDraw category cache files."""
    cache_root = Path(cache_dir)
    
    cache_name = cache_filename(
        "qdraw_cat",
        model_name,
        prefix_count=3,
    )
    
    emb_path = cache_root / f"{cache_name}_emb.npy"
    idx_map_path = cache_root / f"{cache_name}_idx.pt"
    
    return emb_path, idx_map_path



def build_qdraw_cat_cache(
    cache_dir: Union[str, Path],
    category_names: Optional[List[str]] = None,
    quickdraw_data_dir: Optional[Union[str, Path]] = None,
    qdraw_n_points: int = 256,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",#"openai",
    batch_size: int = 64,
    device: str = "cpu",
    force_rebuild: bool = False,
):
    """
    Builds and caches CLIP text embeddings for QuickDraw category names.
    The category names can be provided directly or extracted from a QuickDraw dataset directory.
    """
    cache_dir = Path(cache_dir)

    # 1. Proceed with caching logic
    emb_path, idx_map_path = qdraw_cat_cache_path(
        cache_dir=cache_dir,
        model_name=model_name,
    )

    if not force_rebuild and emb_path.exists() and idx_map_path.exists():
        print(f"QuickDraw category cache already exists and is valid in {cache_dir}")
        return emb_path, idx_map_path


    # 2. Determine the list of category names
    names_to_process: List[str]
    if category_names is not None and quickdraw_data_dir is None:
        names_to_process = category_names
    elif quickdraw_data_dir is not None and category_names is None:
        print(f"Extracting category names from {quickdraw_data_dir}...")
        _, labels_path = quickdraw_cache_path(
            input_path=quickdraw_data_dir,
            n_points=qdraw_n_points,
            split="valid",
            cache_dir=cache_dir
        )
        labels = np.load(labels_path, mmap_mode="r")
        names_to_process = sorted(list(set(labels)))
        print(f"Found {len(names_to_process)} unique categories.")
    else:
        raise ValueError("Exactly one of `category_names` or `quickdraw_data_dir` must be provided.")


    cache = QuickDrawCategoryCache(
        cache_dir=cache_dir,
        device=device,
        model_name=model_name,
        pretrained=pretrained,
    )
    
    cache.build(category_names=names_to_process, batch_size=batch_size)
    
    return emb_path, idx_map_path