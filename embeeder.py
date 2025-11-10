import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union


class EnglishVectorEmbedder:
    """英文专用向量化模型（基于sentence-transformers）"""
    def __init__(self):
        # 选择英文优化模型：all-MiniLM-L6-v2（轻量高效，适合英文语义匹配）
        # 其他可选模型：all-mpnet-base-v2（精度更高）、paraphrase-MiniLM-L6-v2（专注短语匹配）
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """将英文文本编码为向量"""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts)  # 返回形状为 (1, 384) 的向量
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        return self.model.similarity(vec1, vec2).item()  # 直接使用模型内置的相似度计算

class VectorEmbedder:
    """本地向量化模型调用模块"""
    
    def __init__(self, model_name: str = 'shibing624/text2vec-base-chinese'):
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """将文本编码为向量"""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
