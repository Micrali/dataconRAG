import numpy as np
from typing import List, Dict, Any, Tuple
import langdetect
from embeeder import VectorEmbedder  # 假设原有中文向量化模型
from sentence_transformers import SentenceTransformer  # 引入英文专用库
from qwen_model import AnchorExtractor, SentenceExpander
from chat import askanythingLLM

class EnglishVectorEmbedder:
    """英文专用向量化模型（基于sentence-transformers）"""
    def __init__(self):
        # 选择英文优化模型：all-MiniLM-L6-v2（轻量高效，适合英文语义匹配）
        # 其他可选模型：all-mpnet-base-v2（精度更高）、paraphrase-MiniLM-L6-v2（专注短语匹配）
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def encode(self, text: str) -> np.ndarray:
        """将英文文本编码为向量"""
        return self.model.encode([text])  # 返回形状为 (1, 384) 的向量
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        return self.model.similarity(vec1, vec2).item()  # 直接使用模型内置的相似度计算


class CacheManager:
    """缓存管理器 - 支持中英文分别处理"""

    def __init__(self, 
                 qwen_model: Any,
                 similarity_threshold: float = 0.8,
                 overlap_ratio: float = 0.2):
        """初始化缓存管理器"""
        # 初始化向量化模型（中英文分离）
        self.embedder = VectorEmbedder()  # 原有中文模型
        self.english_embedder = EnglishVectorEmbedder()  # 英文专用模型

        # 初始化锚点提取器和句子扩展器
        self.anchor_extractor = AnchorExtractor(qwen_model)
        self.sentence_expander = SentenceExpander(qwen_model, overlap_ratio)

        # 初始化RAG客户端
        self.rag_client = askanythingLLM()

        # 缓存设置
        self.similarity_threshold = similarity_threshold

        # 初始化缓存（短期通用，长期分中英文）
        self.short_term_anchor_cache = []  # [(text, vector, lang), ...]
        self.long_term_anchor_cache = []   # 中文长期锚点
        self.long_term_anchor_cache_en = []  # 英文长期锚点

        self.short_term_sentence_cache = []  # [(text, vector, lang), ...]
        self.long_term_sentence_cache = []   # 中文长期语句
        self.long_term_sentence_cache_en = []  # 英文长期语句

        # 存储RAG查询结果
        self.rag_results = []

    def _detect_language(self, text: str) -> str:
        """检测文本语言（中文返回'zh'，英文返回'en'）"""
        try:
            lang = langdetect.detect(text)
            return 'zh' if lang.startswith('zh') else 'en' if lang == 'en' else 'zh'
        except:
            return 'zh'  # 异常时默认中文

    def _get_embedder(self, lang: str):
        """根据语言选择向量化模型"""
        return self.english_embedder if lang == 'en' else self.embedder

    def _get_long_term_cache(self, cache_type: str, lang: str):
        """根据类型和语言选择长期缓存"""
        if cache_type == 'anchor':
            return self.long_term_anchor_cache_en if lang == 'en' else self.long_term_anchor_cache
        else:  # sentence
            return self.long_term_sentence_cache_en if lang == 'en' else self.long_term_sentence_cache

    def add_to_anchor_cache(self, text: str) -> bool:
        """添加锚点到缓存（自动区分语言）"""
        lang = self._detect_language(text)
        embedder = self._get_embedder(lang)
        vector = embedder.encode(text)[0]  # 获取向量

        # 检查与同语言长期缓存的相似度
        long_term_cache = self._get_long_term_cache('anchor', lang)
        if long_term_cache:
            max_similarity = self._get_max_similarity(vector, long_term_cache, embedder)
            if max_similarity > self.similarity_threshold:
                return False  # 相似度过高，不添加

        # 添加到缓存
        self.short_term_anchor_cache.append((text, vector, lang))
        long_term_cache.append((text, vector))
        return True

    def add_to_sentence_cache(self, text: str) -> bool:
        """添加语句到缓存（自动区分语言）"""
        lang = self._detect_language(text)
        embedder = self._get_embedder(lang)
        vector = embedder.encode(text)[0]

        # 检查与同语言长期缓存的相似度
        long_term_cache = self._get_long_term_cache('sentence', lang)
        if long_term_cache:
            max_similarity = self._get_max_similarity(vector, long_term_cache, embedder)
            if max_similarity > self.similarity_threshold:
                return False

        # 添加到缓存
        self.short_term_sentence_cache.append((text, vector, lang))
        long_term_cache.append((text, vector))
        return True

    def _get_max_similarity(self, query_vector: np.ndarray, cache: List[Tuple[str, np.ndarray]], embedder) -> float:
        """计算与缓存中向量的最大相似度"""
        max_sim = 0.0
        for _, cached_vec in cache:
            sim = embedder.cosine_similarity(query_vector, cached_vec)
            max_sim = max(max_sim, sim)
        return max_sim

    def generate_and_cache_anchors(self, text: str, max_anchors: int = 5) -> List[str]:
        """生成锚点并缓存"""
        anchors = self.anchor_extractor.extract_anchors_intelligent(text, max_anchors)
        return [a for a in anchors if self.add_to_anchor_cache(a)]

    def generate_and_cache_queries(self, chunk: str) -> Dict[str, List[str]]:
        """生成查询并缓存"""
        queries_dict = self.sentence_expander.generate_diversified_queries(chunk)
        for q_type, queries in queries_dict.items():
            [self.add_to_sentence_cache(q) for q in queries]
        return queries_dict

    def query_short_term_cache(self) -> List[Dict[str, Any]]:
        """查询短期缓存并获取RAG结果"""
        results = []
        # 处理锚点缓存
        for text, _, lang in self.short_term_anchor_cache:
            rag_res = self.rag_client.query(text)
            results.append({
                'type': 'anchor', 'language': lang,
                'query': text, 'response': rag_res.get('output', '')
            })
        # 处理语句缓存
        for text, _, lang in self.short_term_sentence_cache:
            rag_res = self.rag_client.query(text)
            results.append({
                'type': 'sentence', 'language': lang,
                'query': text, 'response': rag_res.get('output', '')
            })
        self.rag_results.extend(results)
        return results

    def clear_short_term_cache(self):
        """清空短期缓存"""
        self.short_term_anchor_cache.clear()
        self.short_term_sentence_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            'short_term_anchors': len(self.short_term_anchor_cache),
            'long_term_anchors_zh': len(self.long_term_anchor_cache),
            'long_term_anchors_en': len(self.long_term_anchor_cache_en),
            'short_term_sentences': len(self.short_term_sentence_cache),
            'long_term_sentences_zh': len(self.long_term_sentence_cache),
            'long_term_sentences_en': len(self.long_term_sentence_cache_en),
            'rag_results': len(self.rag_results)
        }

    def process_text_comprehensive(self, text: str) -> Dict[str, Any]:
        """综合处理文本"""
        anchors = self.generate_and_cache_anchors(text)
        queries = self.generate_and_cache_queries(text)
        return {
            'anchors_generated': anchors,
            'queries_generated': queries,
            'cache_stats': self.get_cache_stats()
        }


# 使用示例
if __name__ == "__main__":
    from qwen_model import QwenModel
    
    # 初始化模型和缓存管理器
    qwen_model = QwenModel()
    cache_manager = CacheManager(qwen_model, similarity_threshold=0.75)
    
    # 测试英文文本
    test_text_en = """
    Artificial intelligence (AI) is the simulation of human intelligence processes by machines, 
    especially computer systems. These processes include learning, reasoning, and self-correction.
    """
    results_en = cache_manager.process_text_comprehensive(test_text_en)
    print("英文处理结果：")
    print("生成的锚点：", results_en['anchors_generated'])
    print("缓存统计：", results_en['cache_stats'])
    
    # 测试中文文本
    test_text_zh = """
    人工智能（AI）是机器模拟人类智能过程的技术，尤其指计算机系统。这些过程包括学习、推理和自我修正。
    """
    results_zh = cache_manager.process_text_comprehensive(test_text_zh)
    print("\n中文处理结果：")
    print("生成的锚点：", results_zh['anchors_generated'])
    print("缓存统计：", results_zh['cache_stats'])
