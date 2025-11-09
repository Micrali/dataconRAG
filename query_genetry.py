# cache_manager.py
import numpy as np
from typing import List, Dict, Any, Tuple
from embeeder import VectorEmbedder
from qwen_model import AnchorExtractor, SentenceExpander
from rag_client import RAGAttackClient
from optimizer import OptimizedDeduplication
import re

class CacheManager:
    """缓存管理器 - 管理短期和长期缓存"""
    
    def __init__(self, 
                 qwen_model: Any,
                 similarity_threshold: float = 0.85,
                 overlap_ratio: float = 0.2):
        """
        初始化缓存管理器
        
        Args:
            qwen_model: Qwen模型实例
            similarity_threshold: 相似度阈值，高于此值则舍去
            overlap_ratio: 重叠率
        """
        # 初始化向量化模型
        self.embedder = VectorEmbedder()
        
        # 初始化锚点提取器和句子扩展器
        #self.anchor_extractor = AnchorExtractor(qwen_model)
        self.sentence_expander = SentenceExpander(qwen_model, overlap_ratio)
        
        # 初始化RAG客户端
        self.rag_client = RAGAttackClient()
        
        # 缓存设置
        self.similarity_threshold = similarity_threshold
        
        # # 初始化缓存
        # self.short_term_anchor_cache = []  # 短期锚点缓存 [(text, vector), ...]
        # self.long_term_anchor_cache = []   # 长期锚点缓存 [(text, vector), ...]
        
        self.short_term_sentence_cache = []  # 短期语句缓存 [(text, vector), ...]
        self.long_term_sentence_cache = []   # 长期语句缓存 [(text, vector), ...]
        
        # 存储RAG查询结果
        self.rag_results = []
        
        self.deduplicator=OptimizedDeduplication()
        
    def split_text_by_punctuation(self, text: str) -> List[str]:
        """
        按照中文标点符号划分文本，确保每个句子至少25字
        
        Args:
            text: 输入文本
            
        Returns:
            划分后的句子列表
        """
        # 中文标点符号：句号、问号、感叹号、分号、逗号等
        chinese_punctuation = r'[。！？；，]'
        
        # 使用正则表达式分割文本
        initial_sentences = re.split(chinese_punctuation, text)
        
        # 过滤空字符串并去除前后空白
        initial_sentences = [s.strip() for s in initial_sentences if s.strip()]
        
        # 合并短句，确保每个句子至少20字
        merged_sentences = []
        current_sentence = ""
        
        for sentence in initial_sentences:
            # 如果当前句子为空，直接添加
            if not current_sentence:
                current_sentence = sentence
            else:
                # 尝试合并到当前句子
                current_sentence = current_sentence + "，" + sentence
                # 如果合并后仍然不足20字，继续合并
                if len(current_sentence) > 25:
                    merged_sentences.append(current_sentence)
                    current_sentence = ""
                    
        # 添加最后一个句子
        if current_sentence:
            merged_sentences.append(current_sentence)
        
        return merged_sentences

    # def add_to_anchor_cache(self, text: str) -> bool:
    #     """
    #     将锚点文本添加到缓存中
        
    #     Returns:
    #         bool: 是否成功添加（相似度检查通过）
    #     """
    #     # 向量化文本
    #     vector = self.embedder.encode(text)[0]
        
    #     # 检查与长期缓存的相似度
    #     if self.long_term_anchor_cache:
    #         max_similarity = self._get_max_similarity(vector, self.long_term_anchor_cache)
    #         if max_similarity > self.similarity_threshold:
    #             print(f"相似度过高，舍去\'{text}\'")
    #             return False  # 相似度过高，舍去
        
    #     # 添加到短期缓存和长期缓存
    #     self.short_term_anchor_cache.append((text, vector))
    #     self.long_term_anchor_cache.append((text, vector))
        
    #     return True
    
    def add_to_sentence_cache(self, text: str) -> bool:
        """
        将语句文本添加到缓存中
        
        Returns:
            bool: 是否成功添加（相似度检查通过）
        """
        #定期清理
        if len(self.deduplicator.exact_hashes)%50==0:
            self.deduplicator.cleanup_old_entries()

        # 1. 智能去重检查
        if self.deduplicator.is_duplicate(text):
            print(f"智能去重，舍去\'{text}\'")
            return False

        # 向量化文本
        vector = self.embedder.encode(text)[0]
        
        # 检查与长期缓存的相似度
        if self.long_term_sentence_cache:
            max_similarity = self._get_max_similarity(vector, self.long_term_sentence_cache)
            if max_similarity > self.similarity_threshold:
                print(f"相似度过高，舍去\'{text}\'")
                return False  # 相似度过高，舍去
        
        # 添加到短期缓存和长期缓存
        self.short_term_sentence_cache.append((text, vector))
        self.long_term_sentence_cache.append((text, vector))
        
        return True
    
    def _get_max_similarity(self, query_vector: np.ndarray, cache: List[Tuple[str, np.ndarray]]) -> float:
        """计算查询向量与缓存中所有向量的最大相似度"""
        max_similarity = 0.0
        for _, cached_vector in cache:
            similarity = self.embedder.cosine_similarity(query_vector, cached_vector)
            max_similarity = max(max_similarity, similarity)
        return max_similarity
    
    # def generate_and_cache_anchors(self, text: str, max_anchors: int = 5) -> List[str]:
    #     """生成锚点并添加到缓存"""
    #     anchors = self.anchor_extractor.extract_anchors_intelligent(text, 15)
        
    #     added_anchors = []
    #     for anchor in anchors:
    #         if self.add_to_anchor_cache(anchor):
    #             print("Select "+anchor)
    #             added_anchors.append(anchor)
        
    #     return added_anchors
    
    def generate_and_cache_queries(self, chunk: str) -> Dict[str, List[str]]:
        """生成多样化查询并添加到缓存"""
        queries_dict = self.sentence_expander.generate_diversified_queries(chunk)
        
        # 合并所有查询
        all_queries = []
        for query_type, queries in queries_dict.items():
            all_queries.extend(queries)
        
        # 添加到缓存
        added_queries = []
        for query in all_queries:
            if self.add_to_sentence_cache(query):
                print("Select "+query)
                added_queries.append(query)
        
        return queries_dict
    
    def query_short_term_cache(self):
        """查询短期缓存中的所有内容"""
        # short_anchor=[item[0] for item in self.short_term_anchor_cache]
        # anchor_results = self.rag_client.batch_query(short_anchor,1)
        short_sentence=[item[0] for item in self.short_term_sentence_cache]
        sentence_results=self.rag_client.batch_query(short_sentence,0.1)
        # 清空短期缓存（可选，根据需求决定）
        self.clear_short_term_cache()
        
        return sentence_results
    
    def clear_short_term_cache(self):
        """清空短期缓存"""
        #self.short_term_anchor_cache.clear()
        self.short_term_sentence_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            # 'short_term_anchors': len(self.short_term_anchor_cache),
            # 'long_term_anchors': len(self.long_term_anchor_cache),
            'short_term_sentences': len(self.short_term_sentence_cache),
            'long_term_sentences': len(self.long_term_sentence_cache)
        }
    
    def process_text_comprehensive(self, text: str) -> Dict[str, Any]:
        """
        综合处理文本：先分句，然后生成锚点、查询，并执行RAG查询
        
        Returns:
            包含所有结果的字典
        """
        # 先按照标点符号划分文本
        sentences_tmp = self.split_text_by_punctuation(text)

        print(sentences_tmp)
        sentences=[]
        for sentence in sentences_tmp:

            vector = self.embedder.encode(sentence)[0]


            # 检查与长期缓存的相似度
            if self.long_term_sentence_cache:
                max_similarity = self._get_max_similarity(vector, self.long_term_sentence_cache)
                if max_similarity < self.similarity_threshold:
                    self.long_term_sentence_cache.append((sentence,vector))
                    sentences.append(sentence)
                else:
                    print('舍弃')
            else:
                 self.long_term_sentence_cache.append((sentence,vector))
                 sentences.append(sentence)
        if len(sentences)>10:
            sentences=sentences[0:5]+sentences[-5:]
        print(f"将文本划分为 {len(sentences)} 个句子")
        print(sentences)
        # 存储所有生成的锚点和查询
        # all_anchors = []
        all_queries_dict = {
            'forward': [],
            'backward': [],
            'overlap': []
        }
        
        # 对每个句子分别处理
        for i, sentence in enumerate(sentences):
            print(f"处理第 {i+1} 个句子: {sentence[:50]}...")
            
                    
            # # 生成并缓存锚点
            # anchors = self.generate_and_cache_anchors(sentence)
            # all_anchors.extend(anchors)
            
            # 生成并缓存查询
            queries = self.generate_and_cache_queries(sentence)
            
            # 合并查询
            for query_type, query_list in queries.items():
                all_queries_dict[query_type].extend(query_list)
        
        # 执行RAG查询
        # anchor_results=[]
        sentence_results=[]
        sentence_result = self.query_short_term_cache()
        
        # 处理锚点结果
        # for anchor_t in anchor_result:
        #     anchor=anchor_t['output']
        #     vector = self.embedder.encode(anchor)[0]
        
        #     # 检查与长期缓存的相似度
        #     if self.long_term_sentence_cache:
        #         max_similarity = self._get_max_similarity(vector, self.long_term_sentence_cache)
        #         if max_similarity > self.similarity_threshold:
        #             print(f"anchor结果相似度过高，舍去\'{anchor[:50]}...\'")
        #         else :
        #             print(f'对anchor结果\'{anchor[:50]}...\'进行处理')
        #             anchor_results.append(anchor)

        # 处理句子结果
        for sentence_t in sentence_result:
            sentence=sentence_t['output']
            vector = self.embedder.encode(sentence)[0]
        
            # 检查与长期缓存的相似度
            if self.long_term_sentence_cache:
                max_similarity = self._get_max_similarity(vector, self.long_term_sentence_cache)
                if max_similarity > self.similarity_threshold:
                    print(f"sentence结果相似度过高，舍去\'{sentence[:50]}...\'")
                else :
                    print(f'对sentence结果\'{sentence[:50]}...\'进行处理')
                    sentence_results.append(sentence)
        
        return {
            # 'anchors_generated': all_anchors,
            # 'anchor_results': anchor_results,
            'sentence_results': sentence_results,
            'queries_generated': all_queries_dict,
            'cache_stats': self.get_cache_stats()
        }

if __name__ == "__main__":
    from qwen_model import QwenModel
    
    # 初始化Qwen模型
    qwen_model = QwenModel()
    
    # 初始化缓存管理器
    cache_manager = CacheManager(qwen_model, similarity_threshold=0.75)

    # 测试文本
    test_text = """
    老屋的门轴在风里生锈了，我推门时听见它发出悠长的叹息。檐角垂下的蛛网在夕阳里泛着金边，像母亲织了一半的毛衣。
    院中的槐树还在，只是枝干裂开的缝隙里，爬满了时间的青苔。父亲蹲在门槛上抽旱烟，烟锅里火星明灭，仿佛在数着归家的路。
    """
  
    # 综合处理文本
    results = cache_manager.process_text_comprehensive(test_text)
    
    # print("=== 生成的锚点 ===")
    # for anchor in results['anchors_generated']:
    #     print(f"- {anchor}")
    
    print("\n=== 生成的查询 ===")
    for query_type, queries in results['queries_generated'].items():
        print(f"{query_type}:")
        for query in queries:
            print(f"  - {query}")
    
    print("\n=== RAG查询结果 ===")
    # for i, result in enumerate(results['anchor_results']):
    #     print(f"{i+1}. {result}")
    for i, result in enumerate(results['sentence_results']):
        print(f"{i+1}. {result}")
    print(f"\n=== 缓存统计 ===")
    stats = results['cache_stats']
    for key, value in stats.items():
        print(f"{key}: {value}")