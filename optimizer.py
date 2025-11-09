# 在 query_genetry.py 文件开头添加以下类
import hashlib
from collections import deque
import jieba

class OptimizedDeduplication:
    """优化的智能去重机制 - 专门针对短中文文本"""
    
    def __init__(self, max_history: int = 200):
        self.exact_hashes = set()  # 精确哈希
        self.recent_queries = deque(maxlen=max_history)  # 最近查询记录
        self.semantic_cache = {}  # 语义缓存
        
        # 初始化jieba，提高分词准确性
        jieba.initialize()
    
    def extract_key_phrases(self, text: str) -> list:
        """提取关键短语 - 针对短中文文本优化"""
        if len(text) < 5:
            return [text]
        
        # 使用jieba提取关键词
        words = jieba.cut(text)
        # 过滤单字和停用词
        key_phrases = []
        for word in words:
            if len(word) > 1 and word not in ['的', '了', '在', '是', '有', '和', '就', '都', '而', '及', '与', '这', '那']:
                key_phrases.append(word)
        
        return key_phrases if key_phrases else [text[:min(3, len(text))]]
    
    def calculate_semantic_fingerprint(self, text: str) -> str:
        """计算语义指纹 - 针对短文本优化"""
        if len(text) < 10:
            # 超短文本直接使用哈希
            return f"short_{hashlib.md5(text.encode()).hexdigest()[:8]}"
        
        # 提取关键短语
        key_phrases = self.extract_key_phrases(text)
        
        # 构建语义指纹
        if len(key_phrases) >= 2:
            # 如果有多个关键词，使用前两个
            fingerprint = f"{key_phrases[0]}_{key_phrases[1]}"
        else:
            # 只有一个关键词，结合文本长度和首字符
            fingerprint = f"{key_phrases[0]}_{len(text)}"
        
        return fingerprint
    
    def is_duplicate(self, text: str) -> bool:
        """判断是否重复 - 优化版"""
        # 1. 精确哈希检查
        exact_hash = hashlib.md5(text.encode()).hexdigest()
        if exact_hash in self.exact_hashes:
            return True
        
        # 2. 语义指纹检查
        semantic_fp = self.calculate_semantic_fingerprint(text)
        if semantic_fp in self.semantic_cache:
            # 检查时间衰减，避免永久阻塞相似查询
            count, timestamp = self.semantic_cache[semantic_fp]
            if count > 2:  # 相同模式出现多次
                return True
        
        # 3. 更新记录
        self.exact_hashes.add(exact_hash)
        self.semantic_cache[semantic_fp] = (
            self.semantic_cache.get(semantic_fp, (0, 0))[0] + 1,
            len(self.recent_queries)  # 使用队列长度作为简单时间戳
        )
        self.recent_queries.append(text)
        
        return False
    
    def cleanup_old_entries(self):
        """清理旧条目 - 防止内存无限增长"""
        if len(self.exact_hashes) > 400:
            # 简单清理：保留最近500个
            recent_hashes = set()
            for text in list(self.recent_queries)[-200:]:
                recent_hashes.add(hashlib.md5(text.encode()).hexdigest())
            self.exact_hashes = recent_hashes
            
            # 清理语义缓存
            old_fingerprints = set(self.semantic_cache.keys()) - set(
                self.calculate_semantic_fingerprint(text) for text in list(self.recent_queries)[-200:]
            )
            for fp in old_fingerprints:
                del self.semantic_cache[fp]

