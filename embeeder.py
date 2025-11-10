import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union

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
    

# test=VectorEmbedder()
# texts=["暮色漫过山脊时，我看见了那个稻草人。它还在老屋院角站着，褪色的衣裳被风灌满，像一具被岁月遗忘的旧木偶。车轮碾过石板路的凹痕，碾碎了二十年前的蝉鸣。母亲掀开灶台的柴门，炊烟裹着腊肉香扑面而来，恍惚间我竟分不清是记忆在燃烧，还是现实的烟火。  ","堂屋的八仙桌上，搪瓷碗底凝着半盏茶垢。父亲的烟斗不知何时换了新式样的，却依旧蜷在竹椅扶手上。檐角铜铃被岁月磨得发亮，风起时竟发出孩童嬉笑的回响。我忽然想起，那年离家时也是这般暮色，母亲将晒干的艾草塞进我行囊，说能驱散异乡的寒气。"]
# print(test.cosine_similarity(test.encode(texts[0])[0],test.encode(texts[1])[0]))