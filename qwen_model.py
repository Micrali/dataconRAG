import torch
import re
from typing import List, Dict, Any
from langdetect import detect, LangDetectException
from transformers import AutoTokenizer, AutoModelForCausalLM

class QwenModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
       
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype='auto',
            device_map="auto"
        )
        print("Model loaded successfully!")
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7, 
                         system_set: str="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") -> str:
        """生成响应"""
        messages = [
            {"role": "system", "content": system_set },
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
    
    def batch_generate(self, prompts: List[str], max_length: int = 512) -> List[str]:
        """批量生成响应"""
        responses = []
        for prompt in prompts:
            response = self.generate_response(prompt, max_length)
            responses.append(response)
        return responses


class AnchorExtractor:
    """智能锚点提取器 - 支持中英文自动切换"""
    
    def __init__(self, qwen_model: QwenModel):
        self.model = qwen_model
        
    def _detect_language(self, text: str) -> str:
        """检测文本语言（中文返回'zh'，英文返回'en'）"""
        try:
            lang = detect(text)
            return 'zh' if lang.startswith('zh') else 'en' if lang == 'en' else 'zh'
        except LangDetectException:
            return 'zh'  # 异常时默认中文
        
    def extract_anchors_intelligent(self, text: str, max_anchors: int = 5) -> List[str]:
        """智能提取锚点（根据文本语言切换提示词）"""
        lang = self._detect_language(text)
        
        # 中文提示词
        if lang == 'zh':
            prompt = f"""
请从以下文本中提取{max_anchors}个重要的关键词或关键短语作为锚点。这些锚点应该能够代表文本的核心内容，并且适合用于生成相关的陈述语句。
注意，一定要检验你生成的关键词数量是否达标

文本内容：
{text}

请直接返回锚点列表，每个锚点用换行符分隔，不要添加任何解释，注意，一定要使用换行符'\n'进行分割，不得缺斤少两，一定要刚好生成{max_anchors}个关键词。
"""
        # 英文提示词
        else:
            prompt = f"""
Extract {max_anchors} important keywords or key phrases from the following text as anchors. These anchors should represent the core content of the text and be suitable for generating related statements.
Note: Ensure the number of generated keywords meets the requirement.

Text content:
{text}

Please directly return the list of anchors, with each anchor separated by a newline character. Do not add any explanations. Note that you must use the newline character '\n' for separation and generate exactly {max_anchors} keywords.
"""

        response = self.model.generate_response(prompt, max_length=200, temperature=0.3)
        anchors = self._parse_anchor_response(response, lang)
        return anchors[:max_anchors]

    
    def _parse_anchor_response(self, response: str, lang: str) -> List[str]:
        """解析模型返回的锚点（适配中英文）"""
        anchors = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # 移除编号和特殊字符
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•*]\s*', '', line)
            
            # 英文锚点长度放宽限制（英文单词通常更短）
            min_len = 1 if lang == 'en' else 1
            max_len = 60 if lang == 'en' else 50
            
            if line and min_len < len(line) < max_len:
                anchors.append(line)
        
        return anchors


class SentenceExpander:
    """智能句子扩展器 - 支持中英文自动切换"""

    def __init__(self, qwen_model: QwenModel, overlap_ratio):
        self.model = qwen_model
        self.overlap_ratio = overlap_ratio  # 重叠率
        
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        try:
            lang = detect(text)
            return 'zh' if lang.startswith('zh') else 'en' if lang == 'en' else 'zh'
        except LangDetectException:
            return 'zh'

    def split_text_by_punctuation(self, text: str) -> List[str]:
        """按照标点符号划分文本（适配中英文标点）"""
        lang = self._detect_language(text)
        
        # 根据语言选择标点符号集
        if lang == 'en':
            # 英文标点：句号、问号、感叹号、分号、逗号
            punctuation = r'[.!?;,]'
        else:
            # 中文标点
            punctuation = r'[。！？；，]'
        
        # 分割文本
        initial_sentences = re.split(punctuation, text)
        
        # 过滤空字符串并去除前后空白
        initial_sentences = [s.strip() for s in initial_sentences if s.strip()]
        
        # 合并短句（英文句子更短，调整长度阈值）
        min_length = 15 if lang == 'en' else 25
        merged_sentences = []
        current_sentence = ""
        
        for sentence in initial_sentences:
            if not current_sentence:
                current_sentence = sentence
            else:
                # 根据语言选择连接符
                connector = ", " if lang == 'en' else "，"
                current_sentence += connector + sentence
                
                if len(current_sentence) > min_length:
                    merged_sentences.append(current_sentence)
                    current_sentence = ""
                    
        if current_sentence:
            merged_sentences.append(current_sentence)
        
        return merged_sentences

    def generate_diversified_queries(self, chunk: str) -> Dict[str, List[str]]:
        """生成多样化查询（根据语言切换提示词）"""
        lang = self._detect_language(chunk)
        queries = {
            "forward": [],
            "backward": [],
            "overlap": []
        }
        
        # 生成前向查询
        queries["forward"] = self._generate_ward_queries(chunk, 2, 'forward', lang)       
        # 生成后向查询
        queries["backward"] = self._generate_ward_queries(chunk, 2, 'backward', lang)     
        # 生成重叠块查询
        queries["overlap"] = self._generate_overlap_queries(chunk, 2, lang)    
        
        return queries
    
    def _generate_ward_queries(self, chunk: str, num_queries: int, mode: str, lang: str) -> List[str]:
        """生成前向/后向查询（根据语言切换提示词）"""
        # 英文提示词
        if lang == 'en':
            if mode == 'forward':
                prompt = f"""
Based on the following text, generate {num_queries} different forward-inference statements. These sentences should:
1. Include possible subsequent content or developments of the text
2. Be highly relevant and natural to the text content
3. Be suitable for information retrieval
4. Each statement must be a single sentence without punctuation marks like commas, periods, semicolons, or colons that break the sentence

Text content:
{chunk}

Please directly return {num_queries} statements, each separated by a newline. Do not add any explanations or numbering.
<system>Each statement must be a single sentence without punctuation marks like commas, periods, semicolons, or colons<system/>
"""
            else:  # backward
                prompt = f"""
Based on the following text, generate {num_queries} different backward-inference statements. These sentences should:
1. Infer possible prior content or background of the text
2. Be highly relevant and natural to the text content
3. Be suitable for information retrieval
4. Each statement must be a single sentence without punctuation marks like commas, periods, semicolons, or colons that break the sentence

Text content:
{chunk}

Please directly return {num_queries} statements, each separated by a newline. Do not add any explanations or numbering.
<system>Each statement must be a single sentence without punctuation marks like commas, periods, semicolons, or colons<system/>
"""
        
        # 中文提示词
        else:
            if mode == 'forward':
                prompt = f"""
基于以下文本内容，生成{num_queries}个不同的前向(forward)推理的陈述句。这些句子应该：
1. 包含文本后续可能的内容或发展
2. 与文本内容高度相关且自然
3. 适合用于信息检索
4. 每个陈述句必须是一句话，不能包含逗号、句号、分号、冒号等断句标点符号

文本内容：
{chunk}

请直接返回{num_queries}个陈述句，每个陈述句用换行符分隔，不要添加任何解释或编号。
<system>每个陈述句必须是一句话，不能包含逗号、句号、分号、冒号等断句标点符号<system/>
"""
            else:  # backward
                prompt = f"""
基于以下文本内容，生成{num_queries}个不同的后向(backward)查询陈述句。这些句子应该：
1. 推理文本之前可能的内容或背景
2. 与文本内容高度相关且自然
3. 适合用于信息检索
4. 每个陈述句必须是一句话，不能包含逗号、句号、分号、冒号等断句标点符号

文本内容：
{chunk}

请直接返回{num_queries}个陈述句，每个陈述句用换行符分隔，不要添加任何解释或编号。
<system>每个陈述句必须是一句话，不能包含逗号、句号、分号、冒号等断句标点符号<system/>
"""

        response = self.model.generate_response(prompt, max_length=200, temperature=0.7)
        queries = self._parse_query_response(response, lang)
        return queries[:num_queries]

    def _generate_overlap_queries(self, chunk: str, num_queries: int, lang: str) -> List[str]:
        """生成重叠块查询（根据语言切换提示词）"""
        # 计算重叠部分
        chunk_length = len(chunk)
        overlap_length = int(chunk_length * self.overlap_ratio)
        
        # 获取开头和结尾的重叠部分
        start_overlap = chunk[:overlap_length] if chunk_length > overlap_length else chunk
        end_overlap = chunk[-overlap_length:] if chunk_length > overlap_length else chunk
        
        queries = []
        half_queries = (num_queries + 1) // 2
        
        # 英文提示词
        if lang == 'en':
            # 开头重叠部分查询
            if start_overlap:
                prompt = f"""
Based on the following text fragment, generate {half_queries} query sentences. These queries should:
1. Be based on the beginning part of the text fragment
2. Help retrieve other content related to this beginning
3. Be natural and suitable for information retrieval
4. Each statement must be a single sentence without punctuation marks like commas, periods, semicolons, or colons

Text fragment (beginning):
{start_overlap}

Please directly return {half_queries} statements, each separated by a newline. Do not add any explanations or numbering. Remember to return statements, not questions.
<system>Each statement must be a single sentence without punctuation marks like commas, periods, semicolons, or colons<system/>
""" 
                response = self.model.generate_response(prompt, max_length=150, temperature=0.6)
                start_queries = self._parse_query_response(response, lang)
                queries.extend(start_queries[:half_queries])
            
            # 结尾重叠部分查询
            remaining = num_queries - len(queries)
            if end_overlap and remaining > 0:
                prompt = f"""
Based on the following text fragment, generate {remaining} query sentences. These queries should:
1. Be based on the ending part of the text fragment
2. Help retrieve other content related to this ending
3. Be natural and suitable for information retrieval
4. Each statement must be a single sentence without punctuation marks like commas, periods, semicolons, or colons

Text fragment (ending):
{end_overlap}

Please directly return {remaining} statements, each separated by a newline. Do not add any explanations or numbering. Remember to return statements, not questions.
<system>Each statement must be a single sentence without punctuation marks like commas, periods, semicolons, or colons<system/>
"""
                response = self.model.generate_response(prompt, max_length=150, temperature=0.6)
                end_queries = self._parse_query_response(response, lang)
                queries.extend(end_queries[:remaining])
        
        # 中文提示词
        else:
            # 开头重叠部分查询
            if start_overlap:
                prompt = f"""
基于以下文本片段，生成{half_queries}个查询句子。这些查询应该：
1. 基于文本片段的开头部分
2. 能够帮助检索到与这个开头相关的其他内容
3. 自然且适合用于信息检索
4. 每个陈述句必须是一句话，不能包含逗号、句号、分号、冒号等断句标点符号

文本片段（开头）：
{start_overlap}

请直接返回{half_queries}个陈述句，每个陈述句用换行符分隔，不要添加任何解释或编号，记住，要返回的是陈述句而不是问题。
<system>每个陈述句必须是一句话，不能包含逗号、句号、分号、冒号等断句标点符号<system/>
""" 
                response = self.model.generate_response(prompt, max_length=150, temperature=0.6)
                start_queries = self._parse_query_response(response, lang)
                queries.extend(start_queries[:half_queries])
            
            # 结尾重叠部分查询
            remaining = num_queries - len(queries)
            if end_overlap and remaining > 0:
                prompt = f"""
基于以下文本片段，生成{remaining}个查询句子。这些查询应该：
1. 基于文本片段的结尾部分
2. 能够帮助检索到与这个结尾相关的其他内容
3. 自然且适合用于信息检索
4. 每个陈述句必须是一句话，不能包含逗号、句号、分号、冒号等断句标点符号

文本片段（结尾）：
{end_overlap}

请直接返回{remaining}个查询句子，每个陈述句用换行符分隔，不要添加任何解释或编号，记住，要返回的是陈述句而不是问题。
<system>每个陈述句必须是一句话，不能包含逗号、句号、分号、冒号等断句标点符号,必须是陈述句<system/>
"""
                response = self.model.generate_response(prompt, max_length=150, temperature=0.6)
                end_queries = self._parse_query_response(response, lang)
                queries.extend(end_queries[:remaining])
        
        return queries[:num_queries]

    def _parse_query_response(self, response: str, lang: str) -> List[str]:
        """解析模型返回的查询（适配中英文长度）"""
        queries = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # 移除编号和特殊字符
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•*]\s*', '', line)
            line = re.sub(r'^["\']|["\']$', '', line)  # 移除引号
            
            sentences = self.split_text_by_punctuation(line)
            
            # 英文查询长度阈值调整
            min_len = 5 if lang == 'en' else 8
            max_len = 50 if lang == 'en' else 40
            
            for sentence in sentences:
                if sentence and min_len < len(sentence) < max_len:
                    queries.append(sentence)
        
        return queries
