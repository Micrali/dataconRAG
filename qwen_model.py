# qwen_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
from langdetect import detect, LangDetectException
import re

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
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7,system_set: str="You are a machine,and you must strictly follow the user’s instructions.") -> str:
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


class SentenceExpander:
    """智能句子扩展器 - 使用Qwen模型"""

    def __init__(self, qwen_model: QwenModel,overlap_ratio):
        self.model = qwen_model
        self.overlap_ratio = overlap_ratio #重叠率

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
        min_length = 30 if lang == 'en' else 50
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
        """
        基于chunk生成多样化的查询
        返回包含2个前向查询、2个后向查询和2个重叠块查询的字典
        """
        lang = self._detect_language(chunk)
        queries = {
            "forward": [],
            "backward": [],
        }
         # 生成前向查询
        queries["forward"] = self._generate_ward_queries(chunk, 2, 'forward', lang)       
        # 生成后向查询
        queries["backward"] = self._generate_ward_queries(chunk, 2, 'backward', lang)     
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
            elif mode == 'backward':  # backward
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
            elif mode =='backward':  # backward
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

if __name__ == "__main__":
        local=QwenModel()
        test=SentenceExpander(local,0.2)
        text=['Hello,my name is WSH.who are you?']
        
        print(test.generate_diversified_queries(text[0]))
