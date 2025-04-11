from typing import Dict, Any, List
import google.generativeai as genai
import json
from .base import BaseAgent
from PyPDF2 import PdfReader
from fastapi import UploadFile
import re

class QuestionGeneratorAgent(BaseAgent):
    def __init__(self, api_key: str):
        super().__init__("question_generator")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    async def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        if function_name == "generate_questions":
            return await self.generate_questions(arguments["text"])
        elif function_name == "extract_text":
            return await self.extract_text(arguments["file"])
        raise ValueError(f"Unknown function: {function_name}")
    
    async def extract_text(self, file: UploadFile) -> str:
        """PDF에서 텍스트 추출"""
        pdf = PdfReader(file.file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return self.preprocess_text(text)
    
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        return ' '.join(text.split()).replace('•', '')

    def create_chunks(self, text: str, chunk_size: int = 450, overlap: int = 100) -> List[str]:
        """텍스트를 오버랩이 있는 청크로 분할"""
        sentences = re.split('[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length <= chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                
                if overlap > 0 and current_chunk:
                    overlap_sentences = current_chunk[-(overlap//20):]
                    current_chunk = overlap_sentences.copy()
                    current_length = sum(len(s.split()) for s in overlap_sentences)
                else:
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(sentence)
                current_length = sentence_length
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    async def generate_questions(self, text: str) -> List[Dict]:
        """텍스트 기반으로 문제 생성"""
        try:
            # 텍스트를 청크로 분할
            chunks = self.create_chunks(text)
            all_concepts = []
            
            # 각 청크에서 핵심 개념 추출
            for chunk in chunks:
                concepts_prompt = f"""다음 텍스트에서 핵심 개념들을 추출해주세요.

텍스트:
{chunk}

반드시 다음 JSON 형식으로만 응답하세요:
{{
    "concepts": [
        {{
            "concept": "핵심 개념",
            "description": "개념 설명",
            "importance": 0.9  // 0.0 ~ 1.0 사이의 중요도
        }}
    ]
}}"""
                
                concepts_response = self.model.generate_content(concepts_prompt)
                concepts_json = self._clean_json_response(concepts_response.text)
                chunk_concepts = json.loads(concepts_json).get("concepts", [])
                all_concepts.extend(chunk_concepts)
            
            # 중복 제거 및 중요도 순 정렬
            unique_concepts = []
            seen_concepts = set()
            for concept in sorted(all_concepts, key=lambda x: x["importance"], reverse=True):
                if concept["concept"] not in seen_concepts:
                    unique_concepts.append(concept)
                    seen_concepts.add(concept["concept"])
            
            # 상위 5개 개념만 선택
            top_concepts = unique_concepts[:5]
            
            # 문제 생성
            questions_prompt = f"""다음 핵심 개념들을 바탕으로 문제를 생성해주세요.

핵심 개념:
{json.dumps(top_concepts, ensure_ascii=False, indent=2)}

원본 텍스트:
{text}

다음 지침을 엄격히 따라주세요:
1. 정확히 3개의 객관식 문제를 생성하세요.
2. 각 문제는 반드시 텍스트 내용과 추출된 핵심 개념에 기반해야 합니다.
3. 각 문제는 4개의 보기를 가져야 합니다.
4. 정답은 반드시 보기 중 하나여야 합니다.
5. 해설은 왜 그 답이 정답인지 명확히 설명해야 합니다.
6. 각 문제의 난이도는 개념의 중요도를 반영해야 합니다.
7. 표나 그림을 참조하는 문제는 내지 마세요.

반드시 다음 JSON 형식으로만 응답하세요. 다른 텍스트나 설명을 포함하지 마세요:
{{
    "questions": [
        {{
            "type": "multiple_choice",
            "question": "문제 내용",
            "options": ["보기1", "보기2", "보기3", "보기4"],
            "correct_answer": "정답 (반드시 보기 중 하나와 정확히 일치해야 함)",
            "explanation": "해설 (왜 이 답이 정답인지 설명)",
            "evidence": "텍스트 내 근거가 되는 부분",
            "concept": "관련된 핵심 개념",
            "difficulty": 0.8  // 0.0 ~ 1.0 사이의 난이도
        }}
    ]
}}"""
            
            questions_response = self.model.generate_content(questions_prompt)
            questions_json = self._clean_json_response(questions_response.text)
            result = json.loads(questions_json)
            
            # 문제 검증
            validated_questions = []
            for q in result.get("questions", []):
                if self._validate_question(q):
                    validated_questions.append(q)
            
            if not validated_questions:
                raise ValueError("유효한 문제가 생성되지 않았습니다")
            
            return validated_questions
            
        except Exception as e:
            print(f"문제 생성 중 오류 발생: {str(e)}")
            return []
    
    def _clean_json_response(self, response: str) -> str:
        """JSON 응답 문자열 정제"""
        if isinstance(response, str):
            response = re.sub(r'```json\s*|\s*```', '', response)
            response = re.sub(r'```\s*|\s*```', '', response)
            
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                return response[start:end]
        
        raise ValueError("유효한 JSON을 찾을 수 없습니다")
    
    def _validate_question(self, question: Dict) -> bool:
        """개별 문제 유효성 검증"""
        try:
            required_fields = ["type", "question", "options", "correct_answer", "explanation"]
            if not all(field in question for field in required_fields):
                return False
            
            if question["type"] != "multiple_choice":
                return False
            
            if not isinstance(question["options"], list) or len(question["options"]) != 4:
                return False
            
            if question["correct_answer"] not in question["options"]:
                return False
            
            if not question["question"].strip() or not question["explanation"].strip():
                return False
            
            return True
            
        except Exception as e:
            print(f"문제 검증 중 오류: {str(e)}")
            return False