from typing import Dict, Any, List
import google.generativeai as genai
import json
from .base import BaseAgent
import torch
from sentence_transformers import SentenceTransformer

class CriticAgent(BaseAgent):
    def __init__(self, api_key: str):
        super().__init__("critic")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    
    async def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        if function_name == "verify_questions":
            return await self.verify_questions(arguments["questions"], arguments["context"])
        raise ValueError(f"Unknown function: {function_name}")
    
    def get_embedding(self, text: str) -> torch.Tensor:
        """텍스트의 임베딩 벡터 생성"""
        return self.embedding_model.encode(text, convert_to_tensor=True)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트 간의 코사인 유사도 계산"""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return float(torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)))
    
    async def verify_questions(self, questions: List[Dict], context: str) -> List[Dict]:
        """문제의 품질과 정확성 검증"""
        try:
            verified_questions = []
            
            for question in questions:
                try:
                    # 1. 문맥 관련성 검사
                    question_text = f"{question['question']} {' '.join(question['options'])} {question['explanation']}"
                    context_similarity = self.calculate_similarity(question_text, context)
                    
                    if context_similarity < 0.35:  # 임계값 조정
                        print(f"[DEBUG] 문맥 유사도 낮음 ({context_similarity:.3f}): {question['question'][:100]}...")
                        continue
                    
                    # 2. 문제 품질 검증
                    verification_prompt = f"""
다음 문제가 주어진 문맥에 기반하여 적절하고 정확한지 검증해주세요.

문맥:
{context}

문제:
{json.dumps(question, ensure_ascii=False, indent=2)}

다음 기준으로 검증하고, 반드시 아래 JSON 형식으로만 응답해주세요:
1. 문제가 문맥에 기반하는가?
2. 정답이 명확하고 정확한가?
3. 오답이 적절한 난이도를 가지는가?
4. 설명이 충분하고 정확한가?

{
    "is_valid": true/false,
    "score": 0.0~1.0,  // 종합 품질 점수
    "feedback": "검증 결과에 대한 구체적인 피드백",
    "improvements": [  // 개선이 필요한 사항들
        "개선점1",
        "개선점2"
    ]
}"""
                    
                    verification_response = self.model.generate_content(verification_prompt)
                    verification_result = json.loads(self._clean_json_response(verification_response.text))
                    
                    if verification_result["is_valid"] and verification_result["score"] >= 0.7:
                        question["verification_result"] = verification_result
                        verified_questions.append(question)
                    else:
                        print(f"[DEBUG] 검증 탈락 (점수: {verification_result['score']:.2f}): {verification_result['feedback']}")
                
                except Exception as e:
                    print(f"개별 문제 검증 중 오류: {str(e)}")
                    continue
            
            if not verified_questions:
                print("[DEBUG] 검증을 통과한 문제가 없습니다")
            
            return verified_questions
            
        except Exception as e:
            print(f"문제 검증 중 오류 발생: {str(e)}")
            return []
    
    def _clean_json_response(self, response: str) -> str:
        """JSON 응답 문자열 정제"""
        import re
        
        if isinstance(response, str):
            # JSON 블록 제거
            response = re.sub(r'```json\s*|\s*```', '', response)
            response = re.sub(r'```\s*|\s*```', '', response)
            
            # 실제 JSON 부분 추출
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                return response[start:end]
        
        raise ValueError("유효한 JSON을 찾을 수 없습니다")