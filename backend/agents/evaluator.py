from typing import Dict, Any, List
import google.generativeai as genai
import json
from .base import BaseAgent

class EvaluatorAgent(BaseAgent):
    def __init__(self, api_key: str):
        super().__init__("evaluator")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    async def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        if function_name == "evaluate_answer":
            return await self.evaluate_answer(
                arguments["question"],
                arguments["user_answer"],
                arguments.get("context", "")
            )
        raise ValueError(f"Unknown function: {function_name}")
    
    async def evaluate_answer(self, question: Dict, user_answer: str, context: str = "") -> Dict:
        """사용자 답변 평가"""
        try:
            # 평가 프롬프트 작성
            evaluation_prompt = f"""
다음 문제와 사용자의 답변을 평가해주세요.

문제:
{json.dumps(question, ensure_ascii=False, indent=2)}

사용자 답변:
{user_answer}

문맥 (있는 경우):
{context}

다음 기준으로 평가하고, 반드시 아래 JSON 형식으로만 응답해주세요:
1. 정답 여부
2. 이해도 수준 (0.0 ~ 1.0)
3. 구체적인 피드백

{
    "is_correct": true/false,
    "understanding_score": 0.0~1.0,
    "feedback": "상세한 피드백 메시지",
    "improvement_points": [  // 개선이 필요한 사항들
        "개선점1",
        "개선점2"
    ],
    "additional_explanation": "추가 설명 (필요한 경우)"
}"""
            
            # 평가 수행
            evaluation_response = self.model.generate_content(evaluation_prompt)
            evaluation_result = json.loads(self._clean_json_response(evaluation_response.text))
            
            # 결과 검증 및 보완
            if not isinstance(evaluation_result.get("understanding_score"), (int, float)):
                evaluation_result["understanding_score"] = 0.0
            
            if "improvement_points" not in evaluation_result:
                evaluation_result["improvement_points"] = []
            
            if "additional_explanation" not in evaluation_result:
                evaluation_result["additional_explanation"] = ""
            
            return evaluation_result
            
        except Exception as e:
            print(f"답변 평가 중 오류 발생: {str(e)}")
            return {
                "is_correct": False,
                "understanding_score": 0.0,
                "feedback": "평가 중 오류가 발생했습니다",
                "improvement_points": ["다시 시도해주세요"],
                "additional_explanation": ""
            }
    
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