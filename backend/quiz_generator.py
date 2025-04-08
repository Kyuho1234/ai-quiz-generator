import google.generativeai as genai
import json

def setup_gemini(api_key: str):
    genai.configure(api_key=api_key)

def generate_questions(text: str, num_questions: int = 5) -> list:
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""
    다음 텍스트를 기반로 {num_questions}개의 문제를 생성해주세요.
    각 문제는 객관식 또는 주관식이어야 합니다.
    각 문제는 정답과 함께 자세한 해설을 포함해야 합니다.
    
    응답은 다음 JSON 형식으로 제공해주세요:
    [
      {{
        "question": "문제 내용",
        "type": "multiple_choice",  // 또는 "short_answer"
        "options": ["보기 1", "보기 2", "보기 3", "보기 4"],  // 객관식인 경우에만
        "correct_answer": "정답",
        "explanation": "해설"
      }}
    ]
    
    텍스트:
    {text}
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text
        
        # JSON 문자열 추출
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError('JSON 형식의 응답을 찾을 수 없습니다.')
            
        json_str = response_text[start_idx:end_idx]
        
        # JSON 파싱
        questions = json.loads(json_str)
        
        # 문제 검증
        verified_questions = []
        for q in questions:
            if verify_question(q):
                verified_questions.append(q)
        
        if not verified_questions:
            raise ValueError('유효한 문제를 생성하지 못했습니다.')
            
        return verified_questions
        
    except Exception as e:
        raise Exception(f'문제 생성 중 오류 발생: {str(e)}')

def verify_question(question: dict) -> bool:
    required_fields = ['question', 'type', 'correct_answer', 'explanation']
    
    # 필수 필드 검사
    if not all(field in question for field in required_fields):
        return False
        
    # 문제 유형 검사
    if question['type'] not in ['multiple_choice', 'short_answer']:
        return False
        
    # 객관식인 경우 보기 검사
    if question['type'] == 'multiple_choice':
        if 'options' not in question or not isinstance(question['options'], list):
            return False
        if len(question['options']) < 2:
            return False
            
    return True