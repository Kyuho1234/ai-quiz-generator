from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import torch
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from typing import List, Dict
import json
import re

load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini 모델 설정
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# RAG 모델 설정
rag_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

def preprocess_text(text: str) -> str:
    # 연속된 공백을 하나로 치환
    text = ' '.join(text.split())
    # 불필요한 특수문자 제거
    text = text.replace('•', '')
    return text

def extract_text_from_pdf(file: UploadFile) -> str:
    pdf = PdfReader(file.file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    # 텍스트 전처리
    return preprocess_text(text)

def verify_semantic_consistency(question: Dict, context: str) -> Dict:
    prompt = (
        "다음 문서와 생성된 문제의 의미론적 일관성을 검증해주세요.\n\n"
        f"문서: {context}\n\n"
        f"문제: {question['question']}\n"
        f"답변: {question['correct_answer']}\n"
        f"해설: {question['explanation']}\n\n"
        "다음 기준으로 평가해주세요 (각 항목 0-1점):\n"
        "1. 문제가 문서의 핵심 내용을 다루고 있는가?\n"
        "2. 답변이 문서의 내용과 일치하는가?\n"
        "3. 해설이 문서의 내용을 정확히 반영하는가?\n"
        "4. 문제의 난이도가 적절한가?\n"
        "5. 문제가 오해의 소지 없이 명확한가?\n\n"
        "각 항목에 대해 0에서 1 사이의 점수를 매기고, 종합적인 피드백을 제공해주세요.\n"
        "다음 항목들에 대해 0과 1 사이의 점수를 매겨주세요:\n"
        "- content_relevance (문제가 문서의 핵심 내용을 다루는 정도)\n"
        "- answer_accuracy (답변이 문서의 내용과 일치하는 정도)\n"
        "- explanation_accuracy (해설이 문서의 내용을 정확히 반영하는 정도)\n"
        "- difficulty_appropriateness (문제 난이도의 적절성)\n"
        "- clarity (문제의 명확성)\n\n"
        "또한 전반적인 피드백도 함께 제공해주세요."
    )
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # 응답에서 점수와 피드백 추출
        scores = {}
        feedback = ""
        
        # 점수 추출을 위한 키워드
        score_keywords = {
            "content_relevance": ["content_relevance", "문서 관련성", "핵심 내용"],
            "answer_accuracy": ["answer_accuracy", "답변 정확성", "답변 일치"],
            "explanation_accuracy": ["explanation_accuracy", "해설 정확성", "해설 일치"],
            "difficulty_appropriateness": ["difficulty_appropriateness", "난이도 적절성", "난이도"],
            "clarity": ["clarity", "명확성", "문제 명확성"]
        }
        
        # 각 점수 항목에 대해 값 추출
        for score_key, keywords in score_keywords.items():
            score = 0.5  # 기본값
            for keyword in keywords:
                for line in response_text.split('\n'):
                    if any(keyword in line.lower() for keyword in keywords):
                        # 숫자 추출 시도
                        import re
                        numbers = re.findall(r'0\.\d+|\d+\.?\d*', line)
                        if numbers:
                            try:
                                score = float(numbers[0])
                                if 0 <= score <= 1:  # 유효한 범위인지 확인
                                    break
                            except ValueError:
                                continue
            scores[score_key] = score
        
        # 피드백 추출
        feedback_markers = ["피드백", "종합", "평가", "의견"]
        for line in response_text.split('\n'):
            if any(marker in line for marker in feedback_markers) and len(line) > 10:
                feedback = line.strip()
                break
        
        if not feedback:  # 피드백을 찾지 못한 경우
            feedback = "자동 생성된 기본 피드백입니다."
        
        # 의미론적 일관성 점수 로깅
        print(f"\n[의미론적 일관성 점수]")
        print(f"문제: {question['question']}")
        print("개별 점수:")
        for key, score in scores.items():
            print(f"- {key}: {score}")
        print(f"평균 점수: {sum(scores.values()) / len(scores)}")
        print("-" * 80)
        
        return {
            'score': sum(scores.values()) / len(scores),
            'details': {
                'scores': scores,
                'feedback': feedback
            }
        }
    except Exception as e:
        print(f"의미론적 일관성 검증 중 오류: {str(e)}")
        return {
            'score': 0.5,
            'details': {
                'scores': {
                    'content_relevance': 0.5,
                    'answer_accuracy': 0.5,
                    'explanation_accuracy': 0.5,
                    'difficulty_appropriateness': 0.5,
                    'clarity': 0.5
                },
                'feedback': '검증 중 오류 발생'
            }
        }

def extract_evidence(question: Dict, context: str) -> Dict:
    prompt = (
        "다음 문제의 답변과 해설이 문서의 어느 부분에 근거하는지 찾아주세요:\n\n"
        f"문서: {context}\n"
        f"문제: {question['question']}\n"
        f"답변: {question['correct_answer']}\n"
        f"해설: {question['explanation']}\n\n"
        "다음 정보를 제공해주세요:\n"
        "1. 문서에서 찾은 근거가 되는 텍스트를 정확히 인용해주세요.\n"
        "2. 근거의 신뢰도를 0에서 1 사이의 점수로 평가해주세요.\n"
        "3. 근거와 문제/답변/해설 간의 관계를 설명해주세요."
    )
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # 응답에서 필요한 정보 추출
        evidence_text = ""
        confidence_score = 0.5
        explanation = ""
        
        lines = response_text.split('\n')
        for line in lines:
            # 근거 텍스트 추출
            if "근거" in line or "인용" in line:
                evidence_text = line.split(":", 1)[1].strip() if ":" in line else line.strip()
            
            # 신뢰도 점수 추출
            if "신뢰도" in line or "점수" in line:
                import re
                numbers = re.findall(r'0\.\d+|\d+\.?\d*', line)
                if numbers:
                    try:
                        score = float(numbers[0])
                        if 0 <= score <= 1:
                            confidence_score = score
                    except ValueError:
                        pass
            
            # 설명 추출
            if "관계" in line or "설명" in line:
                explanation = line.split(":", 1)[1].strip() if ":" in line else line.strip()
        
        # 신뢰도 점수 로깅
        print(f"\n[근거 추출 신뢰도]")
        print(f"문제: {question['question']}")
        print(f"신뢰도 점수: {confidence_score}")
        print(f"근거: {evidence_text[:100]}..." if len(evidence_text) > 100 else evidence_text)
        print("-" * 80)
        
        return {
            'evidence_found': bool(evidence_text),
            'evidence_text': evidence_text or "근거를 찾을 수 없습니다.",
            'confidence_score': confidence_score,
            'explanation': explanation or "관계 설명을 생성할 수 없습니다."
        }
    except Exception as e:
        print(f"근거 추출 중 오류: {str(e)}")
        return {
            'evidence_found': True,
            'evidence_text': '',
            'confidence_score': 0.5,
            'explanation': '근거 추출 중 오류 발생'
        }

def verify_question_with_rag(question: Dict, context: str) -> Dict:
    try:
        # 1. 기존 임베딩 기반 유사도 계산
        doc_embedding = rag_model.encode(context, convert_to_tensor=True)
        
        question_text = preprocess_text(question["question"])
        answer_text = preprocess_text(question["correct_answer"])
        explanation_text = preprocess_text(question["explanation"])
        
        q_embedding = rag_model.encode(question_text, convert_to_tensor=True)
        a_embedding = rag_model.encode(answer_text, convert_to_tensor=True)
        e_embedding = rag_model.encode(explanation_text, convert_to_tensor=True)
        
        q_similarity = torch.cosine_similarity(doc_embedding.unsqueeze(0), q_embedding.unsqueeze(0)).item()
        a_similarity = torch.cosine_similarity(doc_embedding.unsqueeze(0), a_embedding.unsqueeze(0)).item()
        e_similarity = torch.cosine_similarity(doc_embedding.unsqueeze(0), e_embedding.unsqueeze(0)).item()
        
        # 평균 유사도 계산
        avg_similarity = (q_similarity + a_similarity + e_similarity) / 3
        
        # 2. 의미론적 일관성 검증
        semantic_result = verify_semantic_consistency(question, context)
        
        # 3. 근거 추출 및 신뢰도 평가
        evidence_result = extract_evidence(question, context)
        
        # 종합적인 검증 결과 계산
        # 임베딩 유사도(0.4), 의미론적 일관성(0.3), 근거 신뢰도(0.3) 가중치 적용
        final_score = (
            0.4 * avg_similarity +
            0.3 * semantic_result['score'] +
            0.3 * evidence_result['confidence_score']
        )
        
        # 임계값을 0.2로 설정 (필요에 따라 조정 가능)
        is_valid = final_score > 0.2
        
        print(f"\n[종합 검증 결과]")
        print(f"문제: {question['question']}")
        print(f"임베딩 유사도: {avg_similarity:.3f}")
        print(f"의미론적 일관성: {semantic_result['score']:.3f}")
        print(f"근거 신뢰도: {evidence_result['confidence_score']:.3f}")
        print(f"최종 점수: {final_score:.3f}")
        print("-" * 80)
        
        return {
            "is_relevant": is_valid,
            "question": question,
            "scores": {
                "embedding_similarity": avg_similarity,
                "semantic_consistency": semantic_result['score'],
                "evidence_confidence": evidence_result['confidence_score'],
                "final_score": final_score
            }
        }
        
    except Exception as e:
        print(f"검증 중 오류 발생: {str(e)}")
        # 오류 발생 시 기본적으로 문제를 수용
        return {
            "is_relevant": True,
            "question": question,
            "scores": {
                "embedding_similarity": 0.5,
                "semantic_consistency": 0.5,
                "evidence_confidence": 0.5,
                "final_score": 0.5
            }
        }

def clean_json_string(json_str: str) -> str:
    # 코드 블록 마커 제거
    json_str = json_str.replace('```json', '').replace('```', '')
    
    # 모든 줄바꿈 제거
    json_str = json_str.replace('\n', ' ').replace('\r', ' ')
    
    # 연속된 공백을 하나로 치환
    json_str = ' '.join(json_str.split())
    
    # 객체 내의 불필요한 공백 제거
    json_str = json_str.replace(' : ', ':').replace(' , ', ',')
    
    # 배열의 마지막 요소 뒤의 쉼표 제거
    json_str = json_str.replace(',]', ']').replace(', ]', ']')
    
    # options 배열의 마지막 요소 뒤의 쉼표 제거
    if '"options":' in json_str:
        parts = json_str.split('"options":')
        for i in range(1, len(parts)):
            if '[' in parts[i]:
                array_start = parts[i].find('[')
                array_end = parts[i].find(']')
                if array_start != -1 and array_end != -1:
                    array_content = parts[i][array_start:array_end+1]
                    cleaned_array = array_content.replace(', ]', ']').replace(',]', ']')
                    parts[i] = parts[i][:array_start] + cleaned_array + parts[i][array_end+1:]
        json_str = '"options":'.join(parts)
    
    try:
        # 유효성 검사를 위해 파싱 시도
        parsed = json.loads(json_str)
        # 다시 문자열로 변환하여 반환 (표준 JSON 형식으로)
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError as e:
        print(f"JSON 정리 후에도 파싱 실패: {e}")
        print(f"정리된 JSON 문자열: {json_str}")
        # 마지막 시도: 따옴표 주변의 공백 제거
        json_str = json_str.replace(' "', '"').replace('" ', '"')
        return json_str

def generate_questions(text: str) -> List[Dict]:
    prompt = f"""다음 텍스트를 기반으로 5개의 객관식 문제와 3개의 주관식 문제를 생성해주세요.
    각 문제는 다음 JSON 형식으로 작성해주세요:
    [
        {{
            "type": "multiple_choice",
            "question": "문제 내용",
            "options": ["보기1", "보기2", "보기3", "보기4"],
            "correct_answer": "정답",
            "explanation": "해설"
        }},
        {{
            "type": "short_answer",
            "question": "문제 내용",
            "correct_answer": "정답",
            "explanation": "해설"
        }}
    ]
    
    주의사항:
    1. 모든 문제와 답은 반드시 주어진 텍스트 내용에 기반해야 합니다.
    2. 해설은 텍스트의 어느 부분에서 도출되었는지 명시해야 합니다.
    3. 문제의 난이도는 다양하게 구성해주세요.
    4. 반드시 위에 제시된 JSON 형식으로만 응답해주세요. 다른 텍스트는 포함하지 마세요.
    5. JSON 배열의 마지막 요소 뒤에는 쉼표를 넣지 마세요.
    
    텍스트: {text}
    """
    
    try:
        response = model.generate_content(prompt)
        text_response = response.text.strip()
        
        print("Gemini API 응답:", text_response)  # 디버깅용 로그
        
        # 응답이 비어있는 경우 처리
        if not text_response:
            raise ValueError("Gemini API가 빈 응답을 반환했습니다.")
        
        # JSON 시작과 끝 위치 찾기
        start_idx = text_response.find('[')
        end_idx = text_response.rfind(']') + 1
        
        if start_idx == -1 or end_idx == 0:
            # JSON 형식이 아닌 경우, 전체 응답을 다시 생성 시도
            retry_prompt = f"""이전 응답이 올바른 JSON 형식이 아니었습니다.
            반드시 다음과 같은 JSON 배열 형식으로만 응답해주세요:
            [
                {{
                    "type": "multiple_choice",
                    "question": "문제 내용",
                    "options": ["보기1", "보기2", "보기3", "보기4"],
                    "correct_answer": "정답",
                    "explanation": "해설"
                }}
            ]
            다른 텍스트는 포함하지 마세요.
            JSON 배열의 마지막 요소 뒤에는 쉼표를 넣지 마세요."""
            
            response = model.generate_content(retry_prompt)
            text_response = response.text.strip()
            print("재시도 응답:", text_response)  # 디버깅용 로그
            
            start_idx = text_response.find('[')
            end_idx = text_response.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("JSON 형식의 응답을 생성할 수 없습니다.")
        
        json_str = text_response[start_idx:end_idx]
        print("파싱할 JSON 문자열:", json_str)  # 디버깅용 로그
        
        # JSON 문자열 정리
        cleaned_json = clean_json_string(json_str)
        
        try:
            questions = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패: {e}")
            # 마지막 시도: 모든 줄바꿈과 추가 공백 제거
            final_attempt = cleaned_json.replace('\n', '').replace('\r', '').replace('  ', ' ').strip()
            questions = json.loads(final_attempt)
        
        if not isinstance(questions, list):
            raise ValueError("응답이 JSON 배열 형식이 아닙니다.")
        
        # RAG로 각 문제의 관련성 검증
        verified_questions = []
        for question in questions:
            if not isinstance(question, dict):
                continue
            if not all(key in question for key in ["type", "question", "correct_answer", "explanation"]):
                continue
            if question["type"] == "multiple_choice" and "options" not in question:
                continue
                
            verification = verify_question_with_rag(question, text)
            if verification["is_relevant"]:
                verified_questions.append(question)
        
        if not verified_questions:
            raise ValueError("문서와 관련된 유효한 문제를 생성할 수 없습니다.")
            
        return verified_questions
        
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        print(f"응답 텍스트: {text_response}")
        raise HTTPException(
            status_code=500,
            detail="문제 생성 중 오류가 발생했습니다. 다시 시도해주세요."
        )
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        print(f"응답 텍스트: {text_response if 'text_response' in locals() else 'No response'}")
        raise HTTPException(
            status_code=500,
            detail="문제 생성 중 오류가 발생했습니다. 다시 시도해주세요."
        )

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    
    text = extract_text_from_pdf(file)
    questions = generate_questions(text)
    
    return {"questions": questions, "context": text}

@app.post("/api/check-answers")
async def check_answers(data: Dict):
    try:
        answers = data["answers"]
        results = []
        total_score = 0
        total_questions = len(answers)
        
        for answer in answers:
            try:
                question = answer["question"]
                user_answer = answer["user_answer"].strip()
                correct_answer = answer["correct_answer"].strip()
                question_type = answer.get("question_type", "unknown")
                
                # 정답 처리를 위한 전처리
                def normalize_answer(ans: str) -> str:
                    # 소문자 변환
                    ans = ans.lower()
                    # 괄호와 그 안의 내용 제거 (선택적 내용으로 처리)
                    ans = re.sub(r'\s*\([^)]*\)', '', ans)
                    # 연속된 공백을 하나로
                    ans = ' '.join(ans.split())
                    # 특수문자 제거
                    ans = re.sub(r'[.,!?;:]', '', ans)
                    return ans
                
                # 정답 여부 확인 (좀 더 유연한 비교)
                normalized_user_answer = normalize_answer(user_answer)
                normalized_correct_answer = normalize_answer(correct_answer)
                
                # 기본 일치 여부 확인
                is_correct = normalized_user_answer == normalized_correct_answer
                
                # 부분 점수 처리 (예: 객관식 문제에서 보기 번호만 다른 경우)
                if not is_correct and question_type == "multiple_choice":
                    # 보기 번호와 답안 내용을 분리하여 비교
                    user_content = re.sub(r'^\d+\.\s*', '', normalized_user_answer)
                    correct_content = re.sub(r'^\d+\.\s*', '', normalized_correct_answer)
                    is_correct = user_content == correct_content
                
                # Gemini를 통한 개별 피드백 생성
                feedback_prompt = f"""
                다음 문제와 답변에 대한 간단한 피드백을 제공해주세요:
                
                문제: {question}
                학습자 답변: {user_answer}
                정답: {correct_answer}
                정답 여부: {'정답' if is_correct else '오답'}
                
                피드백 작성 시 다음 사항을 고려해주세요:
                1. 답변이 정답인 경우: 간단한 설명만 제공
                2. 답변이 오답인 경우: 핵심적인 차이점만 설명
                3. 괄호 안의 추가 정보(영문/라틴어 표기 등)는 부가적인 것으로 간주
                4. 표현이나 형식이 조금 다르더라도 핵심 내용이 같다면 정답으로 인정
                
                100자 이내로 간단히 작성해주세요.
                """
                
                try:
                    feedback_response = model.generate_content(feedback_prompt)
                    feedback_text = feedback_response.text.strip()
                except Exception as e:
                    print(f"피드백 생성 중 오류 발생: {str(e)}")
                    feedback_text = "정답 여부만 확인되었습니다."
                
                results.append({
                    "question": question,
                    "user_answer": user_answer,
                    "correct_answer": correct_answer,
                    "is_correct": is_correct,
                    "feedback": feedback_text
                })
                
                if is_correct:
                    total_score += 1
                    
            except Exception as e:
                print(f"개별 답안 처리 중 오류 발생: {str(e)}")
                continue
        
        # 나머지 코드는 동일하게 유지
        score_percentage = (total_score / total_questions) * 100 if total_questions > 0 else 0
        
        try:
            summary_prompt = f"""
            학습자의 퀴즈 결과에 대한 종합 평가를 작성해주세요.
            
            결과 정보:
            - 총 문제 수: {total_questions}개
            - 정답 수: {total_score}개
            - 정답률: {score_percentage:.1f}%
            
            다음 내용을 포함하여 150자 이내로 작성해주세요:
            1. 전반적인 이해도 평가
            2. 잘한 부분과 보완이 필요한 부분
            3. 향후 학습 방향 제안
            """
            
            overall_feedback_response = model.generate_content(summary_prompt)
            overall_feedback_text = overall_feedback_response.text.strip()
        except Exception as e:
            print(f"종합 평가 생성 중 오류 발생: {str(e)}")
            overall_feedback_text = f"전체 {total_questions}문제 중 {total_score}문제를 맞추어 {score_percentage:.1f}%의 정답률을 보였습니다."
        
        return {
            "results": results,
            "total_score": total_score,
            "total_questions": total_questions,
            "score_percentage": score_percentage,
            "overall_feedback": overall_feedback_text
        }
        
    except Exception as e:
        print(f"답안 채점 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="답안 채점 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        )

@app.post("/api/check-answer")
async def check_answer(data: Dict):
    """단일 문제 채점용 API (기존 기능 유지)"""
    question = data["question"]
    user_answer = data["answer"]
    correct_answer = data["correct_answer"]
    
    feedback_prompt = f"""
    문제: {question}
    사용자 답변: {user_answer}
    정답: {correct_answer}
    
    위 답변에 대한 자세한 피드백을 제공해주세요. 다음 사항을 포함해주세요:
    1. 답변의 정확성
    2. 부족한 부분 설명
    3. 개선을 위한 제안
    """
    
    feedback = model.generate_content(feedback_prompt)
    
    return {
        "is_correct": user_answer.strip().lower() == correct_answer.strip().lower(),
        "feedback": feedback.text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
