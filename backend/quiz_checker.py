import google.generativeai as genai

def check_answers(answers: list) -> dict:
    total_questions = len(answers)
    correct_count = 0
    results = []
    
    for answer in answers:
        result = check_single_answer(answer)
        results.append(result)
        if result['is_correct']:
            correct_count += 1
    
    score_percentage = (correct_count / total_questions) * 100
    overall_feedback = generate_overall_feedback(score_percentage, results)
    
    return {
        'results': results,
        'total_score': correct_count,
        'total_questions': total_questions,
        'score_percentage': score_percentage,
        'overall_feedback': overall_feedback
    }

def check_single_answer(answer: dict) -> dict:
    user_answer = answer['user_answer'].lower().strip()
    correct_answer = answer['correct_answer'].lower().strip()
    
    is_correct = user_answer == correct_answer
    
    feedback = generate_feedback(
        question=answer['question'],
        user_answer=answer['user_answer'],
        correct_answer=answer['correct_answer'],
        is_correct=is_correct
    )
    
    return {
        'is_correct': is_correct,
        'feedback': feedback,
        'question': answer['question'],
        'user_answer': answer['user_answer'],
        'correct_answer': answer['correct_answer']
    }

def generate_feedback(question: str, user_answer: str, correct_answer: str, is_correct: bool) -> str:
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""
    문제: {question}
    학습자의 답: {user_answer}
    정답: {correct_answer}
    
    위 내용을 기반로 자세한 피드백을 작성해주세요.
    {'\n\n정답입니다! ' if is_correct else '\n\n오답입니다. '}
    """
    
    response = model.generate_content(prompt)
    return response.text

def generate_overall_feedback(score_percentage: float, results: list) -> str:
    model = genai.GenerativeModel('gemini-pro')
    
    correct_count = sum(1 for r in results if r['is_correct'])
    total_questions = len(results)
    
    prompt = f"""
    학습자가 총 {total_questions}문제 중 {correct_count}문제를 맞춰 {score_percentage:.1f}%의 점수를 받았습니다.
    
    이러한 결과를 기반로 학습자에게 도움이 될 만한 종합적인 피드백을 작성해주세요.
    """
    
    response = model.generate_content(prompt)
    return response.text