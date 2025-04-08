# AI 퀴즈 생성기

PDF 문서를 기반으로 AI가 자동으로 문제를 생성하고 학습을 도와주는 시스템입니다.

## 주요 기능

- PDF 파일 업로드 및 텍스트 추출
- AI를 활용한 자동 문제 생성 (객관식/주관식)
- 실시간 답안 채점 및 피드백 제공
- 학습 진도 추적 및 성과 분석

## 기술 스택

- **프론트엔드**
  - Next.js 14
  - TypeScript
  - Chakra UI
  - NextAuth.js (인증)

- **백엔드**
  - FastAPI
  - Google Gemini API (AI 문제 생성)
  - PyPDF2 (PDF 처리)

## 시작하기

### 요구사항

- Node.js 18.0.0 이상
- Python 3.8 이상
- Google Cloud API 키

### 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/ai-quiz-generator.git
cd ai-quiz-generator
```

2. 프론트엔드 설정
```bash
# 의존성 설치
npm install

# 환경 변수 설정
cp .env.example .env.local
# .env.local 파일을 열어 필요한 값들을 설정
```

3. 백엔드 설정
```bash
cd backend

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 Google Cloud API 키 등을 설정
```

### 실행 방법

1. 백엔드 서버 실행
```bash
cd backend
python main.py
```

2. 프론트엔드 개발 서버 실행
```bash
npm run dev
```

3. 브라우저에서 http://localhost:3000 접속

## 사용 방법

1. 로그인 (테스트 계정: test@test.com / 1234)
2. PDF 파일 업로드
3. AI가 자동으로 문제 생성
4. 문제 풀이 및 즉시 피드백 확인
5. 전체 문제 제출 후 종합 평가 확인

## 주의사항

- PDF 파일은 텍스트 추출이 가능한 형식이어야 합니다
- 문제 생성에는 일정 시간이 소요될 수 있습니다
- API 사용량에 따라 제한이 있을 수 있습니다

## 라이선스

MIT License

## 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 
