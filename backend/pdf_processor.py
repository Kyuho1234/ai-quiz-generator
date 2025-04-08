import PyPDF2
import io

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        # PDF 파일 내용을 BytesIO 객체로 변환
        pdf_file = io.BytesIO(file_content)
        
        # PDF 리더 객체 생성
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # 모든 페이지의 텍스트 추출
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'
        
        # 텍스트 전처리
        text = clean_text(text)
        
        return text
    except Exception as e:
        raise Exception(f'PDF 텍스트 추출 중 오류 발생: {str(e)}')

def clean_text(text: str) -> str:
    # 불필요한 공백 제거
    text = ' '.join(text.split())
    
    # 특수 문자 처리
    text = text.replace('•', '')
    
    return text