import os
import base64
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from google import genai
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import uvicorn

# .env 파일에서 환경변수 로드
load_dotenv()

# Gemini API 키 (.env 파일에서 가져오기)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

client = genai.Client(api_key=GEMINI_API_KEY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 이벤트 처리"""
    # Startup
    print("Nanobanana API 준비 완료")
    print(f"Gemini API Key 설정: {"O" if GEMINI_API_KEY else "X"}")
    yield
    # Shutdown (필요시 정리 작업)


app = FastAPI(title="Nanobanana Image Composition API", lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Nanobanana Image Composition API", "status": "running"}


@app.post("/api/nano/compose")
async def compose_images(
    files: List[UploadFile] = File(...),
    prompt: str = Form(...),
    model: Optional[str] = Form("gemini-2.5-flash-image")
):
    """
    여러 이미지를 프롬프트에 따라 합성
    
    - **files**: 합성할 이미지 파일들 (크롭된 객체 이미지)
    - **prompt**: 합성 프롬프트 (예: "사진속에 있는 객체들을 함께 있고 색감은 부드러운 화풍은 애니매이션 분위기에 역동적인느낌으로 핸드폰 케이스로 만들어줘")
    - **model**: 사용할 Gemini 모델 (기본값: gemini-2.5-flash-image)
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="이미지 파일이 필요합니다.")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="프롬프트가 필요합니다.")
        
        # 이미지 파일들을 메모리에서 PIL Image로 변환
        imgs_in = []
        
        for file in files:
            # 파일을 메모리에서 읽기
            content = await file.read()
            
            # PIL Image로 직접 로드
            img = Image.open(BytesIO(content))
            imgs_in.append(img)
        
        # Gemini API 호출
        contents = [prompt] + imgs_in
        
        response = client.models.generate_content(
            model=model,
            contents=contents
        )
        
        # 결과 이미지 base64 인코딩 (파일 저장하지 않음)
        saved = False
        result_base64 = None
        
        # 안전 파싱 (이미지 없을 때도 안 터짐)
        if getattr(response, "candidates", None):
            cand = response.candidates[0]
            if cand and cand.content and cand.content.parts:
                for part in cand.content.parts:
                    if getattr(part, "inline_data", None) and part.inline_data.data:
                        out_img = Image.open(BytesIO(part.inline_data.data))
                        saved = True
                        
                        # 이미지를 base64로 인코딩 (파일 저장하지 않음)
                        buffer = BytesIO()
                        out_img.save(buffer, format='PNG')
                        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        break
        
        if not saved:
            raise HTTPException(
                status_code=500,
                detail="이미지 결과가 생성되지 않았습니다. 프롬프트를 더 명확하게 작성해주세요."
            )
        
        return {
            "message": "이미지 합성 완료",
            "output_path": "memory",  # 가상 경로 (파일 저장하지 않음)
            "saved": saved,
            "result_data": result_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")


@app.post("/api/nano/compose-from-paths")
async def compose_images_from_paths(
    crop_paths: List[str] = Form(...),
    prompt: str = Form(...),
    model: Optional[str] = Form("gemini-2.5-flash-image")
):
    """
    크롭된 이미지 경로들을 받아서 합성 (YOLO 서비스와 연동용)
    
    - **crop_paths**: 크롭된 이미지 파일 경로들
    - **prompt**: 합성 프롬프트
    - **model**: 사용할 Gemini 모델
    """
    try:
        if not crop_paths:
            raise HTTPException(status_code=400, detail="이미지 경로가 필요합니다.")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="프롬프트가 필요합니다.")
        
        # 이미지 파일들을 PIL Image로 변환
        imgs_in = []
        
        for crop_path in crop_paths:
            if not os.path.exists(crop_path):
                raise HTTPException(status_code=404, detail=f"이미지를 찾을 수 없습니다: {crop_path}")
            
            img = Image.open(crop_path)
            imgs_in.append(img)
        
        # Gemini API 호출
        contents = [prompt] + imgs_in
        
        response = client.models.generate_content(
            model=model,
            contents=contents
        )
        
        # 결과 이미지 base64 인코딩 (파일 저장하지 않음)
        saved = False
        result_base64 = None
        
        # 안전 파싱
        if getattr(response, "candidates", None):
            cand = response.candidates[0]
            if cand and cand.content and cand.content.parts:
                for part in cand.content.parts:
                    if getattr(part, "inline_data", None) and part.inline_data.data:
                        out_img = Image.open(BytesIO(part.inline_data.data))
                        saved = True
                        
                        # 이미지를 base64로 인코딩
                        buffer = BytesIO()
                        out_img.save(buffer, format='PNG')
                        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        break
        
        if not saved:
            raise HTTPException(
                status_code=500,
                detail="이미지 결과가 생성되지 않았습니다. 프롬프트를 더 명확하게 작성해주세요."
            )
        
        return {
            "message": "이미지 합성 완료",
            "output_path": "memory",  # 가상 경로 (파일 저장하지 않음)
            "saved": saved,
            "result_data": result_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")


# 파일 다운로드 엔드포인트 제거 - 모든 이미지 데이터는 base64로 응답에 포함됨


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
