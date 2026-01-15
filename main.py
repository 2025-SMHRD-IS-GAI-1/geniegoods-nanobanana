import os
import base64
import asyncio
import concurrent.futures
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from google import genai
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import uvicorn
from unicodedata import category

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
    category: str = Form(...),
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

        print(f"category: {category}")
        
        # 이미지 파일들을 메모리에서 PIL Image로 변환
        imgs_in = []
        
        for file in files:
            # 파일을 메모리에서 읽기
            content = await file.read()
            
            # PIL Image로 직접 로드
            img = Image.open(BytesIO(content))
            imgs_in.append(img)

        # category 에 따라서 규격 정할것
        if category == "핸드폰케이스":
            prompt += "\n규격: iPhone 13 하드케이스 기준"
        elif category == "머그컵":
            prompt += "\n규격: 머그컵 330ml(11oz), 인쇄영역 20x9cm"
        elif category == "키링":
            prompt += "\n규격: 아크릴 키링 50x50mm, 두께 3mm"
        elif category == "그립톡":
            prompt += "\n규격: 그립톡 원형 Ø40mm(두께 약 6mm), 인쇄영역 Ø38mm"
        elif category == "카드 지갑":
            prompt += "\n규격: 카드 지갑 65x100mm 슬림 카드홀더 기준"


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


@app.post("/api/nano/sample")
async def sample_images(
    result_image: UploadFile = File(...),
    category: str = Form(...),
):
    try:
        if not result_image:
            raise HTTPException(status_code=400, detail="이미지 파일이 필요합니다.")
        
        print(f"category: {category}")

        # 파일 내용 읽기
        content = await result_image.read()
        
        if not content or len(content) == 0:
            raise HTTPException(status_code=400, detail="이미지 파일이 비어있습니다.")
        
        # PIL Image로 로드 (에러 처리 포함)
        try:
            img = Image.open(BytesIO(content))
            # 이미지가 실제로 로드 가능한지 확인하기 위해 convert 시도
            img = img.convert('RGB')  # RGB로 변환하여 검증
        except Exception as img_error:
            raise HTTPException(
                status_code=400, 
                detail=f"이미지 파일을 읽을 수 없습니다: {str(img_error)}"
            )
        
        prompts = [
                f"객체를 변경하지 말고, 현재 시안의 분위기와 색감은 유지하되 포즈와 장면을 바꿔서 새롭게 구성해줘. "
                f"이 이미지는 반드시 굿즈 '{category}'에 적용된 형태의 디자인 시안으로 표현해줘. "
                f"사진처럼 보이는 결과는 생성하지 말고, 흰 배경에 굿즈만 보이도록 출력해줘.",
                f"객체를 변경하지 말고, 현재 시안의 분위기와 색감은 유지하되 포즈와 장면을 바꿔서 새롭게 구성해줘. "
                f"이 이미지는 반드시 굿즈 '{category}'에 적용된 형태의 디자인 시안으로 표현해줘. "
                f"사진처럼 보이는 결과는 생성하지 말고, 흰 배경에 굿즈만 보이도록 출력해줘."
        ]
        
        # 비동기 병렬 호출
        def call_gemini(prompt_text):
            # 각 호출마다 새로운 BytesIO로 이미지 재생성
            img_copy = Image.open(BytesIO(content))
            contents = [prompt_text] + [img_copy]
            return client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=contents
            )
        
        # ThreadPoolExecutor로 병렬 실행
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(call_gemini, prompts[0]),
                executor.submit(call_gemini, prompts[1])
            ]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        result_base64_list = []
        for response in responses:
            if getattr(response, "candidates", None):
                cand = response.candidates[0]
                if cand and cand.content and cand.content.parts:
                    for part in cand.content.parts:
                        if getattr(part, "inline_data", None) and part.inline_data.data:
                            out_img = Image.open(BytesIO(part.inline_data.data))
                            buffer = BytesIO()
                            out_img.save(buffer, format='PNG')
                            result_base64_list.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
                            break
        
        if len(result_base64_list) < 2:
            raise HTTPException(
                status_code=500,
                detail=f"이미지 생성에 실패했습니다. 생성된 이미지: {len(result_base64_list)}개"
            )
        
        return {
            "result_data_list": result_base64_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
