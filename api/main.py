from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles # 🆕 StaticFiles import
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse # 🆕 RedirectResponse 추가
from api.routers import coin, dashboard, market
import os

app = FastAPI(
    title="Auto Trader Shadow Dashboard API",
    description="섀도우 트레이딩(가상 매매) 데이터를 제공하는 API 서버입니다.",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🆕 정적 파일 서빙 설정 (dashboard 폴더를 /dashboard 경로로 접근 가능하게 함)
# 예: http://localhost:8001/dashboard/index.html
# 예: http://localhost:8001/dashboard/static/css/style.css
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")

# 디버깅용 출력 (컨테이너 로그에서 확인 가능)
print(f"Checking static directory: {static_dir}")

if os.path.exists(static_dir):
    print(f"Mounting dashboard from: {static_dir}")
    app.mount("/dashboard", StaticFiles(directory=static_dir, html=True), name="dashboard")
else:
    print(f"WARNING: Dashboard directory not found at {static_dir}")

# 라우터 등록
app.include_router(coin.router)
app.include_router(dashboard.router)
app.include_router(market.router)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Shadow Trader API is running"}

@app.get("/dashboard") # 🆕 /dashboard 접속 시 index.html로 이동
def dashboard_redirect():
    return RedirectResponse(url="/dashboard/index.html")

if __name__ == "__main__":
    import uvicorn
    # 로컬 개발용 실행 (reload=True)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8001, reload=True)

