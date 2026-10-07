from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import dashboard_router, interview_router

app = FastAPI(title="SkillProof AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(interview_router.router, prefix="/api/interview", tags=["Interview"])


@app.get("/health")
def health():
    return {"status": "ok"}