from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import dashboard_router, interview_router

app = FastAPI(title="SkillProof AI Engine", version="1.0.0")

# CORS Setup: Frontend (React/Next.js) ko backend se API call karne allow karta hai
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hackathon ke liye sab allow kar do
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dono routers ko yahan attach kar diya
app.include_router(dashboard_router.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(interview_router.router, prefix="/api/interview", tags=["Interview"])

# Ek simple health check API taaki pata chale server chal raha hai
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SkillProof AI Engine is running! 🚀"}