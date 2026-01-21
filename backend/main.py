from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import upload, analyze

app = FastAPI(title="Resume AI Detector API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Resume AI Detector API is running"}

app.include_router(upload.router)
app.include_router(analyze.router)
