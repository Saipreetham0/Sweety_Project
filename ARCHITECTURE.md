# Resume AI Detector - Architecture Diagram

This document outlines the high-level architecture of the Resume AI Detector application.

## High-Level Overview

The application is composed of a **Next.js Frontend** for user interaction and a **FastAPI Backend** that handles file processing and AI analysis. The backend leverages several internal services for text extraction, preprocessing, feature engineering, and model inference.

## Architecture Diagram

```mermaid
graph TD
    subgraph Client ["Client Side"]
        Browser["User Browser"]
    end

    subgraph Frontend ["Frontend Container (Next.js)"]
        NextServer["Next.js Server"]
        UI["React UI Components"]
    end

    subgraph Backend ["Backend Container (FastAPI)"]
        API["FastAPI App (main.py)"]
        
        subgraph Routers ["API Routers"]
            UploadRouter["/upload (upload.py)"]
            AnalyzeRouter["/analyze (analyze.py)"]
        end
        
        subgraph Services ["AI Services Layer"]
            Extraction["Text Extraction (extraction.py)"]
            Preprocessing["Preprocessing (preprocessing.py)"]
            Features["Feature Eng. (features.py)"]
            Model["AI Model (model.py)"]
            WeakSup["Weak Supervision (weak_supervision.py)"]
        end
        
        subgraph Storage ["Local Storage (Volumes)"]
            UploadsDir["/app/uploads"]
            DataDir["/app/data"]
        end
    end

    %% Client Interactions
    Browser -->|HTTP/HTTPS| UI
    UI -->|API Requests| NextServer
    NextServer -->|Proxy/Direct| API

    %% API Routing
    API --> UploadRouter
    API --> AnalyzeRouter

    %% Upload Flow
    UploadRouter -->|Save File| UploadsDir
    
    %% Analysis Flow
    AnalyzeRouter -->|Trigger Analysis| Services
    
    %% Service Logic
    Services --> Extraction
    Extraction -->|Read File| UploadsDir
    Extraction --> Preprocessing
    Preprocessing --> Features
    Features --> Model
    Model -->|Load Artifacts| DataDir
    Services -->|Return Results| AnalyzeRouter

    %% Styling
    classDef frontend fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef backend fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef storage fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef client fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    class UI,NextServer frontend;
    class API,UploadRouter,AnalyzeRouter,Extraction,Preprocessing,Features,Model,WeakSup backend;
    class UploadsDir,DataDir storage;
    class Browser client;
```

## Component Description

### 1. Frontend (Next.js)
- **Role**: Provides the user interface for uploading resumes and viewing analysis results.
- **Tech Stack**: Next.js 16, React 19, Tailwind CSS.
- **Port**: 3000.

### 2. Backend (FastAPI)
- **Role**: Exposes REST API endpoints to handle resume uploads and run AI inference.
- **Tech Stack**: FastAPI, Torch, Transformers, Scikit-learn.
- **Port**: 8000.

#### Key Modules:
- **Routers**:
    - `upload.py`: Handles file upload and validation.
    - `analyze.py`: Orchestrates the analysis pipeline.
- **Services**:
    - `extraction.py`: Extracts raw text from PDF and DOCX files.
    - `preprocessing.py`: Cleans and normalizes text data.
    - `features.py`: Generates numerical features from text for the model.
    - `model.py`: Loads the trained model and performs inference.

### 3. Data Storage
- **Uploads**: ephemeral storage for uploaded documents (`/backend/uploads`).
- **Data**: persistant storage for model weights, configuration, and potentially training data (`/backend/data`).
