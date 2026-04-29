from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, UploadFile, File # Updated imports
import PyPDF2
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai
import psycopg2

# This loads the hidden keys from your .env file
load_dotenv()

# Securely fetch the keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_URL = os.getenv("SUPABASE_DB_URL")

# Configure your tools using the secured variables
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel('gemini-2.5-flash')
llm = genai.GenerativeModel('gemini-2.5-flash')
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FastAPI App
app = FastAPI(title="Patent-Scout API")

# Allow the frontend HTML file to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you would lock this down to your specific website URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import Optional # <-- Add this at the top with your other imports

# --- DATA MODELS ---
class SearchRequest(BaseModel):
    query: str
    assignee: Optional[str] = None # <-- Update this line

class TranslateRequest(BaseModel):
    text: str

# --- API ENDPOINTS ---
@app.post("/search")
def search_patents(req: SearchRequest):
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        
        query_vector = encoder.encode(req.query).tolist()
        
        sql = """
            SELECT patent_id, title, abstract, assignee, priority_date,
                   1 - (embedding <=> %s::vector) as similarity
            FROM real_patents
            WHERE 1=1
        """
        params = [query_vector]
        
        if req.assignee:
            sql += " AND assignee ILIKE %s"
            params.append(f"%{req.assignee}%")
            
        sql += " ORDER BY similarity DESC LIMIT 5;"
        
        cursor.execute(sql, params)
        results = cursor.fetchall()
        conn.close()
        
        # Format the SQL results into a clean JSON response
        formatted_results = []
        for pat in results:
            formatted_results.append({
                "patent_id": pat[0],
                "title": pat[1],
                "abstract": pat[2],
                "assignee": pat[3],
                "date": str(pat[4]),
                "similarity": round(pat[5] * 100, 1)
            })
            
        return {"status": "success", "data": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
def translate_legalese(req: TranslateRequest):
    try:
        prompt = f"You are a helpful AI. Rewrite this complex medical/AI patent abstract into a simple, 2-sentence summary that a high schooler could understand:\n\n{req.text}"
        response = llm.generate_content(prompt)
        return {"status": "success", "translation": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/fto_check")
async def check_freedom_to_operate(file: UploadFile = File(...)):
    try:
        # 1. Read the uploaded PDF file
        content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"
            
        # 2. Chunking the Document (Split by paragraphs)
        # We only keep chunks longer than 50 characters to avoid embedding meaningless fragments
        chunks = [chunk.strip() for chunk in full_text.split('\n\n') if len(chunk.strip()) > 50]
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract readable text from PDF.")

        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        
        high_risk_collisions = []
        seen_patents = set() # To avoid duplicate warnings for the same patent
        
        # 3. Analyze each chunk against the vector database
        for chunk in chunks:
            chunk_vector = encoder.encode(chunk).tolist()
            
            # Find the closest match for this specific paragraph
            cursor.execute("""
                SELECT patent_id, title, assignee, 1 - (embedding <=> %s::vector) as similarity
                FROM real_patents
                ORDER BY similarity DESC LIMIT 1;
            """, (chunk_vector,))
            
            match = cursor.fetchone()
            
            # 4. The Risk Threshold (Flag anything over 45% similar)
            if match:
                sim_score = match[3]
                pat_id = match[0]
                
                # Lowered from 0.75 to 0.45
                if sim_score > 0.45 and pat_id not in seen_patents:
                    seen_patents.add(pat_id)
                    high_risk_collisions.append({
                        # Lowered the red-alert threshold from 0.85 to 0.55
                        "risk_level": "🔴 HIGH RISK" if sim_score > 0.55 else "🟡 MEDIUM RISK",
                        "draft_excerpt": chunk[:150] + "...", 
                        "colliding_patent": pat_id,
                        "patent_title": match[1],
                        "assignee": match[2],
                        "similarity": round(sim_score * 100, 1)
                    })
                    
        conn.close()
        
        # Sort risks from highest to lowest similarity
        high_risk_collisions.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            "status": "success", 
            "total_chunks_analyzed": len(chunks),
            "collisions_found": len(high_risk_collisions),
            "report": high_risk_collisions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))