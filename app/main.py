from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json

from app.core.config import settings
from app.services.ingestion import IngestionService
from app.services.agents import TopicAgent, GradingAgent

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json")

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation Error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

# --- Pydantic Models ---
# --- Pydantic Models ---

class TextContent(BaseModel):
    en: str
    regional: Optional[str] = None

class OptionItem(BaseModel):
    id: str
    text: TextContent

class QuestionContext(BaseModel):
    text_primary: TextContent
    type: str # mcq, image, formula, etc.
    max_marks: Optional[float] = None
    options: List[OptionItem] = []

class StudentAnswer(BaseModel):
    text: str
    diagram_description: Optional[str] = ""
    marks_awarded: Optional[float] = None
    is_correct: Optional[bool] = None
    feedback: Optional[str] = None

class ResponseItem(BaseModel):
    q_no: int # This acts as ID
    question_context: QuestionContext
    student_answer: StudentAnswer
    topic_analysis: Optional[Dict[str, Any]] = None

class ExamDetails(BaseModel):
    subject: str
    board: Optional[str] = None
    class_level: Optional[int] = None

class AnswerSheet(BaseModel):
    # This matches the root of the user payload
    answer_sheet_id: Optional[str] = None
    exam_details: ExamDetails
    responses: List[ResponseItem]
    # Allow other fields like student_details, etc.
    model_config = {"extra": "ignore"}

# --- Services ---
ingestion_service = IngestionService()

# --- Endpoints ---

@app.get(f"{settings.API_V1_STR}/knowledge/books")
async def get_books():
    """Lists all uploaded books (aggregated by metadata)."""
    try:
        books = await ingestion_service.get_uploaded_books()
        return {"status": "success", "data": books}
    except Exception as e:
         import traceback
         traceback.print_exc()
         raise HTTPException(status_code=500, detail=str(e))

@app.delete(f"{settings.API_V1_STR}/knowledge/clear")
async def clear_knowledge_base():
    """⚠️ DANGER: Deletes ALL documents."""
    try:
        await ingestion_service.delete_all_documents()
        return {"status": "success", "message": "Knowledge Base cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete(f"{settings.API_V1_STR}/knowledge/books/{{book_name}}")
async def delete_book(book_name: str):
    """Deletes a specific book by name."""
    try:
        await ingestion_service.delete_book(book_name)
        return {"status": "success", "message": f"Book '{book_name}' deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_STR}/knowledge/ingest")
async def ingest_knowledge(
    background_tasks: BackgroundTasks,
    subject: Optional[str] = Form(None),
    book_name: Optional[str] = Form(None),
    board: Optional[str] = Form(None),
    school: Optional[str] = Form(None),
    student_class: Optional[str] = Form(None, alias="class"),
    semester: Optional[str] = Form(None, alias="semester"),
    ingestion_mode: str = Form("standard"),
    file: UploadFile = File(...)
):
    """Parses PDF and stores embeddings in Supabase."""
    try:
        content = await file.read()
        metadata = {
            "subject": subject, 
            "book_name": book_name, 
            "filename": file.filename,
            "board": board,
            "school": school,
            "class": student_class,
            "semester": semester,
            "ingestion_mode": ingestion_mode
        }
        
        result = await ingestion_service.process_document(content, metadata, background_tasks)
        return {"status": "success", "data": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_STR}/intelligence/categorize")
async def categorize_questions(request_list: List[AnswerSheet]):
    """Maps questions to topics. Accepts a List of Answer Sheet objects."""
    all_results = []
    
    for request in request_list:
        agent = TopicAgent(subject=request.exam_details.subject)
        
        # Transform complex request to simple list for Agent
        questions_dicts = []
        for r in request.responses:
            # Use English text by default
            q_text = r.question_context.text_primary.en
            questions_dicts.append({
                "id": str(r.q_no),
                "text": q_text
            })
        
        results = await agent.categorize(questions_dicts)
        all_results.extend(results)
        
    return {"mappings": all_results}

@app.post(f"{settings.API_V1_STR}/intelligence/evaluate")
async def evaluate_answer(request_list: List[AnswerSheet]):
    """Grades a batch of student answer sheets using the Knowledge Base."""
    all_grades = []
    
    for sheet in request_list:
        agent = GradingAgent(subject=sheet.exam_details.subject)
        
        for r in sheet.responses:
            # Extract grading inputs
            q_text = r.question_context.text_primary.en
            student_ans = r.student_answer.text
            # Use diagram description if available
            if r.student_answer.diagram_description:
                student_ans += f" [Diagram: {r.student_answer.diagram_description}]"
                
            max_marks = r.question_context.max_marks
            
            # Grade it
            grade_result = await agent.evaluate(
                question=q_text,
                answer=student_ans,
                max_marks=max_marks
            )
            
            # Append ID info to result
            # Flatten response slightly for easier consumption, or keep it nested.
            # Let's keep the new structure but ensure 'score' is top-level for backwards compat if needed by frontend
            # Actually, let's just expose the new structure directly but maybe add 'score' at top level for safety?
            # User wants "correct answer, details of subject..." so returning the whole object is best.
            # But the 'all_grades' list is what returns.
            
            # Ensure we handle the "grading" key
            if "grading" in grade_result:
                 # Flatten for compatibility with expected "score" in response? 
                 # Or just return as is. Let's return as is, BUT ensure we don't break simple clients.
                 # The user wants details.
                 pass
            
            grade_result["answer_sheet_id"] = sheet.answer_sheet_id
            grade_result["q_no"] = r.q_no
            all_grades.append(grade_result)

    return {"results": all_grades}

@app.post(f"{settings.API_V1_STR}/intelligence/analyze-full-sheet")
async def analyze_full_sheet(request_list: List[AnswerSheet]) -> List[AnswerSheet]:
    """
    Unified Endpoint: Categorizes questions AND Grades answers.
    Returns the enriched Answer Sheets with 'topic_analysis', 'marks_awarded', 'feedback', etc. populated.
    """
    for sheet in request_list:
        subject = sheet.exam_details.subject
        topic_agent = TopicAgent(subject=subject)
        grading_agent = GradingAgent(subject=subject)
        
        # --- 1. Categorization ---
        # Prepare list for TopicAgent
        questions_dicts = []
        for r in sheet.responses:
            questions_dicts.append({
                "id": str(r.q_no),
                "text": r.question_context.text_primary.en
            })
        
        # Get Topics
        topics = await topic_agent.categorize(questions_dicts)
        topic_map = {t['question_id']: t for t in topics}
        
        # --- 2. Evaluation & Enrichment ---
        for r in sheet.responses:
            # Enrichment: Topics
            if str(r.q_no) in topic_map:
                r.topic_analysis = topic_map[str(r.q_no)]
            
            # Enrichment: Grading
            q_text = r.question_context.text_primary.en
            student_ans = r.student_answer.text
            if r.student_answer.diagram_description:
                student_ans += f" [Diagram: {r.student_answer.diagram_description}]"
            
            # Call Grading Agent
            grade_result = await grading_agent.evaluate(
                question=q_text,
                answer=student_ans,
                max_marks=r.question_context.max_marks or 1
            )
            
            # Handle List vs Dict response from AI
            if isinstance(grade_result, list):
                if len(grade_result) > 0:
                     grade_result = grade_result[0]
                else:
                     grade_result = {"score": 0, "feedback": "Error: Empty AI response", "citation": None}
            
            # Update StudentAnswer fields
            # Extract from 'grading' key if present (new format), else assume flat (old format fallback)
            grading_data = grade_result.get("grading", grade_result)
            
            r.student_answer.marks_awarded = grading_data.get("score", 0)
            r.student_answer.feedback = grading_data.get("feedback", "")
            # Determine correctness (simple logic: score == max_marks)
            max_m = r.question_context.max_marks or 1
            r.student_answer.is_correct = (r.student_answer.marks_awarded == max_m)
            
    return request_list

@app.get("/")
def health_check():
    return {"status": "SmartGrade Nexus is Online", "version": "1.0.0"}
