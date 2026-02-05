
import asyncio
import json
import os
from dotenv import load_dotenv
from typing import List, Dict

# Load env vars
load_dotenv()

# Import Services (Simulating API calls)
from app.services.agents import TopicAgent, GradingAgent
from app.main import AnswerSheet  # Pydantic model

# The User's Input JSON (simulated)
input_data = [
  {
    "answer_sheet_id": "dac590df-66e2-4d1a-9f40-929393b798e1",
    "qp_id": "1fdee489-c02b-42b5-b7a4-d50eb42258fc",
    "status": "pending evaluation",
    "exam_details": {
      "board": "BSEAP",
      "class_level": 9,
      "subject": "maths",
      "exam_code": "920"
    },
    "student_details": {
      "name": "P. Varshant Krishna",
      "id": "11",
      "class": "9th A",
      "subject": "Math S"
    },
    "grading_summary": {
      "teacher_awarded_marks": 35
    },
    "responses": [
      {
        "q_no": 1,
        "question_context": {
          "text_primary": {
            "en": "According to Euclid Geometry, How many lines can be drawn through a point and parallel to a given line?",
            "regional": "యూక్లిడ్ జ్యామితి ప్రకారం, ఇచ్చిన రేఖకు సమాంతరంగా ఒక బిందువు ద్వారా ఎన్ని సరళరేఖలను గీయవచ్చు?"
          },
          "type": "mcq",
          "max_marks": 1,
          "options": [
            {"id": "A", "text": {"en": "Two"}},
            {"id": "B", "text": {"en": "Infinite"}},
            {"id": "C", "text": {"en": "Exactly one"}},
            {"id": "D", "text": {"en": "None"}}
          ]
        },
        "student_answer": {
          "text": "B - Infinite",
          "diagram_description": "",
          "marks_awarded": 0,
          "is_correct": None
        }
      }
    ]
  }
]

async def process_sheet():
    print("--- Starting Processing ---")
    
    # 1. Parse into Pydantic Model (Validation)
    request_list = [AnswerSheet(**item) for item in input_data]
    
    for i, sheet in enumerate(request_list):
        print(f"Processing Sheet: {sheet.answer_sheet_id}")
        
        # Initialize Agents
        topic_agent = TopicAgent(subject=sheet.exam_details.subject)
        grading_agent = GradingAgent(subject=sheet.exam_details.subject)
        
        # Prepare for Categorization
        questions_dicts = []
        for r in sheet.responses:
            questions_dicts.append({
                "id": str(r.q_no),
                "text": r.question_context.text_primary.en
            })
            
        # Run Categorization (API Call 1)
        print("Categorizing...")
        topics = await topic_agent.categorize(questions_dicts)
        
        # Run Evaluation (API Call 2)
        print("Evaluating...")
        grades = []
        for r in sheet.responses:
             # Construct prompt data
             q_text = r.question_context.text_primary.en
             ans_text = r.student_answer.text
             if r.student_answer.diagram_description:
                 ans_text += f"[Diagram: {r.student_answer.diagram_description}]"
                 
             grade = await grading_agent.evaluate(
                 question=q_text,
                 answer=ans_text,
                 max_marks=r.question_context.max_marks or 1
             )
             grades.append({"q_no": r.q_no, "result": grade})

        # --- MERGING RESULTS (The "Magic" Part) ---
        original_json = input_data[i]
        
        # Merge Topics
        # (Assuming we might want to attach topics to questions, though user JSON didn't have a field for it)
        # We will add a 'topic_analysis' field to the response item for visibility
        topic_map = {t['question_id']: t for t in topics}
        
        # Merge Grades
        grade_map = {g['q_no']: g['result'] for g in grades}
        
        total_marks = 0
        
        for response in original_json['responses']:
            q_id = str(response['q_no'])
            q_no_int = response['q_no']
            
            # Enrich with Topic
            if q_id in topic_map:
                response['topic_analysis'] = topic_map[q_id]
                
            # Enrich with Grade
            if q_no_int in grade_map:
                res = grade_map[q_no_int]
                # Handle list/dict just in case
                if isinstance(res, list) and len(res) > 0: res = res[0]
                
                # Update fields
                response['student_answer']['marks_awarded'] = res.get('score', 0)
                response['student_answer']['feedback'] = res.get('feedback', '')
                
                # Determine is_correct
                # Simple logic: if marks > 0 and marks == max_marks, it's correct. 
                # Or just use score > 0. Let's say score == max_marks is full correct.
                max_m = response['question_context'].get('max_marks', 1)
                response['student_answer']['is_correct'] = (res.get('score', 0) == max_m)
                
                total_marks += res.get('score', 0)
                
        # Calculate final total
        original_json['grading_summary']['ai_awarded_marks'] = total_marks
        original_json['status'] = "evaluated"

    print("--- Processing Complete ---")
    with open("demo_output.json", "w", encoding="utf-8") as f:
        json.dump(input_data, f, indent=2, ensure_ascii=False)
    print("Output saved to demo_output.json")

if __name__ == "__main__":
    asyncio.run(process_sheet())
