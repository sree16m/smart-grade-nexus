from app.services.ingestion import IngestionService
import json

def test_safe_loads():
    service = IngestionService()
    
    # Case 1: Standard LaTeX with unescaped backslashes (What Gemini often does)
    # The \s and \p are invalid escapes in JSON
    invalid_json = '{"content": "Calculate the square root: $\\sqrt{x}$ and the value of $\\pi$.", "page_summary": "Math rules."}'
    # Wait, in Python string, '\\s' is a literal backslash and s. 
    # But if Gemini returns it, the raw string contains a backslash.
    raw_response_text = r'{"content": "Calculate the square root: $\sqrt{x}$ and the value of $\pi$.", "page_summary": "Math rules."}'
    
    print(f"Testing raw text: {raw_response_text}")
    
    try:
        json.loads(raw_response_text)
        print("Standard json.loads passed (Unexpected for this test)")
    except json.JSONDecodeError as e:
        print(f"Standard json.loads failed as expected: {e}")
        
    result = service._safe_json_loads(raw_response_text)
    print(f"Safe loads result: {result}")
    
    assert "content" in result
    assert "sqrt" in result["content"]
    assert "pi" in result["content"]
    print("Test Case 1 passed!")

    # Case 2: Markdown wrapper
    markdown_json = "```json\n" + raw_response_text + "\n```"
    print("\nTesting markdown wrapped JSON...")
    result_md = service._safe_json_loads(markdown_json)
    assert "content" in result_md
    print("Test Case 2 passed!")

if __name__ == "__main__":
    test_safe_loads()
