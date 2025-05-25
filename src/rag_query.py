import os
import sys
import json
from dotenv import load_dotenv
from rag_system import RagSystem
from llm_generator import LLMGenerator

# Tải biến môi trường từ file .env
load_dotenv()

# Định nghĩa các hằng số
EMBEDDING_MODEL = os.getenv("MODEL_EMBEDDING")
CHUNK_SIZE = 256
TEST_DATA_PATH = "./data/questions.json"
OUTPUT_JSON_PATH = "data/rag_prompt_result.json"

def process_queries_with_rag():
    """
    Xử lý các câu hỏi từ file JSON bằng RAG và LLM, sau đó lưu kết quả.
    """
    # Khởi tạo RAG và LLMGenerator
    rag_system = RagSystem(model_name=EMBEDDING_MODEL, chunk_size=CHUNK_SIZE)
    llm = LLMGenerator(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        embedding_model=EMBEDDING_MODEL,
        use_rag=True
    )

    # Đọc dữ liệu câu hỏi từ file JSON
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Xử lý từng câu hỏi
    for index, item in enumerate(data):
        query = item["question"]
        print(f"\n--- Query {index + 1}: {query} ---")
        
        # Tạo câu trả lời bằng RAG và LLM
        result = llm.generate_text(query, max_length=128)
        rag_prompt = result["rag_prompt"]
        rag_answer = result["rag_answer"]
        
        print(f"rag_answer {index + 1}:", rag_answer)
        item["rag_prompt"] = rag_prompt
        item["rag_answer"] = rag_answer

    # Lưu kết quả vào file JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    
    # Chuyển hướng stdout và stderr vào file log
    output_log = open("logs/rag_query_output.log", "w", encoding="utf-8")
    error_log = open("logs/rag_query_error.log", "w", encoding="utf-8")
    sys.stdout = output_log
    sys.stderr = error_log

    # Gọi hàm xử lý
    process_queries_with_rag()

    output_log.close()
    error_log.close()