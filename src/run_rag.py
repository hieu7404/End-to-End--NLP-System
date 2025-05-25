import os
import sys
import json
from dotenv import load_dotenv
from llm_generator import LLMGenerator

# Tải biến môi trường từ file .env
load_dotenv()

# Định nghĩa các hằng số
EMBEDDING_MODEL = os.getenv("MODEL_EMBEDDING")
CHUNK_SIZE = 256
TEST_DATA_PATH = "./data/questions.json"
SYSTEM_OUTPUT_PATH = "system_output/system_output.txt"

# Danh sách mô hình
MODEL_LIST = [{"name": "llama-3.2-1b-instruct", "link": "meta-llama/Llama-3.2-1B-Instruct"}]

def run_rag_process(model_index: int = 0):
    """
    Chạy quy trình RAG để xử lý các câu hỏi và lưu kết quả.

    Args:
        model_index (int): Chỉ số của mô hình trong danh sách (mặc định: 0).
    """
    # Tải thông tin mô hình
    model_info = MODEL_LIST[model_index]

    # Khởi tạo LLMGenerator
    llm = LLMGenerator(
        model_name=model_info["link"],
        embedding_model=EMBEDDING_MODEL,
        use_rag=True
    )

    # Đọc dữ liệu câu hỏi từ file JSON
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    # Danh sách câu trả lời để lưu vào system_output.txt
    system_outputs = []

    # Xử lý từng câu hỏi
    for index, item in enumerate(data):
        query = item["question"]
        print(f"\n--- Query {index + 1}: {query.encode('utf-8').decode('utf-8', errors='replace')} ---")
        answer = llm.generate_text(query, max_length=128)
        rag_answer = answer["rag_answer"].encode('utf-8').decode('utf-8', errors='replace')
        print(f"rag_answer {index + 1}:", rag_answer)
        item["rag_prompt"] = answer["rag_prompt"]
        item["rag_answer"] = rag_answer
        system_outputs.append(rag_answer)

    # Lưu kết quả vào file JSON
    json_output_path = f"data/{model_info['name']}-result.json"
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

    # Lưu câu trả lời vào file system_output.txt
    with open(SYSTEM_OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        for output in system_outputs:
            output_file.write(f"{output}\n")

if __name__ == "__main__":
    # Tạo thư mục logs nếu chưa tồn tại
    os.makedirs("logs", exist_ok=True)
    
    # Tạo thư mục system_output nếu chưa tồn tại
    os.makedirs("system_output", exist_ok=True)
    
    # Chuyển hướng stdout và stderr vào file log
    output_log = open("logs/run_rag_output.log", "w", encoding="utf-8")
    error_log = open("logs/run_rag_error.log", "w", encoding="utf-8")
    sys.stdout = output_log
    sys.stderr = error_log

    # Gọi hàm xử lý
    run_rag_process(model_index=0)

    # Đóng file log
    output_log.close()
    error_log.close()