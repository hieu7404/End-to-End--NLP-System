import warnings
import os
import sys
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from faiss import IndexFlatL2
from embedding import Embedding

# Bỏ qua cảnh báo UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# Tải biến môi trường từ file .env
load_dotenv()

# Định nghĩa các hằng số
DATA_FILE_PATH = "./data/data.txt"
EMBEDDING_MODEL = os.getenv("MODEL_EMBEDDING")
CHUNK_SIZE = 256

# Kiểm tra xem biến môi trường EMBEDDING_MODEL có được thiết lập không
if not EMBEDDING_MODEL:
    raise ValueError("EMBEDDING_MODEL không được thiết lập trong file .env. Vui lòng thêm biến EMBEDDING_MODEL vào file .env")

def process_and_store_data(input_file_path: str, embedding_model: str, chunk_size: int) -> None:
    """
    Xử lý dữ liệu thô, chia đoạn, tạo nhúng và lưu chỉ mục FAISS.

    Args:
        input_file_path (str): Đường dẫn đến file dữ liệu thô.
        embedding_model (str): Tên mô hình nhúng từ Hugging Face.
        chunk_size (int): Kích thước tối đa của mỗi đoạn (số token).
    """
    # Đọc dữ liệu từ file
    with open(input_file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    # Khởi tạo đối tượng Embedding và chia đoạn văn bản
    embedder = Embedding(model_name=embedding_model, chunk_size=chunk_size)
    text_chunks = embedder.chunk_text(raw_text)

    # In các đoạn văn bản đã chia với mã hóa UTF-8
    for chunk in text_chunks:
        print(chunk.encode('utf-8').decode('utf-8', errors='replace'))

    # Tạo vector nhúng cho các đoạn
    embeddings = embedder.embedding(text_chunks)
    embeddings = np.array(embeddings).astype('float32')  # FAISS yêu cầu kiểu float32

    # Tạo và lưu chỉ mục FAISS
    embedding_dimension = embeddings.shape[1]
    faiss_index = IndexFlatL2(embedding_dimension)
    faiss_index.add(embeddings)

    # Tạo thư mục data nếu chưa tồn tại và lưu chỉ mục FAISS cùng các đoạn
    os.makedirs("data", exist_ok=True)
    faiss.write_index(faiss_index, "data/faiss_index.bin")
    with open("data/chunks.pkl", "wb") as chunk_file:
        pickle.dump(text_chunks, chunk_file)

    print(f"Đã lưu FAISS index và chunks: {faiss_index.ntotal} chunks được xử lý.")

if __name__ == "__main__":
    # Tạo thư mục logs nếu chưa tồn tại
    os.makedirs("logs", exist_ok=True)
    
    # Chuyển hướng stdout và stderr vào file log
    output_log = open("logs/data-processor_output.log", "w", encoding="utf-8")
    error_log = open("logs/data-processor_error.log", "w", encoding="utf-8")
    sys.stdout = output_log
    sys.stderr = error_log

    # Gọi hàm xử lý dữ liệu
    process_and_store_data(DATA_FILE_PATH, EMBEDDING_MODEL, CHUNK_SIZE)

    # Đóng file log
    output_log.close()
    error_log.close()