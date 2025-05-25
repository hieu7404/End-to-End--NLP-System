import re
import pickle
import numpy as np
from faiss import IndexFlatL2, read_index
from embedding import Embedding

class RagSystem:
    def __init__(
        self,
        model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
        chunk_size: int = 256
    ):
        """
        Khởi tạo hệ thống RAG với mô hình nhúng và kích thước đoạn.

        Args:
            model_name (str): Tên mô hình nhúng từ Hugging Face.
            chunk_size (int): Kích thước tối đa của mỗi đoạn (số token).
        """
        # Khởi tạo đối tượng Embedding
        self.embedder = Embedding(model_name=model_name, chunk_size=chunk_size)

        # Tải chỉ mục FAISS và danh sách tài liệu
        self.faiss_index = read_index("data/faiss_index.bin")
        with open("data/chunks.pkl", "rb") as file:
            self.document_list = pickle.load(file)

    def clean_rag_output(self, raw_text: str) -> str:
        """
        Làm sạch văn bản đầu ra từ RAG.

        Args:
            raw_text (str): Văn bản thô cần làm sạch.

        Returns:
            str: Văn bản đã được làm sạch.
        """
        # Bước 1: Loại bỏ số trong ngoặc như [3, [5, [6, v.v.
        cleaned_text = re.sub(r'\[\d+', '', raw_text)

        # Bước 2: Loại bỏ tất cả token <unk>
        cleaned_text = cleaned_text.replace('<unk>', '')

        # Bước 3: Loại bỏ dấu câu thừa và chuẩn hóa khoảng trắng
        cleaned_text = re.sub(
            r'[,:;()]+', lambda m: m.group(0) if m.group(0) in (',', '.', '?', '!') else '', cleaned_text
        )
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Chuẩn hóa khoảng trắng

        # Bước 4: Loại bỏ khoảng trắng thừa ở đầu và cuối
        return cleaned_text.strip()

    def rag_query(self, query_text: str, top_k: int = 3) -> str:
        """
        Thực hiện truy vấn RAG để tạo prompt tăng cường.

        Args:
            query_text (str): Câu hỏi đầu vào.
            top_k (int): Số lượng tài liệu liên quan lấy ra.

        Returns:
            str: Prompt tăng cường với ngữ cảnh và câu hỏi.
        """
        # Tạo vector nhúng cho câu truy vấn
        query_embedding = self.embedder.embedding(query_text)
        query_embedding = np.array([query_embedding]).astype('float32')

        # Truy vấn FAISS để lấy top-k tài liệu gần nhất
        distances, indices = self.faiss_index.search(query_embedding, top_k)

        # Kiểm tra nếu không có kết quả hợp lệ
        if indices.size == 0 or len(indices[0]) == 0:
            return "Không thể tìm thấy thông tin liên quan."

        # Lấy các tài liệu liên quan
        retrieved_docs = [self.document_list[idx] for idx in indices[0]]

        # Làm sạch và giới hạn độ dài của tài liệu
        cleaned_documents = []
        for doc in retrieved_docs:
            cleaned_doc = self.clean_rag_output(doc)
            if len(cleaned_doc) > 256:
                cleaned_doc = cleaned_doc[:256]
            cleaned_documents.append(cleaned_doc)

        # Tạo ngữ cảnh
        context_text = " ".join(cleaned_documents)

        print(f"Context type: {type(context_text)}")

        # Tạo prompt tăng cường
        augmented_prompt = f"""
Dựa vào thông tin sau: {context_text}
Trả lời câu hỏi: {query_text}
Chỉ trả lời bằng một câu ngắn gọn, đúng trọng tâm, không lặp lại thông tin thừa hoặc prompt.
Câu trả lời: """

        return augmented_prompt