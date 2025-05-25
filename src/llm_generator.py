import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dotenv import load_dotenv
from rag_system import RagSystem

class LLMGenerator:
    def __init__(
        self,
        use_rag: bool = True,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        embedding_model: str = "bkai-foundation-models/vietnamese-bi-encoder",
        use_quantization: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Khởi tạo đối tượng LLMGenerator để tạo văn bản với hoặc không dùng RAG.

        Args:
            use_rag (bool): Sử dụng hệ thống RAG hay không (mặc định: True).
            model_name (str): Tên mô hình LLM từ Hugging Face.
            embedding_model (str): Tên mô hình nhúng cho RAG.
            use_quantization (bool): Sử dụng quantization 4-bit để giảm bộ nhớ.
            device (str): Thiết bị chạy mô hình (mặc định: "cuda" nếu có GPU, nếu không thì "cpu").
        """
        load_dotenv()

        # Khởi tạo các thuộc tính
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.use_rag = use_rag
        self.device = device
        self.use_quantization = use_quantization

        # Khởi tạo hệ thống RAG nếu được bật
        if self.use_rag:
            self.rag = RagSystem(model_name=self.embedding_model)

        print(f"Using LLM model for generation: {self.model_name}")
        print(f"Running on device: {self.device}")
        if self.use_rag:
            print(f"Using embedding model for RAG: {self.embedding_model}")

        # Khởi tạo tokenizer
        print(f"Loading tokenizer for {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            print("Tokenizer loaded.")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

        # Giải phóng bộ nhớ GPU nếu dùng CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Tải mô hình LLM với hoặc không dùng quantization
        print(f"Loading model {self.model_name}...")
        try:
            if self.use_quantization and self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="cuda",
                    trust_remote_code=True,
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=self.device if self.device == "cuda" else None,
                    trust_remote_code=True
                )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_text(self, input_prompt: str, max_length: int = 128, use_rag: bool = None) -> dict:
        """
        Tạo văn bản dựa trên prompt với hoặc không dùng RAG.

        Args:
            input_prompt (str): Prompt đầu vào để tạo văn bản.
            max_length (int): Độ dài tối đa của văn bản được tạo.
            use_rag (bool, optional): Sử dụng RAG hay không (nếu None, dùng giá trị self.use_rag).

        Returns:
            dict: Kết quả bao gồm prompt đã tăng cường (rag_prompt) và câu trả lời (rag_answer).
        """
        # Xác định xem có dùng RAG hay không
        should_use_rag = self.use_rag if use_rag is None else use_rag

        # Tăng cường prompt bằng RAG nếu cần
        if should_use_rag:
            rag_prompt = self.rag.rag_query(input_prompt)
            final_prompt = rag_prompt
        else:
            final_prompt = input_prompt

        # Chuẩn bị đầu vào cho mô hình
        model_inputs = self.tokenizer(final_prompt, return_tensors="pt")

        # Chuyển đầu vào sang thiết bị phù hợp
        if self.device == "cuda":
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        # Tạo văn bản
        try:
            output = self.model.generate(**model_inputs, max_new_tokens=max_length)
            print("Output:", output)
        except Exception as e:
            print(f"Error generating text: {e}")
            raise
        
        generated_answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Hậu xử lý để chỉ lấy câu trả lời ngắn gọn
        if "Câu trả lời:" in generated_answer:
            answer = generated_answer.split("Câu trả lời:")[-1].strip()
        else:
            answer = generated_answer

        # Lấy câu đầu tiên (nếu có nhiều câu)
        answer = answer.split('\n')[0].split('. ')[0].strip()
        
        
        return {
            "rag_prompt": final_prompt,
            "rag_answer": answer
        }