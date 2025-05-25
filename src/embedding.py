import re
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

class Embedding:
    def __init__(
        self,
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        chunk_size=256
    ):
        """
        Khởi tạo đối tượng Embedding với mô hình nhúng và kích thước đoạn.

        Args:
            model_name (str): Tên mô hình nhúng từ Hugging Face.
            chunk_size (int): Kích thước tối đa của mỗi đoạn (số token).
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()

    
    def chunk_text(self, input_text: str) -> list:
        """
        Chia đoạn văn bản theo cách tối ưu dựa trên dấu chấm, xuống dòng và độ dài token.

        Args:
            input_text (str): Văn bản đầu vào cần chia đoạn.

        Returns:
            list: Danh sách các đoạn văn bản đã được chia.
        """
        def tokenize_and_check_length(text_segment: str) -> tuple:
            tokens = self.tokenizer.encode(text_segment, add_special_tokens=False)
            return tokens, len(tokens)

        def decode_tokens(tokens: list) -> str:
            return self.tokenizer.decode(tokens)

        def split_by_token(tokens: list, max_length: int) -> list:
            if len(tokens) <= max_length * 1.1:
                return [tokens]
            else:
                mid_point = len(tokens) // 2
                return (split_by_token(tokens[:mid_point], max_length) +
                        split_by_token(tokens[mid_point:], max_length))

        def process_chunk(text_segment: str) -> list:
            tokens, length = tokenize_and_check_length(text_segment)

            if length <= self.chunk_size * 1.1:
                return [decode_tokens(tokens)]
            else:
                # Chia theo token nếu đoạn quá dài
                token_chunks = split_by_token(tokens, self.chunk_size)
                return [decode_tokens(t) for t in token_chunks]

        import re
        rough_chunks = re.split(r'\.\s+|\n+', input_text)
        rough_chunks = [chunk.strip() for chunk in rough_chunks if chunk.strip()]

        final_chunks = []
        for chunk in rough_chunks:
            final_chunks.extend(process_chunk(chunk))

        return final_chunks

    def embedding(self, input_text: str) -> list:
        """
        Tạo vector nhúng cho văn bản đầu vào.

        Args:
            input_text (str): Văn bản cần nhúng.

        Returns:
            list: Vector nhúng dưới dạng danh sách.
        """
        return self.embedding_model.encode(input_text).tolist()
