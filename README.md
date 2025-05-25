ASSIGNMENT END-TO-END - NLP-SYSTEM BUILDING
Dự án này phát triển một hệ thống hỏi đáp tự động sử dụng mô hình Retrieval-Augmented Generation (RAG) dựa trên dữ liệu từ các trang web của Đại học Quốc gia Hà Nội (VNU) và Trường Đại học Công nghệ (UET). Hệ thống bao gồm các bước thu thập dữ liệu, xử lý văn bản, tạo embedding với mô hình bkai-foundation-models/vietnamese-bi-encoder và sinh câu trả lời bằng mô hình Llama-3.2-1B-Instruct.
CẤU TRÚC DỰ ÁN  

data/: Thư mục chứa các tệp dữ liệu:

data_source.csv: Danh sách các URL để thu thập dữ liệu.  
questions.json: Dữ liệu kiểm tra (các câu hỏi).  
Các tệp đã xử lý: all_data.txt, data_clean.txt, data.txt, faiss_index.bin, chunks.pkl.  


scr/: Thư mục chứa source code

crawl_data.py: Thu thập dữ liệu từ các URL được liệt kê trong data_source.csv.  
processing_data.py: Làm sạch dữ liệu thô thu thập được.  
data_processor.py: Chia nhỏ văn bản thành các đoạn (chunk) và tạo chỉ mục FAISS.  
embedding.py: Tạo embedding cho văn bản bằng mô hình bkai-foundation-models/vietnamese-bi-encoder.  
rag_system.py: Truy xuất các tài liệu liên quan sử dụng chỉ mục FAISS.  
llm_generator.py: Sinh câu trả lời dựa trên câu hỏi và tài liệu truy xuất được bằng mô hình Llama-3.2-1B-Instruct.  
run_rag.py: Tự động hóa toàn bộ quy trình RAG, từ xử lý câu hỏi trong questions.json, sinh câu trả lời, đến lưu kết quả vào system_output.txt và tệp JSON của mô hình.  
evaluate.py: Đánh giá hiệu suất hệ thống (đạt độ chính xác 32.1%).  



# HƯỚNG DẪN SỬ DỤNG

Chạy crawl_data.py để thu thập dữ liệu từ các URL.  
Chạy processing_data.py để làm sạch dữ liệu thô.  
Chạy data_processor.py để chia nhỏ văn bản và tạo chỉ mục FAISS.  
Sử dụng rag_system.py và llm_generator.py để thực hiện hỏi đáp.  
Chạy run_rag.py để tự động hóa quy trình hỏi đáp và lưu kết quả.  
Chạy evaluate.py để đánh giá hiệu suất hệ thống.  

THÀNH VIÊN NHÓM

Bùi Duy Hải - 22022575  
Lê Trung Hiếu - 22022576  
Nguyễn Lâm Tùng Bách - 22022640  
