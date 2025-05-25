import json
import re

# Đường dẫn đến các file
REFERENCE_DATA_PATH = "./data/questions.json"
PREDICTION_PATH = "./system_output/system_output.txt"

def clean_text(text):
    text = text.strip().lower()
    return text

def evaluate_model():
    # Đọc dữ liệu tham chiếu từ questions.json
    try:
        with open(REFERENCE_DATA_PATH, "r", encoding="utf-8") as ref_file:
            reference_data = json.load(ref_file)
        reference_answers = [item["reference_answer"] for item in reference_data]
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file {REFERENCE_DATA_PATH}")
        return
    except json.JSONDecodeError:
        print(f"❌ Lỗi: File {REFERENCE_DATA_PATH} không đúng định dạng JSON")
        return
    except Exception as e:
        print(f"❌ Lỗi khi đọc {REFERENCE_DATA_PATH}: {e}")
        return

    # Đọc file system_output.txt
    try:
        with open(PREDICTION_PATH, "r", encoding="utf-8") as pred_file:
            predicted_answers = [line.strip() for line in pred_file if line.strip()]
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file {PREDICTION_PATH}")
        return
    except UnicodeDecodeError:
        print(f"❌ Lỗi: File {PREDICTION_PATH} có vấn đề về mã hóa")
        return
    except Exception as e:
        print(f"❌ Lỗi khi đọc {PREDICTION_PATH}: {e}")
        return

    # Kiểm tra số lượng câu trả lời
    print(f"\nSố câu trả lời tham chiếu: {len(reference_answers)}")
    print(f"Số câu trả lời dự đoán: {len(predicted_answers)}")
    if len(reference_answers) != len(predicted_answers):
        print(f"\n⚠️ Số lượng câu trả lời không khớp: Tham chiếu = {len(reference_answers)}, Dự đoán = {len(predicted_answers)}")
        return

    # So sánh từng cặp câu trả lời và hiển thị bảng
    print("\n--- So sánh từng câu trả lời ---")
    print(f"{'STT':<5} {'Tham chiếu':<50} {'Dự đoán':<50} {'Kết quả'}")
    print("-" * 120)

    correct = 0
    total = len(reference_answers)
    for idx, (ref, pred) in enumerate(zip(reference_answers, predicted_answers)):
        ref_clean = clean_text(ref)
        pred_clean = clean_text(pred)
        # Logic khớp lệnh được sửa đổi để xử lý các khớp lệnh một phần như đã chỉ định
        match = ref_clean == pred_clean or ref_clean in pred_clean or pred_clean in ref_clean
        result = "✅ Đúng" if match else "❌ Sai"
        if match:
            correct += 1
        # Cắt bớt văn bản hiển thị nếu quá dài
        ref_display = ref if len(ref) <= 47 else ref[:44] + "..."
        pred_display = pred if len(pred) <= 47 else pred[:44] + "..."
        print(f"{idx + 1:<5} {ref_display:<50} {pred_display:<50} {result}")
        if not match:
            print(f"    Không khớp: Tham chiếu = '{ref}', Dự đoán = '{pred}'")

    # Tính và in độ chính xác
    accuracy = (correct / total) * 100
    print(f"\n🎯 Độ chính xác: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    evaluate_model()