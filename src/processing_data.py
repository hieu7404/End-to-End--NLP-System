def remove_extra_empty_lines(input_file, output_file):
    try:
        # Đọc nội dung file
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Loại bỏ các dòng trống thừa, chỉ giữ lại tối đa một dòng trống giữa các nội dung
        cleaned_lines = []
        previous_line_empty = False

        for line in lines:
            # Kiểm tra xem dòng hiện tại có rỗng hay không
            is_empty = line.strip() == ''

            if is_empty:
                # Nếu dòng rỗng và dòng trước cũng rỗng, bỏ qua
                if previous_line_empty:
                    continue
                previous_line_empty = True
            else:
                previous_line_empty = False

            cleaned_lines.append(line)

        # Ghi nội dung đã xử lý vào file mới
        with open(output_file, 'w', encoding='utf-8') as file:
            file.writelines(cleaned_lines)

        print(f"Đã xử lý xong! Kết quả được lưu tại: {output_file}")

    except FileNotFoundError:
        print(f"Không tìm thấy file: {input_file}")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

# Sử dụng hàm
input_file = './data/all_data.txt'  # File đầu vào
output_file = 'data/data_clean.txt'  # File đầu ra
remove_extra_empty_lines(input_file, output_file)