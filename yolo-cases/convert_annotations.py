import os
import xml.etree.ElementTree as ET

def convert(xml_path, txt_path, classes):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图片尺寸
    size = root.find('size')
    if size is None:
        raise ValueError("No size information in XML file")

    w = float(size.find('width').text)
    h = float(size.find('height').text)

    with open(txt_path, 'w', encoding='utf-8') as f:
        for obj in root.findall('object'):
            # 获取类别名称，优先使用 <n> 标签，如果没有则尝试使用 <name> 标签
            name_elem = obj.find('n')
            if name_elem is None:
                name_elem = obj.find('name')
            if name_elem is None or not name_elem.text:
                print(f"Warning: No valid class name found in {xml_path}")
                continue

            name = name_elem.text.strip()
            if name not in classes:
                print(f"Warning: Unknown class '{name}' in {xml_path}")
                continue

            cls_id = classes.index(name)

            # 获取边界框坐标
            box = obj.find('bndbox')
            if box is None:
                print(f"Warning: No bounding box found in {xml_path}")
                continue

            try:
                xmin = float(box.find('xmin').text)
                ymin = float(box.find('ymin').text)
                xmax = float(box.find('xmax').text)
                ymax = float(box.find('ymax').text)
            except (AttributeError, ValueError) as e:
                print(f"Warning: Invalid bounding box values in {xml_path}: {str(e)}")
                continue

            # 转换为YOLO格式
            x = (xmin + xmax) / 2.0 / w
            y = (ymin + ymax) / 2.0 / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h

            # 检查坐标是否有效
            if not all(0 <= val <= 1 for val in [x, y, width, height]):
                print(f"Warning: Invalid normalized coordinates in {xml_path}")
                continue

            # 写入文件
            f.write(f"{cls_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n")

def main():
    classes = ['crazing', 'inclusion', 'pitted_surface', 'scratches', 'patches', 'rolled-in_scale']

    # 训练集转换
    xml_dir = './steel_data/train/ANNOTATIONS'
    txt_dir = './steel_data/train/LABELS'
    os.makedirs(txt_dir, exist_ok=True)

    print("Converting training set annotations...")
    converted_count = 0
    error_count = 0

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        txt_file = xml_file.replace('.xml', '.txt')
        txt_path = os.path.join(txt_dir, txt_file)

        try:
            convert(xml_path, txt_path, classes)
            converted_count += 1
            if converted_count % 100 == 0:
                print(f'Converted {converted_count} files...')
        except Exception as e:
            error_count += 1
            print(f'Error converting {xml_file}: {str(e)}')

    print(f"\nTraining set conversion completed:")
    print(f"Successfully converted: {converted_count} files")
    print(f"Failed to convert: {error_count} files")

    # 测试集转换
    xml_dir = 'steel_data/test/ANNOTATIONS'
    txt_dir = 'steel_data/test/LABELS'
    os.makedirs(txt_dir, exist_ok=True)

    if os.path.exists(xml_dir):
        print("\nConverting test set annotations...")
        converted_count = 0
        error_count = 0

        for xml_file in os.listdir(xml_dir):
            if not xml_file.endswith('.xml'):
                continue

            xml_path = os.path.join(xml_dir, xml_file)
            txt_file = xml_file.replace('.xml', '.txt')
            txt_path = os.path.join(txt_dir, txt_file)

            try:
                convert(xml_path, txt_path, classes)
                converted_count += 1
                if converted_count % 100 == 0:
                    print(f'Converted {converted_count} files...')
            except Exception as e:
                error_count += 1
                print(f'Error converting {xml_file}: {str(e)}')

        print(f"\nTest set conversion completed:")
        print(f"Successfully converted: {converted_count} files")
        print(f"Failed to convert: {error_count} files")

if __name__ == '__main__':
    main()
