import os
import pymupdf  # PyMuPDF

def parse_article(pdf_path, output_dir="extracted_images"):
    os.makedirs(output_dir, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    
    full_text = []
    images_info = []

    for page_num, page in enumerate(doc, start=1):
        # 1. Извлекаем текст с сохранением естественного порядка чтения
        text = page.get_text("text", sort=True)
        full_text.append(f"--- Страница {page_num} ---\n{text}")

        # 2. Извлекаем изображения как отдельные объекты
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            image_filename = f"page{page_num}_img{img_index}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
                
            images_info.append({
                "page": page_num,
                "file": image_path,
                "width": base_image["width"],
                "height": base_image["height"]
            })

    return "\n".join(full_text), images_info

text, images = parse_article(r"dev_articles\D001.pdf")
