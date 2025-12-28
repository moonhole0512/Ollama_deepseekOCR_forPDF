import os
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import threading
import math
import textwrap
import io
from PIL import Image

# Font Registration attempt
# Common paths for Korean fonts on Windows
FONT_PATHS = [
    r"C:\Windows\Fonts\NanumGothic.ttf",
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\arial.ttf" # Fallback
]

FONT_NAME = "StandardFont"

def register_fonts():
    global FONT_NAME
    for path in FONT_PATHS:
        if os.path.exists(path):
            try:
                # Use filename as font name
                font_name = "KoreanFont"
                pdfmetrics.registerFont(TTFont(font_name, path))
                FONT_NAME = font_name
                print(f"Registered font: {path}")
                return
            except Exception as e:
                print(f"Failed to register font {path}: {e}")
    print("Warning: No preferred font found. Text might not render correctly if it contains special chars.")

class PDFProcessor:
    def __init__(self, ollama_handler):
        self.ollama_handler = ollama_handler
        register_fonts()

    def process_pdf(self, input_path, progress_callback=None):
        """
        Converts PDF to Images, runs OCR, and rebuilds the PDF.
        input_path: str
        progress_callback: func(current, total, message)
        """
        try:
            if progress_callback:
                progress_callback(0, 0, "Opening PDF...")
            
            # Open PDF with PyMuPDF
            doc = fitz.open(input_path)
            total_pages = len(doc)
            
            output_path = os.path.splitext(input_path)[0] + "_ocr.pdf"
            
            # Create ReportLab Canvas
            c = canvas.Canvas(output_path)
            
            for page_num, page in enumerate(doc):
                display_num = page_num + 1
                if progress_callback:
                    progress_callback(display_num, total_pages, f"Processing page {display_num}/{total_pages}...")
                
                # 1. Capture Page Image (Prefer Native Extraction for Scanned PDFs)
                img = None
                img_list = page.get_images(full=True)
                
                # If it's a scanned PDF, the largest image is usually the page background
                if img_list:
                    # Find the largest image by area
                    best_img = None
                    max_area = 0
                    for img_info in img_list:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        # Width * Height
                        temp_img = Image.open(io.BytesIO(base_image["image"]))
                        area = temp_img.width * temp_img.height
                        if area > max_area:
                            max_area = area
                            best_img = temp_img
                    
                    # Only use extracted image if it's large enough to be a page (e.g. > 1000px height)
                    if best_img and best_img.height > 1000:
                        img = best_img
                        print(f"[PDF] Extracted native image: {img.width}x{img.height}")
                
                if img is None:
                    # Fallback: Render at high-fidelity DPI (300)
                    zoom = 300 / 72 
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                img_width, img_height = img.size
                
                # Create a new PDF page with the EXACT same dimensions as the image (1px = 1pt)
                # to prevent ReportLab from resampling/antialiasing during drawing
                c.setPageSize((img_width, img_height))
                
                # Draw the image on the PDF (Background)
                from reportlab.lib.utils import ImageReader
                # Use mask=None to avoid transparency overhead and preserve pixels
                c.drawImage(ImageReader(img), 0, 0, width=img_width, height=img_height, mask=None)
                
                # 2. Perform OCR on the high-res image
                max_retries = 2 # Grounding attempt + Fallback attempt
                ocr_data = []
                is_fallback = False
                ocr_text_raw = ""
                
                try:
                    # Attempt 1: Grounding OCR (60s timeout)
                    ocr_text_raw = self.ollama_handler.perform_ocr(img, timeout=60)
                    ocr_data = self.ollama_handler.parse_response(ocr_text_raw)
                except Exception as e:
                    # Timeout or error occurred with Grounding OCR
                    is_fallback = True
                    fallback_msg = f"Timeout/Error on page {display_num}. Switching to Free OCR fallback..."
                    if progress_callback:
                        progress_callback(display_num, total_pages, fallback_msg)
                    print(fallback_msg)
                    
                    # Wait a bit for Ollama to recover
                    import time
                    time.sleep(2)
                    
                    try:
                        # Attempt 2: Free OCR Fallback (60s timeout)
                        # Optimize image for fallback (faster processing)
                        max_dim = 1600
                        if img.width > max_dim or img.height > max_dim:
                            scale = max_dim / max(img.width, img.height)
                            fb_img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
                            print(f"Resized fallback image to {fb_img.width}x{fb_img.height}")
                        else:
                            fb_img = img

                        # The user requested "Free OCR." prompt
                        ocr_text_raw = self.ollama_handler.perform_ocr(fb_img, prompt="Free OCR.", timeout=60)
                        # parse_response will likely return empty list if tags are missing
                        ocr_data = self.ollama_handler.parse_response(ocr_text_raw)
                    except Exception as e_inner:
                        error_msg = f"Permanent failure on page {display_num}: {e_inner}"
                        print(error_msg)
                        # The user says skipping is not an option ("안되 다음페이지로 넘어가면 내 목적과 달라져")
                        # So we might have to raise or try one more time? 
                        # But 60s is requested. If it keeps failing, it might be a model/server issue.
                        # For now, let's raise to let the user see the persistent error.
                        raise e_inner

                # 3. Draw Debug Text & Boxes
                # If ocr_data is empty but we have raw text, it might be Free OCR output without tags
                if not ocr_data and ocr_text_raw.strip():
                    # Fallback rendering: treat raw text as plain content
                    # We'll put it at the top of the page in a block
                    lines = ocr_text_raw.strip().split('\n')
                    # Convert to a format similar to ocr_data but without meaningful bboxes
                    # or handle it separately. Let's handle it separately for clarity.
                    pass
                
                text_obj = c.beginText()
                # Debug: Visible text (0) instead of Invisible (3)
                text_obj.setTextRenderMode(0) 
                
                # Debug: Red Text
                c.setFillColor("red")
                text_obj.setFillColor("red")
                
                if not ocr_data and ocr_text_raw.strip():
                    # --- Special handling for coordinate-less text ---
                    # We render it at the top-left, wrapping it to fit the page width
                    font_size = 10
                    text_obj.setFont(FONT_NAME, font_size)
                    from reportlab.lib.utils import simpleSplit
                    
                    current_y = img_height - 20 # 20px margin from top
                    for block in ocr_text_raw.strip().split('\n\n'):
                        if not block.strip(): continue
                        wrapped_lines = simpleSplit(block.strip(), FONT_NAME, font_size, img_width - 40)
                        for line in wrapped_lines:
                            if current_y < 20: break # Page bottom
                            text_obj.setTextOrigin(20, current_y)
                            text_obj.textLine(line)
                            current_y -= font_size * 1.2
                        current_y -= font_size * 0.5 # Extra space between blocks
                else:
                    for item in ocr_data:
                        text = item['text']
                        if not text:
                            continue
                            
                        xmin, ymin, xmax, ymax = item['bbox']
                        
                        # bbox is normalized (0-1)
                        x_pos = xmin * img_width
                        y_pos = (1 - ymax) * img_height # Bottom of the box
                        
                        box_width = (xmax - xmin) * img_width
                        box_height = (ymax - ymin) * img_height
                        
                        # Debug: Draw Blue Rectangle
                        c.setStrokeColor("blue")
                        c.rect(x_pos, y_pos, box_width, box_height, fill=0)
                    
                        # --- New Smart Text Sizing & Wrapping ---
                        # Using reportlab's simpleSplit for accurate width calculation
                        from reportlab.lib.utils import simpleSplit 
                        
                        if len(text) == 0: continue
                        
                        if len(text) == 0: continue
                        
                        # --- Search for optimal font size ---
                        
                        # Heuristic start
                        target_char_area = (box_width * box_height) / (len(text) * 0.8)
                        start_font_size = math.sqrt(target_char_area)
                        start_font_size = min(max(start_font_size, 5), box_height)
                        
                        font_size = start_font_size
                        final_lines = []
                        
                        # Iterative shrinking
                        # Try at most 5 times to prevent infinite loops
                        for _ in range(5):
                            lines = simpleSplit(text, FONT_NAME, font_size, box_width)
                            
                            # Calculate total height needed
                            line_spacing = font_size * 1.2
                            total_height = len(lines) * line_spacing
                            
                            # Does it fit height? 
                            # We accept if it fits or if font size gets too small
                            if total_height <= box_height + (font_size * 0.2): # Allow slight overflow (20%)
                                 final_lines = lines
                                 break
                            
                            # Too tall, shrink font
                            font_size *= 0.85
                            if font_size < 4: # Minimum readable size
                                font_size = 4
                                final_lines = simpleSplit(text, FONT_NAME, font_size, box_width)
                                break
                                
                        if not final_lines:
                             # Fallback if loop logic fails weirdly
                             final_lines = simpleSplit(text, FONT_NAME, font_size, box_width)

                        # 3. Render
                        text_obj.setFont(FONT_NAME, font_size)
                        
                        # We start rendering from top-most line
                        # In PDF coords: top y = y_pos + box_height
                        # First line baseline = top y - font_size (approx ascender)
                        
                        current_y = y_pos + box_height - font_size
                        
                        for line in final_lines:
                            # Hard stop if we really go way below
                            if current_y < y_pos - (font_size * 0.5): 
                                break
                                
                            text_obj.setTextOrigin(x_pos, current_y)
                            text_obj.textLine(line)
                            current_y -= font_size * 1.2
                        
                        # ----------------------------------------
                
                c.drawText(text_obj)
                c.showPage()
            
            c.save()
            doc.close()
            return True, output_path
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)
