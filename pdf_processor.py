import os
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import threading
import math
import textwrap
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
                
                # 1. Convert Page to Image
                # Single High Res render (300 DPI) as requested for rollback
                zoom = 300 / 72 
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                if pix.alpha: mode = "RGBA"
                else: mode = "RGB"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                
                img_width, img_height = img.size
                
                # Set page size to match high-res image
                c.setPageSize((img_width, img_height))
                
                # Draw the original image on the PDF (Background)
                from reportlab.lib.utils import ImageReader
                c.drawImage(ImageReader(img), 0, 0, width=img_width, height=img_height)
                
                # 2. Perform OCR on the high-res image
                ocr_text_raw = self.ollama_handler.perform_ocr(img)
                ocr_data = self.ollama_handler.parse_response(ocr_text_raw)
                
                # 3. Draw Debug Text & Boxes
                # Use TextObject for advanced rendering
                text_obj = c.beginText()
                # Debug: Visible text (0) instead of Invisible (3)
                text_obj.setTextRenderMode(0) 
                
                # Debug: Red Text
                c.setFillColor("red")
                text_obj.setFillColor("red")
                
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
