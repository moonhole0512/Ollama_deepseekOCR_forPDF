import re
import requests
import base64
import io
from PIL import Image

class OllamaHandler:
    def __init__(self, base_url="http://localhost:11434", model="deepseek-ocr"):
        self.base_url = base_url
        self.model = model

    def check_connection(self):
        """
        Checks if Ollama is running and if a model containing 'deepseek-ocr' exists.
        Returns:
            tuple: (bool, str) -> (Success, Message or Model Name)
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                found_model = None
                
                # Check for deepseek-ocr in model names
                for m in models:
                    name = m.get('name', '')
                    if 'deepseek-ocr' in name:
                        found_model = name
                        break
                
                if found_model:
                    self.model = found_model
                    return True, found_model
                else:
                    return False, "Model 'deepseek-ocr' not found. Please pull it first."
            else:
                return False, f"Ollama returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "Could not connect to Ollama. Is it running?"
        except Exception as e:
            return False, str(e)

    def perform_ocr(self, image: Image.Image):
        """
        Sends the image to Ollama for OCR.
        """
        # Debug: Save image for verification
        try:
            image.save("ocr_debug_input.jpg", format="JPEG")
        except:
            pass

        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = "<|grounding|>Extract text with bounding boxes."
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_str],
            "stream": False,
            "options": {
                "num_ctx": 8192, # Increase context window
                "num_predict": -1 # Unlimited generation
            }
        }

        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                print(f"Error: {response.text}")
                return ""
        except Exception as e:
            print(f"Exception during OCR: {e}")
            return ""

    def parse_response(self, text):
        """
        Parses the DeepSeek-OCR output format:
        <|ref|>text<|/ref|><|det|>[[ymin, xmin, ymax, xmax]]<|/det|>\n(content)
        
        Returns:
            list of dict: [{'bbox': [x, y, w, h], 'text': 'content'}]
            Note: bbox is normalized (0.0-1.0) and converted to [x, y, w, h] or similar for easier usage.
            Actually, let's return [xmin, ymin, xmax, ymax] normalized.
        """
        results = []
        
        # Regex to capture the block. 
        # Structure: <|ref|>(type)<|/ref|><|det|>\[\[(coords)\]\]<|/det|>\n(content...)
        # Content might span multiple lines until the next tag or end of string? 
        # The prompt says "Tag block immediately followed by text... until empty line".
        # Let's try splitting by the specific tag pattern.
        
        # Pattern to find the start of a block
        # Updated to be robust against whitespace between tags
        # <|ref|> type <|/ref|> <|det|> [[ coords ]] <|/det|>
        pattern = re.compile(r'<\|ref\|>\s*(?P<type>.*?)\s*<\|/ref\|>\s*<\|det\|>\s*\[\[(?P<coords>.*?)\]\]\s*<\|/det\|>')
        
        # Debug: Log raw response to file for analysis
        try:
            with open("ocr_debug_log.txt", "a", encoding="utf-8") as f:
                f.write("-" * 40 + "\n")
                f.write(text + "\n")
        except:
            pass
        
        # We iterate over matches
        matches = list(pattern.finditer(text))
        
        for i, match in enumerate(matches):
            type_str = match.group('type')
            
            # Filter only text - DISABLED based on user feedback
            # DeepSeek-OCR uses various tags like 'title', 'sub_title', 'text', 'table', 'figure_text'
            # We want to capture text from all of them.
            # if type_str != 'text':
            #    continue
                
            coords_str = match.group('coords')
            try:
                # coords are valid python list string "ymin, xmin, ymax, xmax"
                # e.g. "100, 200, 300, 400"
                coords = [float(x.strip()) for x in coords_str.split(',')]
            except:
                continue
                
            if len(coords) != 4:
                continue
                
            # Parse Content
            # Content starts after the end of this match
            start_index = match.end()
            
            # Content ends at the start of the next match, or end of string
            if i + 1 < len(matches):
                end_index = matches[i+1].start()
            else:
                end_index = len(text)
                
            raw_content = text[start_index:end_index]
            
            # The prompt says: "Tag block immediately followed by text... until empty line"
            # But usually the model outputs: [Tag]\nContent\n
            # Let's strip whitespace and take the content.
            # If there are multiple lines, we preserve them?
            # Usually OCR result is line-based. 
            # Let's strip leading newline if present.
            
            content = raw_content.strip()
            
            # Normalize coordinates (0-1000) -> (0.0-1.0)
            # Input: [xmin, ymin, xmax, ymax] relative to 1000
            xmin, ymin, xmax, ymax = coords
            
            norm_ymin = ymin / 1000.0
            norm_xmin = xmin / 1000.0
            norm_ymax = ymax / 1000.0
            norm_xmax = xmax / 1000.0
            
            results.append({
                'bbox': [norm_xmin, norm_ymin, norm_xmax, norm_ymax], # Standard x1, y1, x2, y2
                'text': content
            })
            
        return results
