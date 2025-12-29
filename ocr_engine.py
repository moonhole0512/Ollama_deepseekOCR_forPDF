import re
import requests
import base64
import io
import json
import time
from PIL import Image

class HallucinationError(Exception):
    """Raised when the model output is detected as a repetitive hallucination loop."""
    pass

class OllamaHandler:
    def __init__(self, base_url="http://localhost:11434", model="deepseek-ocr"):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()

    def check_connection(self):
        """
        Checks if Ollama is running and if a model containing 'deepseek-ocr' exists.
        Returns:
            tuple: (bool, str) -> (Success, Message or Model Name)
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=2)
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

    def perform_ocr(self, image: Image.Image, prompt="<|grounding|>Extract text with bounding boxes.", timeout=120, page_num=None):
        """
        Sends the image to Ollama for OCR using the /api/generate endpoint with streaming.
        Detects hallucinatory repetitive patterns early to avoid long waits.
        """
        # 1. Color Mode Protection
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # 2. Encode image to Base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=100, subsampling=0)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        try:
            image.save("ocr_debug_input.jpg", format="JPEG", quality=100, subsampling=0)
        except:
            pass

        page_ctx = f" [Page {page_num}]" if page_num is not None else ""
        print(f"[OCR Request API]{page_ctx} Model: {self.model} | Format: {image.width}x{image.height} | Timeout: {timeout}s (Streaming)")
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": True,
            "options": {
                "num_ctx": 8192,
                "num_predict": 4096,
                "temperature": 0
            }
        }
        
        full_response = []
        hallucination_detected = False
        
        try:
            start_time = time.time()
            # Note: total timeout for the stream
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=(5, 120) # Connect timeout, Time between chunks
            )
            
            if response.status_code != 200:
                error_msg = f"Ollama API error (code {response.status_code}): {response.text}"
                raise Exception(error_msg)

            try:
                # --- Hallucination Detection Buffer ---
                # We look for long repeated patterns in the last N tokens
                recent_buffer = ""
                
                for line in response.iter_lines():
                    if time.time() - start_time > timeout:
                        raise requests.exceptions.Timeout()
                        
                    if not line:
                        continue
                        
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get('response', '')
                    full_response.append(token)
                    recent_buffer += token
                    
                    # Keep buffer manageable but long enough for pattern detection
                    if len(recent_buffer) > 200:
                        recent_buffer = recent_buffer[-200:]
                    
                    # Detection 1: Identical short pattern repeat (e.g. "1.1.1.1." or "abcabcabc")
                    if len(recent_buffer) > 50:
                        for p_len in range(1, 11):
                            segment = recent_buffer[-p_len:]
                            # If a character/sequence repeats more than 15 times consecutively
                            if recent_buffer.endswith(segment * 15):
                                hallucination_detected = True
                                print(f"{page_ctx} !!! Early Hallucination Detected (pattern: {repr(segment)}) !!!")
                                break
                    
                    if hallucination_detected:
                        break
                    
                    if chunk.get('done'):
                        break
                
                if hallucination_detected:
                    raise HallucinationError("Repetitive pattern detected in model output.")
            finally:
                response.close() # CRITICAL: Ensure server knows we stopped reading
                
            response_text = "".join(full_response).strip()
            
            # Final debug log
            try:
                with open("ocr_debug_log.txt", "w", encoding="utf-8") as f:
                    f.write(response_text)
            except:
                pass
                
            return response_text
                
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout):
            raise Exception(f"OCR Timed out after {timeout}s")
        except HallucinationError:
            raise
        except Exception as e:
            print(f"{page_ctx} Exception during OCR API: {e}")
            raise e

    def parse_response(self, text):
        """
        Parses the DeepSeek-OCR output format:
        <|ref|>text<|/ref|><|det|>[[ymin, xmin, ymax, xmax]]<|/det|>\n(content)
        """
        if isinstance(text, list): return text
        results = []
        
        # Robust pattern to handle various whitespace/tag variations
        pattern = re.compile(r'<\|ref\|>\s*(?P<type>.*?)\s*<\|/ref\|>\s*<\|det\|>\s*\[\[(?P<coords>.*?)\]\]\s*<\|/det\|>')
        
        matches = list(pattern.finditer(text))
        
        for i, match in enumerate(matches):
            try:
                coords_str = match.group('coords')
                coords = [float(x.strip()) for x in coords_str.split(',')]
                if len(coords) != 4: continue
                
                # Parse Content
                start_index = match.end()
                if i + 1 < len(matches):
                    end_index = matches[i+1].start()
                else:
                    end_index = len(text)
                    
                content = text[start_index:end_index].strip()
                
                # DeepSeek-OCR (this version) uses [xmin, ymin, xmax, ymax]
                # as verified by horizontal text spanning (1st/3rd numbers are X).
                xmin, ymin, xmax, ymax = coords
                
                results.append({
                    'bbox': [xmin / 1000.0, ymin / 1000.0, xmax / 1000.0, ymax / 1000.0],
                    'text': content
                })
            except:
                continue
            
        return results
