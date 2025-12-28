import re
import requests
import base64
import io
from PIL import Image

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

    def perform_ocr(self, image: Image.Image, prompt="<|grounding|>Extract text with bounding boxes.", timeout=120):
        """
        Sends the image to Ollama for OCR using the CLI.
        The CLI (ollama run) is 100% reliable for DeepSeek-OCR.
        Maintains native resolution (no resizing) to match user manual test.
        """
        import subprocess
        import os
        import tempfile

        # 1. Color Mode & Quality Protection
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # 2. Save image to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_img_path = os.path.join(temp_dir, f"ocr_input_{os.getpid()}.jpg")
        try:
            # Use quality=100 to avoid any compression artifacts
            image.save(temp_img_path, format="JPEG", quality=100, subsampling=0)
            # Also save to current dir for debug
            image.save("ocr_debug_input.jpg", format="JPEG", quality=100, subsampling=0)
        except Exception as e:
            print(f"Failed to save temporary image: {e}")
            raise e

        # 2. Prepare the CLI command
        # Using shell=True and a single string mimics a manual terminal run,
        # which avoids the ~1250 character truncation issue.
        # Ensure path and prompt are correctly formatted for the shell.
        safe_path = temp_img_path.replace("\\", "\\\\")
        cmd_str = f'ollama run {self.model} "{safe_path}\\n{prompt}"'
        
        print(f"[OCR Request CLI] Model: {self.model} | Image: {image.width}x{image.height} (Shell-wrapped)")
        
        try:
            # We use shell=True to replicate the user's manual success
            res = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout
            )
            
            if res.returncode == 0:
                # The CLI output might include "Added image..." at the top
                # but parse_response regex will handle it.
                response_text = res.stdout.strip()
                
                # Debug logging
                try:
                    with open("ocr_debug_log.txt", "w", encoding="utf-8") as f:
                        f.write(response_text)
                except:
                    pass
                    
                return response_text
            else:
                error_msg = f"Ollama CLI error (code {res.returncode}): {res.stderr}"
                print(error_msg)
                raise Exception(error_msg)
                
        except subprocess.TimeoutExpired:
            raise Exception(f"OCR Timed out after {timeout}s")
        except Exception as e:
            print(f"Exception during OCR CLI: {e}")
            raise e
        finally:
            if os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                except:
                    pass

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
