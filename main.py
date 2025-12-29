import tkinter as tk
import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_FILES
import threading
import os
from ocr_engine import OllamaHandler
from pdf_processor import PDFProcessor

# Configure Appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)
        
        self.title("DeepSeek-OCR PDF Generator")
        self.geometry("600x450")
        
        # Data
        self.ollama = OllamaHandler()
        self.processor = PDFProcessor(self.ollama)
        self.is_processing = False
        self.queue = []  # List of dicts: {'path': str, 'frame': ctk.CTkFrame, 'pb': ctk.CTkProgressBar, 'lbl': ctk.CTkLabel, 'remove_btn': ctk.CTkButton, 'id': int}
        self.current_item = None
        self.item_counter = 0

        # Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0) # Header
        self.grid_rowconfigure(1, weight=1) # Main Content (Drop + List)
        self.grid_rowconfigure(2, weight=0) # Status Footer
        
        # Header / Connection Status
        self.header_frame = ctk.CTkFrame(self)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        self.conn_label = ctk.CTkLabel(self.header_frame, text="Checking Ollama...", text_color="gray")
        self.conn_label.pack(side="left", padx=10, pady=5)
        
        self.retry_btn = ctk.CTkButton(self.header_frame, text="Retry Connection", command=self.check_ollama, width=100)
        
        # Main Split Content
        self.main_content = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.main_content.grid_columnconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(1, weight=2) # List takes more space

        # Drop Zone
        self.drop_frame = ctk.CTkFrame(self.main_content)
        self.drop_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 5))
        
        self.drop_label = ctk.CTkLabel(self.drop_frame, text="Drag & Drop PDF Here", font=("Arial", 16))
        self.drop_label.place(relx=0.5, rely=0.5, anchor="center")
        
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.drop_file)
        
        # Queue List Section
        self.queue_frame = ctk.CTkScrollableFrame(self.main_content, label_text="Processing Queue")
        self.queue_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=(5, 0))
        
        # Footer
        self.status_footer = ctk.CTkLabel(self, text="Ready", height=20)
        self.status_footer.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 5))
        
        # Initial Check
        self.after(500, self.check_ollama)

    def check_ollama(self):
        self.conn_label.configure(text="Checking...", text_color="gray")
        self.retry_btn.pack_forget()
        threading.Thread(target=self._check_ollama_thread, daemon=True).start()
        
    def _check_ollama_thread(self):
        success, msg = self.ollama.check_connection()
        self.after(0, lambda: self._update_conn_status(success, msg))
        
    def _update_conn_status(self, success, msg):
        if success:
            self.conn_label.configure(text=f"Connected: {msg}", text_color="green")
        else:
            self.conn_label.configure(text=f"Error: {msg}", text_color="red")
            self.retry_btn.pack(side="right", padx=10, pady=5)

    def drop_file(self, event):
        raw_data = event.data
        import re
        files = re.findall(r'\{(.*?)\}|(\S+)', raw_data)
        file_paths = [f[0] if f[0] else f[1] for f in files]
        
        valid_files = [fp for fp in file_paths if os.path.isfile(fp) and fp.lower().endswith(".pdf")]
        
        if not valid_files:
            self.status_footer.configure(text="No valid PDFs found.")
            return

        for fp in valid_files:
            self.add_to_queue_ui(fp)

        if not self.is_processing:
            self.process_next_in_queue()

    def add_to_queue_ui(self, file_path):
        item_id = self.item_counter
        self.item_counter += 1
        
        item_frame = ctk.CTkFrame(self.queue_frame)
        item_frame.pack(fill="x", padx=5, pady=2)
        
        name_lbl = ctk.CTkLabel(item_frame, text=os.path.basename(file_path), width=150, anchor="w")
        name_lbl.pack(side="left", padx=5)
        
        pb = ctk.CTkProgressBar(item_frame)
        pb.pack(side="left", fill="x", expand=True, padx=5)
        pb.set(0)
        
        status_lbl = ctk.CTkLabel(item_frame, text="Waiting", width=120)
        status_lbl.pack(side="left", padx=5)
        
        remove_btn = ctk.CTkButton(item_frame, text="X", width=30, fg_color="red", hover_color="#8B0000",
                                    command=lambda: self.remove_item(item_id))
        remove_btn.pack(side="right", padx=5)
        
        item_data = {
            'id': item_id,
            'path': file_path,
            'frame': item_frame,
            'pb': pb,
            'lbl': status_lbl,
            'btn': remove_btn,
            'is_cancelled': False
        }
        self.queue.append(item_data)

    def remove_item(self, item_id):
        target = None
        for item in self.queue:
            if item['id'] == item_id:
                target = item
                break
        
        if not target:
            if self.current_item and self.current_item['id'] == item_id:
                target = self.current_item
            else:
                return

        # If it's the active one, we mark it cancelled
        if target == self.current_item:
            target['is_cancelled'] = True
            target['lbl'].configure(text="Cancelling...")
            target['btn'].configure(state="disabled")
            # We can't easily kill the thread without force, so we let it finish and skip the 'done' part
        else:
            # If it's in queue, just remove it
            target['frame'].destroy()
            self.queue = [i for i in self.queue if i['id'] != item_id]

    def process_next_in_queue(self):
        if not self.queue:
            self.is_processing = False
            self.status_footer.configure(text="All tasks finished.")
            return
            
        self.current_item = self.queue.pop(0)
        self.is_processing = True
        self.current_item['lbl'].configure(text="Initializing...")
        self.current_item['btn'].configure(text="X", state="normal") # Keep X for active task (cancellation)
        self.status_footer.configure(text=f"Active: {os.path.basename(self.current_item['path'])}")
        
        threading.Thread(target=self._process_thread, args=(self.current_item,), daemon=True).start()

    def _process_thread(self, item):
        def progress(current, total, msg):
            if item['is_cancelled']:
                return # Don't update UI if cancelled
            val = current / total if total > 0 else 0
            self.after(0, lambda: item['pb'].set(val))
            self.after(0, lambda: item['lbl'].configure(text=f"{int(val*100)}% - P{current}/{total}"))
            
        success, res = self.processor.process_pdf(item['path'], progress_callback=progress)
        
        self.after(0, lambda: self._finish_processing(item, success, res))
        
    def _finish_processing(self, item, success, res):
        if item['is_cancelled']:
            item['frame'].destroy()
        else:
            if success:
                item['lbl'].configure(text="Done", text_color="green")
                item['pb'].set(1)
                # Auto-remove finished items after X seconds? 
                # Better to just change button to 'Open' or 'Clear'
                item['btn'].configure(text="OK", command=lambda: item['frame'].destroy(), fg_color="gray")
                print(f"Finished: {res}")
            else:
                item['lbl'].configure(text="Error", text_color="red")
                item['btn'].configure(text="X", command=lambda: item['frame'].destroy())
                print(f"Failed: {res}")
                
        self.current_item = None
        self.process_next_in_queue()

if __name__ == "__main__":
    app = App()
    app.mainloop()
