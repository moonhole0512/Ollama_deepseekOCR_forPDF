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
        
        # Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0) # Header
        self.grid_rowconfigure(1, weight=1) # Drop Zone
        self.grid_rowconfigure(2, weight=0) # Status / Progress
        
        # Header / Connection Status
        self.header_frame = ctk.CTkFrame(self)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        self.conn_label = ctk.CTkLabel(self.header_frame, text="Checking Ollama...", text_color="gray")
        self.conn_label.pack(side="left", padx=10, pady=5)
        
        self.retry_btn = ctk.CTkButton(self.header_frame, text="Retry Connection", command=self.check_ollama, width=100)
        # Hidden by default
        
        # Drop Zone
        self.drop_frame = ctk.CTkFrame(self)
        self.drop_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        
        self.drop_label = ctk.CTkLabel(self.drop_frame, text="Drag & Drop PDF Here", font=("Arial", 20))
        self.drop_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Enable Drag & Drop
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.drop_file)
        
        # Progress Section
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0)
        
        self.status_text = ctk.CTkLabel(self.status_frame, text="Ready")
        self.status_text.pack(padx=10, pady=5)
        
        # Initial Check
        self.after(500, self.check_ollama)

    def check_ollama(self):
        self.conn_label.configure(text="Checking...", text_color="gray")
        self.retry_btn.pack_forget()
        
        # Run in thread to not freeze UI? Actually check is fast, but better safe.
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
        if self.is_processing:
            return
            
        file_path = event.data
        # TkinterDnD returns paths with {} if they have spaces?
        # Sometimes it returns lines?
        # Let's clean it up.
        file_path = file_path.strip('{}')
        
        if not os.path.isfile(file_path) or not file_path.lower().endswith(".pdf"):
            self.status_text.configure(text="Invalid file. Please drop a PDF.")
            return

        self.start_processing(file_path)

    def start_processing(self, file_path):
        self.is_processing = True
        self.status_text.configure(text=f"Starting processing: {os.path.basename(file_path)}")
        self.progress_bar.set(0)
        
        # Thread
        threading.Thread(target=self._process_thread, args=(file_path,), daemon=True).start()

    def _process_thread(self, file_path):
        def progress(current, total, msg):
            # Update UI from main thread
            val = current / total if total > 0 else 0
            self.after(0, lambda: self.progress_bar.set(val))
            self.after(0, lambda: self.status_text.configure(text=msg))
            
        success, res = self.processor.process_pdf(file_path, progress_callback=progress)
        
        self.after(0, lambda: self._finish_processing(success, res))
        
    def _finish_processing(self, success, res):
        self.is_processing = False
        if success:
            self.status_text.configure(text=f"Done! Saved to: {os.path.basename(res)}")
            self.progress_bar.set(1)
        else:
            self.status_text.configure(text=f"Failed: {res}")
            self.progress_bar.set(0)

if __name__ == "__main__":
    app = App()
    app.mainloop()
