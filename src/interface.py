import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import threading
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.io.wavfile import write, read
from gender_classifier import predict
from extract_features import extract_mfcc

class AudioClassifierGUI:
    def __init__(self):
        self.current_file = None
        self.recorded_file = "recordings/temp_recording.wav"
        self.is_recording = False
        self.setup_directories()
        self.setup_gui()
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs("recordings", exist_ok=True)
    
    def setup_gui(self):
        """Initialize the GUI components"""
        self.window = tk.Tk()
        self.window.title("Speech Gender & Digit Classifier")
        self.window.geometry("700x600")
        self.window.configure(bg="#1a1a2e")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom button styles
        style.configure('Custom.TButton',
                       background='#16213e',
                       foreground='#e94560',
                       borderwidth=2,
                       focuscolor='none',
                       relief='flat')
        
        style.map('Custom.TButton',
                 background=[('active', '#0f3460'),
                           ('pressed', '#0a2c54')])
        
        style.configure('Accent.TButton',
                       background='#e94560',
                       foreground='white',
                       borderwidth=2,
                       focuscolor='none',
                       relief='flat')
        
        style.map('Accent.TButton',
                 background=[('active', '#c73650'),
                           ('pressed', '#a52a40')])
        
        # Configure progressbar
        style.configure('Custom.Horizontal.TProgressbar',
                       background='#e94560',
                       troughcolor='#16213e',
                       borderwidth=0,
                       lightcolor='#e94560',
                       darkcolor='#e94560')
        
        self.create_widgets()
        self.setup_layout()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main title
        self.title_label = tk.Label(
            self.window, 
            text="Speech Gender & Digit Classifier",
            font=("Arial", 16, "bold"),
            bg="#1a1a2e",
            fg="#f5f5f5"
        )
        
        # Button frame
        self.btn_frame = tk.Frame(self.window, bg="#1a1a2e")
        
        # Buttons with improved styling
        self.choose_btn = ttk.Button(
            self.btn_frame, 
            text="Choose File", 
            command=self.choose_file,
            width=15,
            style='Custom.TButton'
        )
        
        self.record_btn = ttk.Button(
            self.btn_frame, 
            text="Record Audio", 
            command=self.toggle_recording,
            width=15,
            style='Custom.TButton'
        )
        
        self.test_btn = ttk.Button(
            self.btn_frame, 
            text="Test Audio", 
            command=self.test_audio,
            width=15,
            style='Custom.TButton'
        )
        
        self.clear_btn = ttk.Button(
            self.btn_frame, 
            text="Clear All", 
            command=self.clear_all,
            width=15,
            style='Custom.TButton'
        )
        
        # Status and result labels
        self.status_label = tk.Label(
            self.window, 
            text="No file selected", 
            fg="#0ff0fc",
            font=("Arial", 10),
            bg="#1a1a2e"
        )
        
        self.result_label = tk.Label(
            self.window, 
            text="Prediction → ", 
            font=("Arial", 12, "bold"), 
            fg="#f39c12",
            bg="#1a1a2e"
        )
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.window, 
            mode='indeterminate',
            length=300,
            style='Custom.Horizontal.TProgressbar'
        )
        
        # Frames for plots with borders
        self.waveform_frame = tk.LabelFrame(
            self.window, 
            text="Waveform", 
            height=180, 
            width=600, 
            bg="#16213e",
            fg="#f5f5f5",
            font=("Arial", 10, "bold")
        )
        self.waveform_frame.pack_propagate(False)
        
        self.mfcc_frame = tk.LabelFrame(
            self.window, 
            text="MFCC Features", 
            height=180, 
            width=600, 
            bg="#16213e",
            fg="#f5f5f5",
            font=("Arial", 10, "bold")
        )
        self.mfcc_frame.pack_propagate(False)
        
        # Recording time label
        self.time_label = tk.Label(
            self.window,
            text="",
            font=("Arial", 10, "bold"),
            fg="#e94560",
            bg="#1a1a2e"
        )
    
    def setup_layout(self):
        """Arrange widgets in the window"""
        self.title_label.pack(pady=10)
        
        self.btn_frame.pack(pady=10)
        self.choose_btn.grid(row=0, column=0, padx=5)
        self.record_btn.grid(row=0, column=1, padx=5)
        self.test_btn.grid(row=0, column=2, padx=5)
        self.clear_btn.grid(row=0, column=3, padx=5)
        
        self.status_label.pack(pady=5)
        self.time_label.pack()
        self.result_label.pack(pady=5)
        self.progress.pack(pady=5)
        
        self.waveform_frame.pack(pady=10, padx=20, fill=tk.BOTH)
        self.mfcc_frame.pack(pady=10, padx=20, fill=tk.BOTH)
    
    def choose_file(self):
        """Handle file selection"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select an audio file",
                filetypes=(
                    ("WAV files", "*.wav"),
                    ("MP3 files", "*.mp3"),
                    ("All audio files", "*.wav;*.mp3"),
                    ("All files", "*.*")
                )
            )
            
            if file_path:
                if not self.validate_audio_file(file_path):
                    messagebox.showerror("Invalid File", "Please select a valid audio file.")
                    return
                
                self.current_file = file_path
                filename = os.path.basename(file_path)
                self.status_label.config(text=f"Selected: {filename}")
                self.result_label.config(text="Prediction → ")
                self.plot_waveform_threaded(file_path)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting file: {str(e)}")
    
    def validate_audio_file(self, file_path):
        """Validate if the file is a proper audio file"""
        try:
            sr, audio = read(file_path)
            return len(audio) > 0
        except:
            return False
    
    def toggle_recording(self):
        """Toggle recording state"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start audio recording in a separate thread"""
        self.is_recording = True
        self.record_btn.config(text="Stop Recording", style="Accent.TButton")
        self.status_label.config(text="Recording... Speak now!")
        self.result_label.config(text="Prediction → ")
        self.progress.start()
        
        # Start recording in separate thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
        
        # Start countdown timer
        self.start_countdown(5)  # 5 seconds recording
    
    def start_countdown(self, seconds):
        """Display countdown during recording"""
        if seconds > 0 and self.is_recording:
            self.time_label.config(text=f"Recording: {seconds}s remaining")
            self.window.after(1000, lambda: self.start_countdown(seconds - 1))
        elif self.is_recording:
            self.time_label.config(text="Processing...")
    
    def _record_audio(self):
        """Internal recording function"""
        try:
            duration = 5  # seconds
            sr = 16000    # sample rate
            
            audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='int16')
            sd.wait()
            
            if self.is_recording:  # Check if recording wasn't stopped
                write(self.recorded_file, sr, audio)
                self.current_file = self.recorded_file
                
                # Update UI in main thread
                self.window.after(0, self._recording_complete)
                
        except Exception as e:
            self.window.after(0, lambda: messagebox.showerror("Recording Error", f"Error during recording: {str(e)}"))
            self.window.after(0, self._reset_recording_ui)
    
    def _recording_complete(self):
        """Handle recording completion"""
        self.progress.stop()
        self.is_recording = False
        self.record_btn.config(text="Record Audio", style="Custom.TButton")
        self.time_label.config(text="")
        self.status_label.config(text=f"Recorded: {os.path.basename(self.recorded_file)}")
        self.plot_waveform_threaded(self.recorded_file)
    
    def stop_recording(self):
        """Stop ongoing recording"""
        self.is_recording = False
        self._reset_recording_ui()
    
    def _reset_recording_ui(self):
        """Reset recording UI elements"""
        self.progress.stop()
        self.record_btn.config(text="Record Audio", style="Custom.TButton")
        self.time_label.config(text="")
        if self.status_label.cget("text").startswith("Recording"):
            self.status_label.config(text="Recording stopped")
    
    def test_audio(self):
        """Test the current audio file"""
        if not self.current_file:
            messagebox.showwarning("No file", "Please select or record an audio file first.")
            return
        
        if not os.path.exists(self.current_file):
            messagebox.showerror("File Error", "Selected file does not exist.")
            return
        
        # Start processing in separate thread
        self.progress.start()
        self.status_label.config(text="Processing audio...")
        
        processing_thread = threading.Thread(target=self._test_audio)
        processing_thread.start()
    
    def _test_audio(self):
        """Internal audio testing function"""
        try:
            gender, digit = predict(self.current_file)
            
            # Update UI in main thread
            self.window.after(0, lambda: self._test_complete(gender, digit))
            
        except Exception as e:
            self.window.after(0, lambda: messagebox.showerror("Processing Error", f"Error processing file: {str(e)}"))
            self.window.after(0, self._reset_processing_ui)
    
    def _test_complete(self, gender, digit):
        """Handle test completion"""
        self.progress.stop()
        
        if gender is None or digit is None:
            messagebox.showerror("Error", "Could not process the file. Please try a different audio file.")
            self.result_label.config(text="Prediction → Error")
            return
        
        self.result_label.config(text=f"Prediction → Gender: {gender}, Digit: {digit}")
        self.status_label.config(text="Processing complete")
        self.plot_mfcc_threaded(self.current_file)
    
    def _reset_processing_ui(self):
        """Reset processing UI elements"""
        self.progress.stop()
        self.status_label.config(text="Processing failed")
    
    def plot_waveform_threaded(self, file_path):
        """Plot waveform in separate thread"""
        threading.Thread(target=lambda: self._plot_waveform(file_path)).start()
    
    def _plot_waveform(self, file_path):
        """Internal waveform plotting function"""
        try:
            sr, audio = read(file_path)
            audio = audio.squeeze()
            
            # Create plot in main thread
            self.window.after(0, lambda: self._display_waveform(audio))
            
        except Exception as e:
            print(f"Error plotting waveform: {e}")
    
    def _display_waveform(self, audio):
        """Display waveform plot"""
        # Clear previous plot
        for widget in self.waveform_frame.winfo_children():
            widget.destroy()
        
        # Set dark theme
        plt.style.use('dark_background')
        
        # Create new plot
        fig, ax = plt.subplots(figsize=(6, 2))
        fig.patch.set_facecolor('#16213e')
        ax.set_facecolor('#16213e')
        
        time_axis = np.linspace(0, len(audio) / 16000, len(audio))  # Assume 16kHz
        ax.plot(time_axis, audio, color='#0ff0fc', linewidth=0.8)
        ax.set_title("Audio Waveform", fontsize=10, fontweight='bold', color='#f5f5f5')
        ax.set_xlabel("Time (seconds)", color='#f5f5f5')
        ax.set_ylabel("Amplitude", color='#f5f5f5')
        ax.grid(True, alpha=0.3, color='#e94560')
        ax.tick_params(colors='#f5f5f5')
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.waveform_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)
    
    def plot_mfcc_threaded(self, file_path):
        """Plot MFCC in separate thread"""
        threading.Thread(target=lambda: self._plot_mfcc(file_path)).start()
    
    def _plot_mfcc(self, file_path):
        """Internal MFCC plotting function"""
        try:
            mfcc = extract_mfcc(file_path)
            if mfcc is not None:
                self.window.after(0, lambda: self._display_mfcc(mfcc))
        except Exception as e:
            print(f"Error plotting MFCC: {e}")
    
    def _display_mfcc(self, mfcc):
        """Display MFCC plot"""
        # Clear previous plot
        for widget in self.mfcc_frame.winfo_children():
            widget.destroy()
        
        # Set dark theme
        plt.style.use('dark_background')
        
        # Create new plot
        fig, ax = plt.subplots(figsize=(6, 2))
        fig.patch.set_facecolor('#16213e')
        ax.set_facecolor('#16213e')
        
        img = ax.imshow(mfcc, aspect="auto", origin="lower", cmap='plasma')
        ax.set_title("MFCC Features", fontsize=10, fontweight='bold', color='#f5f5f5')
        ax.set_xlabel("Time Frames", color='#f5f5f5')
        ax.set_ylabel("MFCC Coefficients", color='#f5f5f5')
        ax.tick_params(colors='#f5f5f5')
        
        cbar = fig.colorbar(img, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_tick_params(color='#f5f5f5')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#f5f5f5')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.mfcc_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)
    
    def clear_all(self):
        """Clear all data and reset UI"""
        self.current_file = None
        self.status_label.config(text="No file selected")
        self.result_label.config(text="Prediction → ")
        self.time_label.config(text="")
        
        # Clear plots
        for widget in self.waveform_frame.winfo_children():
            widget.destroy()
        for widget in self.mfcc_frame.winfo_children():
            widget.destroy()
        
        # Reset progress bar
        self.progress.stop()
        
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
    
    def run(self):
        """Start the application"""
        try:
            self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.window.mainloop()
        except Exception as e:
            messagebox.showerror("Application Error", f"An error occurred: {str(e)}")
    
    def on_closing(self):
        """Handle application closing"""
        if self.is_recording:
            self.stop_recording()
        self.window.destroy()

# Run the application
if __name__ == "__main__":
    app = AudioClassifierGUI()
    app.run()