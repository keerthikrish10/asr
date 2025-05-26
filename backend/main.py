from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import shutil
import os
import librosa
import soundfile as sf
import numpy as np
import nemo.collections.asr as nemo_asr
import asyncio
import onnxruntime as ort
import torch
import pickle
from pathlib import Path
import logging
from scipy import signal
from scipy.ndimage import median_filter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class TranscriptionResult(BaseModel):
    text: str
    success: bool = True
    message: str = "Transcription completed successfully"
    converted_from_mp3: bool = False
    inference_time: float = 0.0
    preprocessing_applied: str = ""

class AudioPreprocessor:
    """Advanced audio preprocessing for noise reduction"""
    
    def __init__(self):
        self.target_sr = 16000
        
    def apply_bandpass_filter(self, audio, sr, low_freq=80, high_freq=8000):
        """Apply bandpass filter to remove frequencies outside speech range"""
        try:
            nyquist = sr / 2
            low = low_freq / nyquist
            high = min(high_freq / nyquist, 0.99)  # Prevent aliasing
            
            # Design Butterworth bandpass filter
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_audio = signal.filtfilt(b, a, audio)
            
            logger.info(f"Applied bandpass filter: {low_freq}Hz - {high_freq}Hz")
            return filtered_audio
        except Exception as e:
            logger.warning(f"Bandpass filter failed: {e}, using original audio")
            return audio
    
    def remove_dc_offset(self, audio):
        """Remove DC offset from audio signal"""
        return audio - np.mean(audio)
    
    def apply_preemphasis(self, audio, coeff=0.97):
        """Apply pre-emphasis filter to balance frequency spectrum"""
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])
    
    def spectral_subtraction_denoising(self, audio, sr, noise_factor=2.0, alpha=2.0):
        """Advanced spectral subtraction for noise reduction"""
        try:
            # Compute STFT
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first few frames (assuming initial silence/noise)
            noise_frames = min(10, magnitude.shape[1] // 4)
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Apply spectral subtraction
            subtracted_magnitude = magnitude - alpha * noise_spectrum
            
            # Apply spectral floor to prevent over-subtraction
            spectral_floor = 0.1 * magnitude
            enhanced_magnitude = np.maximum(subtracted_magnitude, spectral_floor)
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            logger.info("Applied spectral subtraction denoising")
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Spectral subtraction failed: {e}, using original audio")
            return audio
    
    def wiener_filter_denoising(self, audio, sr):
        """Wiener filter based denoising"""
        try:
            # Compute power spectral density
            f, psd = signal.welch(audio, sr, nperseg=1024)
            
            # Estimate noise PSD (assume it's the minimum across frequency bins)
            noise_psd = np.percentile(psd, 10)  # Use 10th percentile as noise estimate
            
            # Compute Wiener filter
            wiener_filter = psd / (psd + noise_psd)
            
            # Apply filter in frequency domain
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Interpolate Wiener filter to match STFT dimensions
            freq_bins = magnitude.shape[0]
            wiener_interp = np.interp(np.linspace(0, len(wiener_filter)-1, freq_bins), 
                                    np.arange(len(wiener_filter)), wiener_filter)
            wiener_interp = wiener_interp.reshape(-1, 1)
            
            # Apply filter
            filtered_magnitude = magnitude * wiener_interp
            filtered_stft = filtered_magnitude * np.exp(1j * phase)
            filtered_audio = librosa.istft(filtered_stft, hop_length=512)
            
            logger.info("Applied Wiener filter denoising")
            return filtered_audio
            
        except Exception as e:
            logger.warning(f"Wiener filter failed: {e}, using original audio")
            return audio
    
    def adaptive_noise_reduction(self, audio, sr):
        """Adaptive noise reduction based on signal statistics"""
        try:
            # Calculate frame-wise energy
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            frames = librosa.util.frame(audio, frame_length=frame_length, 
                                      hop_length=hop_length, axis=0)
            energy = np.sum(frames**2, axis=0)
            
            # Identify noise frames (lowest 20% energy)
            noise_threshold = np.percentile(energy, 20)
            noise_frames = energy < noise_threshold
            
            # Apply stronger filtering to noise frames
            enhanced_audio = audio.copy()
            
            for i, is_noise in enumerate(noise_frames):
                start_idx = i * hop_length
                end_idx = min(start_idx + frame_length, len(audio))
                
                if is_noise:
                    # Apply stronger smoothing to noise frames
                    segment = audio[start_idx:end_idx]
                    # Simple moving average for noise reduction
                    window_size = min(5, len(segment))
                    if window_size > 1:
                        smoothed = np.convolve(segment, np.ones(window_size)/window_size, mode='same')
                        enhanced_audio[start_idx:end_idx] = smoothed * 0.5  # Reduce amplitude
            
            logger.info("Applied adaptive noise reduction")
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Adaptive noise reduction failed: {e}, using original audio")
            return audio
    
    def normalize_audio(self, audio, target_rms=0.1):
        """Normalize audio to target RMS level"""
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms > 0:
            scaling_factor = target_rms / current_rms
            # Prevent excessive amplification
            scaling_factor = min(scaling_factor, 5.0)
            audio = audio * scaling_factor
        
        # Peak normalization as fallback
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
            
        return audio
    
    def remove_silence(self, audio, sr, top_db=20, frame_length=2048, hop_length=512):
        """Remove silence from beginning and end of audio"""
        try:
            # Trim silence
            audio_trimmed, _ = librosa.effects.trim(
                audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length
            )
            
            # Ensure minimum length
            min_length = int(0.1 * sr)  # 100ms minimum
            if len(audio_trimmed) < min_length:
                return audio  # Return original if too short after trimming
                
            logger.info(f"Trimmed silence: {len(audio)} -> {len(audio_trimmed)} samples")
            return audio_trimmed
            
        except Exception as e:
            logger.warning(f"Silence removal failed: {e}, using original audio")
            return audio
    
    def comprehensive_preprocessing(self, audio, sr, intensity='medium'):
        """Apply comprehensive preprocessing based on intensity level"""
        preprocessing_steps = []
        
        try:
            # Always apply basic preprocessing
            audio = self.remove_dc_offset(audio)
            preprocessing_steps.append("DC offset removal")
            
            # Trim silence
            audio = self.remove_silence(audio, sr)
            preprocessing_steps.append("silence trimming")
            
            if intensity in ['medium', 'high']:
                # Apply bandpass filter
                audio = self.apply_bandpass_filter(audio, sr)
                preprocessing_steps.append("bandpass filter")
                
                # Apply pre-emphasis
                audio = self.apply_preemphasis(audio)
                preprocessing_steps.append("pre-emphasis")
            
            if intensity == 'high':
                # Apply advanced denoising
                audio = self.spectral_subtraction_denoising(audio, sr)
                preprocessing_steps.append("spectral subtraction")
                
                audio = self.wiener_filter_denoising(audio, sr)
                preprocessing_steps.append("Wiener filter")
                
                audio = self.adaptive_noise_reduction(audio, sr)
                preprocessing_steps.append("adaptive noise reduction")
            
            # Always normalize at the end
            audio = self.normalize_audio(audio)
            preprocessing_steps.append("normalization")
            
            logger.info(f"Applied preprocessing steps: {', '.join(preprocessing_steps)}")
            return audio, ', '.join(preprocessing_steps)
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return audio, "preprocessing failed"

class OptimizedASRModel:
    def __init__(self, model_name="stt_hi_conformer_ctc_medium"):
        self.model_name = model_name
        self.onnx_model_path = f"models/{model_name}_optimized.onnx"
        self.vocab_path = f"models/{model_name}_vocab.pkl"
        self.model_dir = "models"
        
        # Create models directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models
        self.nemo_model = None
        self.onnx_session = None
        self.vocab = None
        
        # Initialize audio preprocessor
        self.preprocessor = AudioPreprocessor()
        
        # Load or create optimized model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize ONNX model or convert from NeMo if needed"""
        try:
            if os.path.exists(self.onnx_model_path) and os.path.exists(self.vocab_path):
                logger.info("Loading existing ONNX model...")
                self._load_onnx_model()
            else:
                logger.info("Converting NeMo model to ONNX...")
                self._convert_to_onnx()
        except Exception as e:
            logger.error(f"Failed to initialize optimized model: {e}")
            logger.info("Falling back to original NeMo model...")
            self._load_nemo_model()
    
    def _load_nemo_model(self):
        """Load original NeMo model as fallback"""
        self.nemo_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            model_name=self.model_name
        )
        self.nemo_model.eval()
        logger.info("NeMo model loaded successfully")
    
    def _convert_to_onnx(self):
        """Convert NeMo model to ONNX format using the correct method"""
        try:
            # Load NeMo model
            nemo_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                model_name=self.model_name
            )
            nemo_model.eval()
            
            # Use NeMo's built-in export method with correct ONNX opset version
            logger.info("Exporting model to ONNX format...")
            nemo_model.export(self.onnx_model_path, onnx_opset_version=14)
            
            # Save vocabulary and decoder info
            vocab_data = {
                'vocab': list(nemo_model.decoder.vocabulary),
                'blank_id': 0,  # CTC blank token is typically at index 0
                'vocab_size': len(nemo_model.decoder.vocabulary)
            }
            
            with open(self.vocab_path, 'wb') as f:
                pickle.dump(vocab_data, f)
            
            logger.info(f"Model successfully converted to ONNX: {self.onnx_model_path}")
            
            # Store the NeMo model for preprocessing
            self.nemo_model = nemo_model
            
            # Load the converted ONNX model
            self._load_onnx_model()
            
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            self._load_nemo_model()
    
    def _load_onnx_model(self):
        """Load ONNX model and vocabulary"""
        try:
            # Configure ONNX Runtime for optimization
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = os.cpu_count()
            sess_options.inter_op_num_threads = os.cpu_count()
            
            self.onnx_session = ort.InferenceSession(
                self.onnx_model_path,
                sess_options,
                providers=providers
            )
            
            # Load vocabulary
            with open(self.vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
                self.vocab = vocab_data['vocab']
                self.blank_id = vocab_data.get('blank_id', 0)
                self.vocab_size = vocab_data.get('vocab_size', len(self.vocab))
            
            # Log model output info for debugging
            output_info = self.onnx_session.get_outputs()[0]
            logger.info(f"ONNX model loaded successfully - Output shape: {output_info.shape}, Vocab size: {self.vocab_size}")
            logger.info(f"ONNX model outputs: {[out.name for out in self.onnx_session.get_outputs()]}")
            
            # Also load NeMo model for preprocessing if not already loaded
            if self.nemo_model is None:
                self.nemo_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                    model_name=self.model_name
                )
                self.nemo_model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self._load_nemo_model()
    
    def _preprocess_to_log_mel(self, audio_file_path, target_sr=16000, preprocessing_intensity='medium'):
        """Enhanced preprocessing with noise reduction before log-mel feature extraction"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_file_path, sr=None)
            
            # Apply comprehensive preprocessing for noise reduction
            enhanced_audio, preprocessing_steps = self.preprocessor.comprehensive_preprocessing(
                audio, sr, intensity=preprocessing_intensity
            )
            
            # Resample to target sample rate if needed
            if sr != target_sr:
                enhanced_audio = librosa.resample(enhanced_audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            # Convert to tensor format expected by NeMo
            audio_tensor = torch.tensor(enhanced_audio, dtype=torch.float32).unsqueeze(0)  # [1, samples]
            length = torch.tensor([audio_tensor.shape[1]], dtype=torch.int64)     # [1]
            
            # Use NeMo model's preprocessor for log-mel features
            log_mel, log_mel_len = self.nemo_model.preprocessor(
                input_signal=audio_tensor, 
                length=length
            )
            
            return log_mel.cpu().numpy(), log_mel_len.cpu().numpy(), preprocessing_steps
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise
    
    def _greedy_ctc_decoder(self, logits, vocab):
        """Fixed CTC decoder that handles tensor dimensions correctly"""
        try:
            # Convert logits to torch tensor if it's numpy array
            if isinstance(logits, np.ndarray):
                probs = torch.tensor(logits)
            else:
                probs = logits
            
            # Handle different tensor shapes
            if len(probs.shape) == 3:  # [batch, time, vocab]
                probs = probs[0]  # Take first batch
            elif len(probs.shape) == 2:  # [time, vocab]
                pass  # Already correct shape
            else:
                raise ValueError(f"Unexpected logits shape: {probs.shape}")
            
            # Debug: Check logits shape and vocab size alignment
            vocab_size = len(vocab)
            logits_vocab_dim = probs.shape[-1]
            
            if logits_vocab_dim != vocab_size:
                logger.warning(f"Logits vocab dimension ({logits_vocab_dim}) != vocab size ({vocab_size})")
                # Clip logits to vocab size if needed
                if logits_vocab_dim > vocab_size:
                    logger.info(f"Clipping logits from {logits_vocab_dim} to {vocab_size}")
                    probs = probs[:, :vocab_size]
            
            # Get argmax indices along vocab dimension
            argmax_indices = torch.argmax(probs, dim=-1)  # [time]
            
            # Convert to list for iteration
            pred = argmax_indices.tolist()
            
            # Ensure pred is a list even if it's a single value
            if isinstance(pred, int):
                pred = [pred]
            
            # Debug: Show prediction statistics
            unique_preds = set(pred)
            max_pred = max(pred) if pred else -1
            logger.info(f"Prediction stats: max_index={max_pred}, vocab_size={vocab_size}, unique_predictions={len(unique_preds)}")
            
            transcript = []
            previous = None
            invalid_count = 0
            
            for i, idx in enumerate(pred):
                # Check if index is within vocab range
                if idx >= vocab_size:
                    invalid_count += 1
                    if invalid_count <= 5:  # Only log first 5 occurrences to avoid spam
                        logger.warning(f"Time step {i}: Index {idx} exceeds vocab size {vocab_size}, skipping")
                    continue
                
                # CTC decode: skip blanks (blank_id) and repeated tokens
                if idx != self.blank_id and idx != previous:
                    if idx < len(vocab):
                        transcript.append(vocab[idx])
                    else:
                        logger.warning(f"Index {idx} not found in vocab (this shouldn't happen)")
                        
                previous = idx
            
            if invalid_count > 5:
                logger.warning(f"Total invalid indices: {invalid_count} (showing only first 5)")
            
            result = ''.join(transcript)
            logger.info(f"Decoded transcript: '{result}' (from {len(pred)} time steps)")
            return result
            
        except Exception as e:
            logger.error(f"CTC decoding failed: {e}")
            return ""
    
    async def transcribe(self, audio_file_path, preprocessing_intensity='medium'):
        """Transcribe audio file using optimized model with enhanced preprocessing"""
        import time
        start_time = time.time()
        
        try:
            if self.onnx_session is not None and self.nemo_model is not None:
                # Use ONNX model for inference with enhanced NeMo preprocessing
                log_mel, log_mel_len, preprocessing_steps = await asyncio.to_thread(
                    self._preprocess_to_log_mel, audio_file_path, preprocessing_intensity=preprocessing_intensity
                )
                
                # Get input names from ONNX session
                input_names = [inp.name for inp in self.onnx_session.get_inputs()]
                
                # Prepare inputs for ONNX inference
                inputs = {
                    input_names[0]: log_mel.astype(np.float32),
                    input_names[1]: log_mel_len.astype(np.int64)
                }
                
                # Run ONNX inference
                outputs = await asyncio.to_thread(
                    self.onnx_session.run, None, inputs
                )
                
                logits = outputs[0]  # First output should be logits
                
                # Decode predictions using greedy CTC decoder
                transcription = await asyncio.to_thread(
                    self._greedy_ctc_decoder, logits, self.vocab
                )
                
            else:
                # Fallback to NeMo model with preprocessing
                # First apply preprocessing to the audio file
                audio, sr = librosa.load(audio_file_path, sr=None)
                enhanced_audio, preprocessing_steps = self.preprocessor.comprehensive_preprocessing(
                    audio, sr, intensity=preprocessing_intensity
                )
                
                # Save preprocessed audio to temporary file
                temp_processed_file = audio_file_path.replace('.', '_processed.')
                sf.write(temp_processed_file, enhanced_audio, sr)
                
                try:
                    transcription_result = await asyncio.to_thread(
                        self.nemo_model.transcribe, [temp_processed_file]
                    )
                    
                    if transcription_result and len(transcription_result) > 0:
                        hypothesis = transcription_result[0]
                        transcription = hypothesis.text if hasattr(hypothesis, 'text') else str(hypothesis)
                    else:
                        transcription = ""
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_processed_file):
                        os.remove(temp_processed_file)
            
            inference_time = time.time() - start_time
            logger.info(f"Transcription completed in {inference_time:.2f}s: {transcription}")
            
            return transcription, inference_time, preprocessing_steps
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return "", time.time() - start_time, "preprocessing failed"

# Initialize optimized ASR model
asr_model = OptimizedASRModel()

def preprocess_audio(file_path):
    """Legacy preprocessing function for compatibility"""
    y, sr = librosa.load(file_path, sr=None)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    y = y / np.max(np.abs(y))
    return y, sr

async def convert_mp3_to_wav(mp3_file_path):
    """Convert MP3 file to WAV format using librosa"""
    try:
        # Load MP3 file
        y, sr = await asyncio.to_thread(librosa.load, mp3_file_path, sr=None)
        
        # Create WAV file path
        wav_file_path = mp3_file_path.replace('.mp3', '_converted.wav').replace('.MP3', '_converted.wav')
        
        # Save as WAV
        await asyncio.to_thread(sf.write, wav_file_path, y, sr)
        
        return wav_file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MP3 to WAV conversion failed: {str(e)}")

@app.post("/transcribe", response_model=TranscriptionResult)
async def transcribe(
    background_tasks: BackgroundTasks, 
    audio_file: UploadFile = File(...),
    preprocessing_intensity: Optional[str] = 'medium'
):
    # Validate preprocessing intensity
    if preprocessing_intensity not in ['low', 'medium', 'high']:
        preprocessing_intensity = 'medium'
    
    # Validate file type
    if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
        raise HTTPException(status_code=400, detail="Supported audio formats: WAV, MP3, M4A, FLAC")
        
    # Check if file is MP3 and needs conversion
    is_mp3_file = audio_file.filename.lower().endswith(('.mp3', '.MP3'))
    converted_from_mp3 = False
    
    # Save uploaded file
    temp_file = f"temp_upload_{os.getpid()}_{audio_file.filename}"
    converted_wav_file = None
    
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Convert MP3 to WAV if needed
        if is_mp3_file:
            logger.info(f"Converting MP3 file {audio_file.filename} to WAV format...")
            converted_wav_file = await convert_mp3_to_wav(temp_file)
            processing_file = converted_wav_file
            converted_from_mp3 = True
        else:
            processing_file = temp_file
            
        # Check audio duration
        y, sr = librosa.load(processing_file, sr=None)
        duration = len(y) / sr
        if duration < 1 or duration > 30:  # More flexible duration limits
            raise HTTPException(status_code=400, detail="Audio duration must be between 1 and 30 seconds")
                
        # Add background task to clean up files
        background_tasks.add_task(cleanup_files, temp_file, converted_wav_file)
                
        # Use optimized transcription with enhanced preprocessing
        transcription, inference_time, preprocessing_steps = await asr_model.transcribe(
            processing_file, preprocessing_intensity=preprocessing_intensity
        )
        
        message = f"Audio transcribed successfully (inference: {inference_time:.2f}s, preprocessing: {preprocessing_intensity})"
        if converted_from_mp3:
            message += " (converted from MP3 to WAV)"
                
        return TranscriptionResult(
            text=transcription,
            success=True,
            message=message,
            converted_from_mp3=converted_from_mp3,
            inference_time=inference_time,
            preprocessing_applied=preprocessing_steps
        )
            
    except Exception as e:
        # Clean up files if error occurs
        cleanup_files(temp_file, converted_wav_file)
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

def cleanup_files(temp_file, converted_wav_file=None):
    """Clean up temporary files"""
    if temp_file and os.path.exists(temp_file):
        os.remove(temp_file)
    if converted_wav_file and os.path.exists(converted_wav_file):
        os.remove(converted_wav_file)

@app.get("/model/info")
async def get_model_info():
    """Get information about the current model"""
    return {
        "model_name": asr_model.model_name,
        "using_onnx": asr_model.onnx_session is not None,
        "onnx_model_exists": os.path.exists(asr_model.onnx_model_path),
        "vocab_exists": os.path.exists(asr_model.vocab_path),
        "vocab_size": getattr(asr_model, 'vocab_size', 0),
        "preprocessing_levels": ["low", "medium", "high"]
    }

@app.post("/model/optimize")
async def optimize_model():
    """Force re-optimization of the model"""
    try:
        # Remove existing ONNX files
        if os.path.exists(asr_model.onnx_model_path):
            os.remove(asr_model.onnx_model_path)
        if os.path.exists(asr_model.vocab_path):
            os.remove(asr_model.vocab_path)
        
        # Re-initialize model
        asr_model._initialize_model()
        
        return {
            "success": True,
            "message": "Model optimization completed",
            "using_onnx": asr_model.onnx_session is not None
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Model optimization failed: {str(e)}"
        }

# Serve the frontend
@app.get("/")
async def get_frontend():
    return FileResponse("frontend/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)