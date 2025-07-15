import os
import asyncio
import uvloop
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import Response
import torch
import numpy as np
from PIL import Image
import io
import time
import cv2
from streamdiffusion.image_utils import postprocess_image
from diffusers import StableDiffusionPipeline, AutoencoderKL
from streamdiffusion import StreamDiffusion
import argparse
import signal
import sys
import traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

os.environ['LD_LIBRARY_PATH'] = ''
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor
    import sys
    gpu_device = int(sys.argv[sys.argv.index('--gpu') + 1]) if '--gpu' in sys.argv else 0
    processor = StableProcessor(gpu_device=gpu_device)
    yield

    if processor:
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

class StableProcessor:
    def __init__(self, gpu_device=0):
        self.device = f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu"
        self.stream = None
        self.is_ready = False
        self.process_count = 0
        self.error_count = 0
        
        self._initialize()

    def _apply_cuda_optimizations(self, gpu_device=0):
        print("Applying optimizations...")
        
        if torch.cuda.is_available():
            with torch.cuda.device(gpu_device):
                memory_limit_gb = 45
                total_memory = torch.cuda.get_device_properties(gpu_device).total_memory
                fraction = (memory_limit_gb * 1024**3) / total_memory
                torch.cuda.set_per_process_memory_fraction(min(fraction, 0.8), gpu_device)
                print(f"Set GPU #{gpu_device} memory limit to {memory_limit_gb}GB")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        
        try:
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, 'enable_math_sdp'):
                torch.backends.cuda.enable_math_sdp(True)
            if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
                torch.backends.cuda.enable_cudnn_sdp(True)
            print("SDP optimizations enabled successfully")
        except Exception as e:
            print(f"SDP optimization warning: {e}")

    def _initialize(self):
        try:
            print(f"Initializing processor on {self.device}")
            
            gpu_id = int(self.device.split(':')[1]) if ':' in self.device else 0
            self._apply_cuda_optimizations(gpu_id)
            
            torch.cuda.empty_cache()
            
            print("Loading pipeline...")
            pipe = StableDiffusionPipeline.from_pretrained(
                "KBlueLeaf/kohaku-v2.1",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(self.device)
            
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
            pipe.vae = vae.to(self.device)
            
            pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
            pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

            self.stream = StreamDiffusion(
                pipe,
                t_index_list=[35, 40, 45],
                torch_dtype=torch.float16,
                cfg_type="none",
            )

            self.stream.load_lcm_lora()
            self.stream.fuse_lora()
            
            print("Compiling (could take 20 mins)...")

            try:
                self.stream.unet = torch.compile(
                    self.stream.unet, 
                    mode="reduce-overhead",
                    fullgraph=True,
                    dynamic=False
                )
                print("Compilation successful with reduce-overhead mode")
            except Exception as e:
                print(f"Compilation failed, running without torch.compile: {e}")
            
            pipe.enable_attention_slicing()
            pipe.enable_model_cpu_offload = lambda: None
            self.stream.scheduler.set_timesteps(50)

            prompt = "anime art style, MAPPA animation style"
            negative_prompt = "small head, tiny head, cropped face, cut off head, shadow face, dark blob head"
            self.stream.prepare(prompt, negative_prompt=negative_prompt)
            
            torch.manual_seed(69)
            torch.cuda.manual_seed(69)
            np.random.seed(69)
            
            print("Running warmup...")
            dummy_input = Image.new("RGB", (1024, 1024))
            for i in range(3):
                try:
                    result = self.stream(dummy_input)
                    print(f"Warmup {i+1}/3 successful")
                except Exception as e:
                    print(f"Warmup {i+1} failed: {e}")
            
            torch.cuda.synchronize()
            
            self.is_ready = True
            print("✅ Processor ready!")
            
        except Exception as e:
            print(f"Failed to initialize: {e}")
            traceback.print_exc()
            self.is_ready = False

    def stable_process(self, image_bytes: bytes) -> bytes:
        if not self.is_ready:
            return None
            
        try:
            self.process_count += 1
            start_time = time.time()
            
            decode_start = time.time()
            try:
                nparr = np.frombuffer(image_bytes, np.uint8)
                cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if cv_image is None:
                    print("OpenCV decode failed, falling back to PIL")
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    cv_image = np.array(pil_image)
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                
                if cv_image.shape[:2] != (1024, 1024):
                    cv_image = cv2.resize(cv_image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                
                pil_image = Image.fromarray(cv_image)
                decode_time = (time.time() - decode_start) * 1000
                
            except Exception as e:
                print(f"Decode error, falling back to PIL: {e}")

                pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                if pil_image.size != (1024, 1024):
                    pil_image = pil_image.resize((1024, 1024), Image.LANCZOS)
                decode_time = (time.time() - decode_start) * 1000
            
            inference_start = time.time()
            torch.manual_seed(69)
            torch.cuda.manual_seed(69)
            np.random.seed(69)
            
            with torch.no_grad():
                try:
                    self.stream.unet.eval()
                    result = self.stream(pil_image)
                    
                    if torch.isnan(result).any() or torch.isinf(result).any():
                        print("WARNING: NaN/inf detected in inference result, regenerating...")

                        torch.manual_seed(42)
                        result = self.stream(pil_image)
                    
                    anime_frame = postprocess_image(result, output_type="pil")[0]
                    
                except Exception as e:
                    print(f"Inference error: {e}")
                    self.error_count += 1
                    anime_frame = pil_image
                    
            inference_time = (time.time() - inference_start) * 1000
            
            encode_start = time.time()
            try:
                anime_array = np.array(anime_frame)
                
                if anime_array.shape[2] == 3:
                    anime_bgr = cv2.cvtColor(anime_array, cv2.COLOR_RGB2BGR)
                else:
                    anime_bgr = cv2.cvtColor(anime_array, cv2.COLOR_RGBA2BGR)
                
                encode_params = [
                    cv2.IMWRITE_JPEG_QUALITY, 85,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 0,
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 0
                ]
                
                success, encoded_img = cv2.imencode('.jpg', anime_bgr, encode_params)
                if not success:
                    raise Exception("OpenCV encoding failed")
                    
                result_bytes = encoded_img.tobytes()
                
            except Exception as e:
                print(f"OpenCV encode failed, using PIL: {e}")

                output_buffer = io.BytesIO()
                anime_frame.save(output_buffer, format='JPEG', quality=85)
                output_buffer.seek(0)
                result_bytes = output_buffer.getvalue()
                
            encode_time = (time.time() - encode_start) * 1000
            total_time = (time.time() - start_time) * 1000
            
            if total_time > 200 or self.process_count % 50 == 0:
                print(f"Process {self.process_count}: Decode {decode_time:.0f}ms + Inference {inference_time:.0f}ms + Encode {encode_time:.0f}ms = {total_time:.0f}ms (Errors: {self.error_count})")
            
            return result_bytes

        except Exception as e:
            print(f"CRITICAL: Process error: {e}")
            traceback.print_exc()
            self.error_count += 1
            return None

def signal_handler(sig, frame):
    print(f"Received signal {sig}, shutting down gracefully...")
    global processor
    if processor:
        torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.get("/health")
async def health():
    global processor
    if processor and processor.is_ready:
        return {
            "status": "ready", 
            "mode": "stable_production",
            "processed": processor.process_count,
            "errors": processor.error_count
        }
    else:
        return {"status": "initializing"}

@app.post("/process")
async def process_image(image: UploadFile = File(...)):
    global processor
    
    if not processor or not processor.is_ready:
        raise HTTPException(status_code=503, detail="Processor not ready")
    
    try:
        image_bytes = await image.read()
        
        result = processor.stable_process(image_bytes)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Processing failed")
        
        return Response(content=result, media_type="image/jpeg")
        
    except Exception as e:
        print(f"Request error: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")