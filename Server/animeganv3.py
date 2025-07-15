import onnxruntime as ort
import cv2
import numpy as np
import argparse
from flask import Flask, request, jsonify, send_file
from PIL import Image
import io

app = Flask(__name__)

class EnhancedProcessor:
    def __init__(self, model_path):
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 12 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
                'cudnn_conv_use_max_workspace': True,
            }),
            'CPUExecutionProvider'
        ]
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.is_ready = True

    def process_frame_tiled(self, frame, tile_size=1024, overlap=64):
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        step = tile_size - overlap
        tiles_x = (w + step - 1) // step
        tiles_y = (h + step - 1) // step
        
        result = np.zeros((h, w, 3), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        for y in range(tiles_y):
            for x in range(tiles_x):
                start_x = x * step
                start_y = y * step
                end_x = min(start_x + tile_size, w)
                end_y = min(start_y + tile_size, h)
                
                tile = frame_rgb[start_y:end_y, start_x:end_x]
                tile_resized = cv2.resize(tile, (512, 512), interpolation=cv2.INTER_LINEAR)
                
                tile_array = tile_resized.astype(np.float32) / 127.5 - 1.0
                tile_array = np.expand_dims(tile_array, axis=0)
                
                tile_result = self.session.run(None, {self.input_name: tile_array})[0]
                tile_result = (tile_result.squeeze() + 1.0) * 127.5
                tile_result = np.clip(tile_result, 0, 255).astype(np.uint8)
                
                tile_height = end_y - start_y
                tile_width = end_x - start_x
                tile_result = cv2.resize(tile_result, (tile_width, tile_height), interpolation=cv2.INTER_LINEAR)
                
                weight = np.ones((tile_height, tile_width), dtype=np.float32)
                if overlap > 0:
                    fade = overlap // 2
                    if fade > 0:
                        weight[:fade, :] *= np.linspace(0.5, 1, fade)[:, np.newaxis]
                        weight[-fade:, :] *= np.linspace(1, 0.5, fade)[:, np.newaxis]
                        weight[:, :fade] *= np.linspace(0.5, 1, fade)[np.newaxis, :]
                        weight[:, -fade:] *= np.linspace(1, 0.5, fade)[np.newaxis, :]
                
                result[start_y:end_y, start_x:end_x] += tile_result.astype(np.float32) * weight[:,:,np.newaxis]
                weight_map[start_y:end_y, start_x:end_x] += weight
        
        weight_map[weight_map == 0] = 1
        result = result / weight_map[:,:,np.newaxis]
        return cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)

    def process_image(self, image_bytes):
        if not self.is_ready:
            return None
            
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            processed_frame = self.process_frame_tiled(frame)
            
            processed_image = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            
            output_buffer = io.BytesIO()
            processed_image.save(output_buffer, format='JPEG', quality=95, optimize=False)
            output_buffer.seek(0)
            
            return output_buffer.getvalue()

        except Exception as e:
            print(f"Processing error: {e}")
            return None

processor = None

@app.route('/health')
def health():
    global processor
    if processor and processor.is_ready:
        return jsonify({"status": "ready"})
    else:
        return jsonify({"status": "initializing"}), 503

@app.route('/process', methods=['POST'])
def process():
    global processor
    
    if not processor or not processor.is_ready:
        return jsonify({"error": "Processor not ready"}), 503
    
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    image_bytes = image_file.read()
    
    result = processor.process_image(image_bytes)
    
    if result is None:
        return jsonify({"error": "Processing failed"}), 500
    
    return send_file(
        io.BytesIO(result),
        mimetype='image/jpeg',
        as_attachment=False
    )

def main():
    parser = argparse.ArgumentParser(description='AnimeGAN Processing Server')
    parser.add_argument('--port', type=int, default=8001, help='Port to run server on')
    parser.add_argument('--model', type=str, default="deploy/AnimeGANv3_Hayao_36.onnx", help='Path to ONNX model')
    args = parser.parse_args()
    
    global processor
    processor = EnhancedProcessor(args.model)
    
    app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)

if __name__ == "__main__":
    main()