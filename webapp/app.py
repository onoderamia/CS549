import os
import sys
import tempfile
import importlib.util
from PIL import Image
from flask import Flask, render_template, request, jsonify
from io import BytesIO
import base64

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'out'))
GSV_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'gsv'))
VLM_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'vlm'))
CUSTOM_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'custom'))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))

CHECKPOINTS = {
    'gsv': os.path.join(MODELS_DIR, 'gsv.pth'),
    'vlm': os.path.join(MODELS_DIR, 'vlm.pth'),
    'custom': os.path.join(MODELS_DIR, 'custom.pth')
}

sys.path.insert(0, GSV_DIR)
sys.path.insert(0, VLM_DIR)
sys.path.insert(0, CUSTOM_DIR)
sys.path.insert(0, os.path.join(GSV_DIR, 'gsv-cities'))


def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

UTILS = {
    'gsv': load_module_from_path('gsv_model_utils', os.path.join(GSV_DIR, 'model_utils.py')),
    'vlm': load_module_from_path('vlm_model_utils', os.path.join(VLM_DIR, 'model_utils.py')),
    'custom': load_module_from_path('custom_model_utils', os.path.join(CUSTOM_DIR, 'model_utils.py'))
}

MODELS = {
    'gsv': None,
    'vlm': None,
    'custom': None
}

app = Flask(__name__)

def load_model(model_type):
    global MODELS, UTILS
    if MODELS[model_type] is not None:
        return MODELS[model_type]
    
    UTILS[model_type].load_cities(DATA_DIR)
    MODELS[model_type] = UTILS[model_type].load_model(CHECKPOINTS[model_type])
    MODELS[model_type].eval()
    
    print(f"{model_type} model loaded on {UTILS[model_type].DEVICE} with {len(UTILS[model_type].CITIES)} cities")
    return MODELS[model_type]

def classify_with_model(image_path, model_type):
    predictions = UTILS[model_type].classify_image(MODELS[model_type], image_path)
    
    return [
        {
            'city': city,
            'confidence': float(confidence),
            'percentage': f'{confidence:.1%}'
        }
        for city, confidence in predictions
    ]

def save_temp_image(image_data):
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(temp_file.name, 'JPEG')
    
    return temp_file.name


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    temp_path = None
    try:
        data = request.json
        image_data = data.get('image')
        model_type = data.get('model', 'gsv')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        temp_path = save_temp_image(image_data)
        
        if model_type in ['gsv', 'vlm', 'custom']:
            predictions = classify_with_model(temp_path, model_type)
            return jsonify({
                'success': True,
                'model': model_type,
                'predictions': predictions,
                'top_prediction': predictions[0]['city'],
                'top_confidence': predictions[0]['percentage']
            })
        
        elif model_type == 'all':
            gsv_predictions = classify_with_model(temp_path, 'gsv')
            vlm_predictions = classify_with_model(temp_path, 'vlm')
            custom_predictions = classify_with_model(temp_path, 'custom')
            
            return jsonify({
                'success': True,
                'model': 'all',
                'gsv': {
                    'predictions': gsv_predictions,
                    'top_prediction': gsv_predictions[0]['city'],
                    'top_confidence': gsv_predictions[0]['percentage']
                },
                'vlm': {
                    'predictions': vlm_predictions,
                    'top_prediction': vlm_predictions[0]['city'],
                    'top_confidence': vlm_predictions[0]['percentage']
                },
                'custom': {
                    'predictions': custom_predictions,
                    'top_prediction': custom_predictions[0]['city'],
                    'top_confidence': custom_predictions[0]['percentage']
                }
            })
        
        else:
            return jsonify({'error': f'Unknown model type: {model_type}'}), 400
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == '__main__':
    # preload models
    print("Loading GSV model...")
    load_model("gsv")
    print("Loading VLM model...")
    load_model("vlm")
    print("Loading Custom model...")
    load_model("custom")
    print("Starting server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
