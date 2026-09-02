# Simplified GeoGuessr: Visual Place Recognition from Google Street View Images

A deep learning project for classifying cities from street view images using multiple model architectures. This project compares three different approaches to  visual place recognition: a GSV-based model adapted from an existing open-source implementation, a vision-based transformer model, and our hybrid model, GeoSceneNet, which fuses computer-vision based scene descriptors with CNN image features.

## Demo

[![Watch the demo](https://img.youtube.com/vi/9_rmPO3VvQA/0.jpg)](https://www.youtube.com/watch?v=9_rmPO3VvQA) (linked)

You can also try the web application live on the same GeoGuessr map [here](https://www.geoguessr.com/maps/6927ba2c3537ac4b3ca9fde1).

## Models

### GSV 
Based on the [GSV-Cities](https://github.com/amaralibey/gsv-cities) framework:
- **Backbone**: ResNet50
- **Aggregation**: ConvAP (Convolutional Aggregation Pooling)
- Fine-tuned with a classification head for city prediction

### VLM 
Uses OpenAI's CLIP model:
- **Base Model**: `openai/clip-vit-base-patch32`
- **Architecture**: CLIP vision encoder with a linear classification head
- Leverages pre-trained vision-language representations

### GeoSceneNet 
Our own custom model:
- **Model**: Fusion of CV scene descriptors and CNN image features (ResNet18)
- Classification head predicts off of these features

## Installation

1. Clone the repository:
```bash
git clone https://github.com/onoderamia/CS549.git
cd CS549
```

2. Prepare the GSV-cities repository:
```bash
git submodule init
git submodule update
```

Then comment out line 7 in `gsv/gsv-cities/main.py`:
```python
# from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Scrape your own data OR download our data from [here](https://uofi.box.com/v/cs543-fa25-group14-data)
```bash
cd scraper
echo "API_KEY=[YOUR_GOOGLE_API_KEY]" > .env
python scraper.py
```

## Usage

### Training

**Train the GSV model:**
```bash
cd gsv
python train.py                      # Train from scratch
python train.py ../models/gsv.pth    # Resume from checkpoint
```

**Train the VLM model:**
```bash
cd vlm
python train.py                      # Train from scratch
python train.py ../models/vlm.pth    # Resume from checkpoint
```

**Train the custom model:**
```bash
cd custom
python train.py
```

All training scripts save the model to the `models/` directory. You can also use our pretrained models available [here](https://uofi.box.com/v/cs543-fa25-group14-data)

### Web Application

**Run the web server:**
```bash
cd webapp
python app.py
```
