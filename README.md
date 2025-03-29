# Pneumonia Detection Web App

![Pneumonia Detection](https://img.shields.io/badge/AI-Medical%20Imaging-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-brightgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey)

A web application that uses deep learning to detect pneumonia from chest X-ray images. This project demonstrates how AI can be used to assist in medical diagnosis.

## Overview

This project combines TensorFlow for deep learning and Flask for web deployment to create an interactive pneumonia detection system. Users can upload chest X-ray images and receive instant diagnostic predictions.



## Features

- Deep learning model trained on chest X-ray images
- User-friendly web interface for image upload
- Real-time image processing and prediction
- Visual results with confidence scores
- Responsive design that works on mobile devices

## Dataset

The model was trained on the Chest X-Ray Images (Pneumonia) dataset, which includes:
- Normal chest X-ray images
- Pneumonia chest X-ray images (bacterial and viral)
- https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Model Architecture

The CNN model consists of multiple convolutional and pooling layers:
- 5 convolutional layers with increasing filters (16→32→64→64→64)
- Max pooling after each convolutional layer
- Dense layer with 512 units
- Binary classification output

## Directory Structure

```
pneumonia-detector/
├── app.py                  # Flask application
├── pneumonia_model.h5      # Trained model
├── requirements.txt        # Dependencies
├── static/
│   └── uploads/            # Uploaded images storage
├── templates/
│   ├── index.html          # Upload page
│   └── result.html         # Results page
└── test/                   # Test images
```

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/pneumonia-detection-app.git
cd pneumonia-detection-app
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
python app.py
```

4. Open your browser and go to:
```
http://127.0.0.1:5000
```

## How to Use

1. Open the web application
2. Click the "Choose File" button to upload a chest X-ray image
3. Click "Detect Pneumonia"
4. View the diagnosis result and confidence score

## Testing

Sample X-ray images are included in the `test` directory to help you test the application's functionality.

## Model Training

The model was trained using TensorFlow on a dataset of labeled chest X-ray images. Training process included:
- Image rescaling and normalization
- Data prefetching and caching for performance
- 15 epochs of training
- RMSprop optimizer with learning rate of 0.001
- Binary cross-entropy loss function



## Future Improvements

- [ ] Add data augmentation for improved model accuracy
- [ ] Implement Grad-CAM visualization to highlight areas of interest
- [ ] Add user authentication for medical use cases
- [ ] Create a mobile app version
- [ ] Expand to detect other lung conditions

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Contact

Feel free to connect with me on LinkedIn or email with any questions!
