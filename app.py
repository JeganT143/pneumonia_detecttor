import os
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the saved model
model = tf.keras.models.load_model('pneumonia_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((300, 300))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Preprocess the image and make prediction
            processed_image = preprocess_image(file_path)
            prediction = model.predict(processed_image)[0][0]
            
            result = {
                'prediction': float(prediction),
                'diagnosis': 'Pneumonia' if prediction > 0.5 else 'Normal',
                'confidence': float(prediction) if prediction > 0.5 else 1 - float(prediction),
                'image_path': file_path
            }
            
            return render_template('result.html', result=result)
        
        return render_template('index.html', error='File type not allowed')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)