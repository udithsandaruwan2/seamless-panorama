import base64
import os
import json
from flask import Flask, request, render_template, url_for
import cv2
from utils import sharpen_image, equalize_histogram, denoise, color_correction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/preview')
def uploadedPreview():
    return render_template('uploaded-preview.html')

@app.route('/edit')
def edit():
    return render_template('edit.html')

@app.route('/process-images', methods=['POST'])
def process_images():
    image_data = request.form.get('imageData')
    
    if image_data:
        images = json.loads(image_data)
        saved_images = []
        
        if not os.path.exists('static/tmp'):
            os.makedirs('static/tmp')
        
        for idx, img_url in enumerate(images):
            # Assuming img_url is a base64 encoded image data URL
            if img_url.startswith('data:image'):
                img_type = img_url.split(';')[0].split('/')[1]
                img_data = img_url.split(',')[1]
                img_path = f'static/tmp/{idx + 1}.{img_type}'
                with open(img_path, 'wb') as img_file:
                    img_file.write(base64.b64decode(img_data))
                # Read the saved image and append it to the list
                img = cv2.imread(img_path)
                if img is not None:
                    saved_images.append(img)
                else:
                    print(f"Error loading image from path: {img_path}")
        
        if not saved_images:
            return "No valid images received", 400
        
        result_list = []
        
        for img in saved_images:
            # Apply the denoise function
            denoised_image = denoise(img)
            
            # Append the result to result_list
            result_list.append(img)
         
        # Create stitcher instance
        stitcher = cv2.Stitcher_create() if hasattr(cv2, 'Stitcher_create') else cv2.createStitcher(False)
        status, result = stitcher.stitch(result_list)
        
        if status == cv2.Stitcher_OK:
            # result = denoise(result)
            result = equalize_histogram(result)
            result = sharpen_image(result)
            # result = color_correction(result)
            
            result_path = 'static/processed_image.jpg'
            if cv2.imwrite(result_path, result):
                return render_template('processed.html', result_image=url_for('static', filename='processed_image.jpg'))
            else:
                return "Error saving processed image", 500
        else:
            return f"Stitching failed with status {status}", 500
            
    else:
        return "No images received", 400

if __name__ == '__main__':
    app.run(debug=True)
