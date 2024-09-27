import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from io import BytesIO
from core.config import PathConfig, settings
from scripts import sib

app = Flask(__name__, template_folder=PathConfig.TEMPLATE_DIRECTORY)
app.config['BASE_PATH'] = settings.DEPLOYED_BASE_PATH

PathConfig.init_app(app)
sib_instance = sib.SIB(PathConfig)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        data = request.get_json()
        base64_image = data.get('image')
        if not base64_image:
            return jsonify(success=False, message="No image provided"), 400
        
        _, _, processed_image = sib_instance.DetectProblemsInImage(base64_image)

        return jsonify(success=True, image=processed_image), 200
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
