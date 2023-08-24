import whisperx
import gc
import torch
import os
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import config

device = "cuda"
audio_file = "audio.mp3"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
min_speakers=1
max_speakers=10
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {"mp3", "wav", "awb", "aac", "ogg", "oga", "m4a", "wma", "amr"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
diarize_model = whisperx.DiarizationPipeline(use_auth_token=config.HF_TOKEN, device=device)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # check if the post request has the file part
    print(request.files)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = transcribe(filepath)
        return jsonify(result)
    else:
        return jsonify({'error': 'Invalid file type'}), 400

def transcribe(audio_file):
    global model, diarize_model
    # 1. Transcribe with original whisper (batched)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # 3. Assign speaker labels
    diarize_segments = diarize_model(audio_file)

    # Return result
    result = whisperx.assign_word_speakers(diarize_segments, result)
    return result

if __name__ == '__main__':
    app.run(debug=True)