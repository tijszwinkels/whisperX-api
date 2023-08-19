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
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

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
    global model
    #audio_file: FileStorage = request.files['file']
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model
    gc.collect()
    torch.cuda.empty_cache()
    del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print(result["segments"]) # after alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=config.HF_TOKEN, device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio_file)
    #diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    gc.collect()
    torch.cuda.empty_cache()
    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(diarize_segments)
    print(result["segments"]) # segments are now assigned speaker IDs
    return result

# api.add_resource(Transcription, '/transcribe')

if __name__ == '__main__':
    app.run(debug=True)