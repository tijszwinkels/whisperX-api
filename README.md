# WhisperX API

The WhisperX API is a containerized solution for transcribing audio files with diarization using the powerful [`whisperX`](https://github.com/m-bain/whisperX/) project. This API provides an easy-to-use endpoint for audio transcription and is packaged into a Docker container for easy deployment.

## Prerequisites

- Docker with GPU support. Follow the instructions to install [NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/user-guide.html).
- A Huggingface API token.

### Huggingface Token

Include your Huggingface access token that you can generate from [Here](https://huggingface.co/settings/tokens). After generating the token, accept the user agreement for the following models: 
- [Segmentation](https://huggingface.co/pyannote/segmentation)
- [Voice Activity Detection (VAD)](https://huggingface.co/pyannote/voice-activity-detection)
- [Speaker Diarization](https://huggingface.co/pyannote/speaker-diarization).

## Building the Docker Image

1. Rename config.py.example to config.py and update it with your Huggingface token:

```bash
mv config.py.example config.py
echo "HF_TOKEN = '<my-hf-token>'" > config.py
```

Replace <my-hf-token> with your actual Hugging Face token.

Build the Docker image for the WhisperX API:

```bash
docker build -t whisperx-api --network=host --build-arg hftoken=<my-hf-token> .
```

Again, replace `<my-hf-token>` with your Huggingface token. This might take a while.

## Running the API

After building the Docker image, you can run the WhisperX API with:

```bash
docker run --gpus all -p 5000:5000 whisperx-api
```

This will start the API and make it accessible on port 5000.

## Using the API

To transcribe an audio file, send a POST request to the API endpoint. Here's an example using `curl`:

```bash
curl http://127.0.0.1:5000/transcribe -X POST -F "file=@./audio_en.mp3"
```

Replace `./audio.mp3` with the path to your audio file.

The output looks as following:
```
{
   "segments" : [
      {
         "end" : 10.192,
         "speaker" : "SPEAKER_01",
         "start" : 2.883,
         "text" : " This is a test audio file of about phone line quality in English.",
         "words" : [
            {
               "end" : 3.043,
               "score" : 0.718,
               "speaker" : "SPEAKER_00",
               "start" : 2.883,
               "word" : "This"
            },
            {
               "end" : 3.163,
               "score" : 0.096,
               "speaker" : "SPEAKER_00",
               "start" : 3.123,
               "word" : "is"
            },
            {
               "end" : 3.344,
               "score" : 0.456,
               "speaker" : "SPEAKER_00",
               "start" : 3.324,
               "word" : "a"
            },
            <...>
         ],
      }
   ],
    "word_segments" : [
        {
            "end" : 3.043,
            "score" : 0.718,
            "speaker" : "SPEAKER_00",
            "start" : 2.883,
            "word" : "This"
        },
        {
            "end" : 3.163,
            "score" : 0.096,
            "speaker" : "SPEAKER_00",
            "start" : 3.123,
            "word" : "is"
        },
        {
            "end" : 3.344,
            "score" : 0.456,
            "speaker" : "SPEAKER_00",
            "start" : 3.324,
            "word" : "a"
        },
        <...>
    ]
}
        
```