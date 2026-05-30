import requests
import sys
import time
from time import sleep
from threading import Thread
import json
import os
import numpy as np
from scipy.io import wavfile
import sounddevice as sd
import whisper

END_POINT = "http://localhost:3000/mindwave/data"

data = []

def grab_eeg_data():
    response  = requests.get(END_POINT).json()
    return response

def collect_data(duration, delay):
    global data
    start_time = time.time()
    while time.time() - start_time < duration:
        packet = []
        for i in range(10):
            packet.append(grab_eeg_data())
            sleep(delay)
        data.append(packet)

def main(arguments = []):
    duration = int(arguments[0]) if len(arguments) > 0 else 20
    delay = float(arguments[1]) if len(arguments) > 1 else .1

    input_dir = 'input'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    
    eeg_thread = Thread(target=collect_data, args=(duration, delay))
    eeg_thread.start()

    audio = sd.rec(int(duration * 44100), samplerate=44100, channels=2, dtype='int16')
    sd.wait()
    wavfile.write(os.path.join(input_dir, 'audio.wav'), 44100, audio)

    audio_file = os.path.join(input_dir, 'audio.wav')
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, word_timestamps=True)

    time_stamped_transcription = []
    for segment in result['segments']:
        for word in segment['words']:
            time_stamped_transcription.append({
                'time': (word['start'], word['end']),
                'text': word['word']
            })

    eeg_thread.join()

    
    with open(os.path.join(input_dir, 'data.json'), 'w') as f:
        json.dump(data, f, indent=4)

    with open(os.path.join(input_dir, 'transcription.json'), 'w') as f:
        json.dump(time_stamped_transcription, f, indent=4)

if __name__ == "__main__":
    main(sys.argv[1:])