import requests
import sys
import time
from time import sleep
from threading import Thread
import json
import os
from scipy.io import wavfile
import sounddevice as sd
import whisper
from sentence_transformers import SentenceTransformer

END_POINT = "http://localhost:3000/mindwave/data"
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

start = None
data = []

def packet_time_range(packet):
    times = [sample['time'] for sample in packet]
    return (min(times), max(times)) if times else (0.0, 0.0)

def word_overlaps_packet(word, packet_start, packet_end):
    word_start, word_end = word['time']
    return word_start < packet_end and word_end > packet_start

def pair_eeg_with_transcription(eeg_packets, words):
    """Map each EEG packet to words spoken during its time window."""
    paired = []
    for packet in eeg_packets:
        p_start, p_end = packet_time_range(packet)
        matched = [
            (w['time'][0], w['text'])
            for w in words
            if word_overlaps_packet(w, p_start, p_end)
        ]
        matched.sort(key=lambda item: item[0])
        text = ' '.join(text.strip() for _, text in matched)
        embedding = MODEL.encode(text)
        output = {
            'text': text,
            'time': [p_start, p_end],
            'dimensions' : embedding.shape[0],
            'embedding': embedding.tolist()
        }
        paired.append({'input': packet, 'output': output})
    return paired

def grab_eeg_data():
    response  = requests.get(END_POINT).json()
    return response

def collect_data(duration, delay):
    global data, start
    start = time.time()
    start_time = start
    while time.time() - start_time < duration:
        packet = []
        for i in range(int(1/delay)):
            raw_eeg_data = grab_eeg_data()
            raw_eeg_data['time'] = time.time() - start
            packet.append(raw_eeg_data)
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

    print("Recording audio...")
    audio = sd.rec(int(duration * 44100), samplerate=44100, channels=2, dtype='int16')
    sd.wait()
    wavfile.write(os.path.join(input_dir, 'audio.wav'), 44100, audio)

    print("Transcribing audio...")
    audio_file = os.path.join(input_dir, 'audio.wav')
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, word_timestamps=True)
    print("Audio transcribed successfully")
    os.remove(audio_file)

    transcription = []
    for segment in result['segments']:
        for word in segment['words']:
            transcription.append({
                'time': (word['start'], word['end']),
                'text': word['word']
            })

    eeg_thread.join()

    paired = pair_eeg_with_transcription(data, transcription)

    with open(os.path.join(input_dir, 'paired.json'), 'w') as f:
        json.dump(paired, f, indent=4)

    

if __name__ == "__main__":
    main(sys.argv[1:])