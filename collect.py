import requests
import sys
import uuid
import time
from time import sleep
import json
import os
import re
import webbrowser
from urllib.parse import urlencode
from sentence_transformers import SentenceTransformer
from simple_term_menu import TerminalMenu

BASE_URL = "http://localhost:3000"
END_POINT = f"{BASE_URL}/mindwave/data"
COLLECT_URL = f"{BASE_URL}/collect"
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

data = []
words_config: dict

def pair_eeg_with_word(eeg_packets, word):
    """Map each EEG packet to words spoken during its time window."""
    paired = []
    index = 0
    embedding = MODEL.encode(word)

    if word not in words_config:
        words_config[word] = {
            'embedding': MODEL.encode(word).tolist(),
        }

    for packet in eeg_packets:
        output = {
            'text': word,
            'time': [packet[0]["time"], packet[-1]["time"]],
            'dimensions' : embedding.shape[0],
            'embedding': embedding.tolist()
        }
        paired.append({'input': packet, 'output': output})
        index += 1
    return paired

def grab_eeg_data():
    response  = requests.get(END_POINT).json()
    return response

def collect_data(duration, delay):
    global data
    print("EEG data collection started")

    start_time = time.time()
    while time.time() - start_time < duration:
        packet = []
        for i in range(int(1/delay)):
            raw_eeg_data = grab_eeg_data()
            raw_eeg_data['time'] = time.time() - start_time
            packet.append(raw_eeg_data)
            sleep(delay)
        data.append(packet)

def grabText():
    TEXTS_DIR = 'texts'
    
    if not os.path.isdir(TEXTS_DIR):
        os.makedirs(TEXTS_DIR)

    files = sorted(
        name for name in os.listdir(TEXTS_DIR)
        if os.path.isfile(os.path.join(TEXTS_DIR, name))
    )
    if not files:
        sys.exit(f"No text files found in '{TEXTS_DIR}/'. Add files and try again.")

    menu = TerminalMenu(
        files,
        title="Select a text file (↑/↓ to move, Enter to confirm):",
    )
    choice = menu.show()
    if choice is None:
        sys.exit("No file selected.")

    with open(os.path.join(TEXTS_DIR, files[choice]), 'r') as f:
        return f.read()

def selectWord(words):
    menu = TerminalMenu(
        words,
        title = f"Select your word (↑/↓ to move, Enter to confirm):"
    )
    return words[menu.show()]


def main(arguments = []):
    duration = int(arguments[0]) if len(arguments) > 0 else 20
    delay = float(arguments[1]) if len(arguments) > 1 else .1
    text = arguments[2] if len(arguments) > 2 else grabText()

    words_config_dir = "output/words_config.json"
    if os.path.exists(words_config_dir):
        with open(words_config_dir, 'r') as f:
            words_config = json.load(f)
    else:
        words_config = {}

    input_dir = 'input/samples'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    
    words = "\n".join([line for line in text.split("\n") if not line.strip().startswith('#')])
    words = re.findall(r'\b[a-zA-Z0-9\']+\b', words)
    word = selectWord(words)

    params = {
        'text': word,
        'duration': duration,
        'delay': 0
    }
    collect_url = f"{COLLECT_URL}?{urlencode(params)}"
    webbrowser.open(collect_url)
    
    collect_data(duration, delay)

    paired = pair_eeg_with_word(data, word)

    with open(os.path.join(input_dir, f'{uuid.uuid4()}.json'), 'w') as f:
        json.dump(paired, f, indent=4)
    
    with open(words_config_dir, 'w') as f:
        json.dump(words_config, f, indent=4)

if __name__ == "__main__":
    main(sys.argv[1:])