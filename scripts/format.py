import os
import sys
import json
import random

def main():
    input_dir = "input"

    samples_dir = os.path.join(input_dir, "samples")
    if not os.path.exists(input_dir) or not os.path.exists(samples_dir):
        print("ERROR: Please make an input and sample directory before running this application")
        sys.exit(1)
    
    train_dir = os.path.join(input_dir, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    test_dir = os.path.join(input_dir, "test")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    samples = []
    files = os.listdir(samples_dir)
    for file in files:
        with open(os.path.join(samples_dir, file), 'r') as f:
            sample_set = json.load(f)
        for sample in sample_set:
            if sample["output"]["text"]:
                samples.append(sample)

    with open(os.path.join(input_dir, "compiled.json"), "w") as f:
        json.dump(samples, f, indent=4)

    proportion = .7
    point = int(len(samples) * proportion)
    random.shuffle(samples)
    print(f"{len(samples)=} {point=}")

    proportion = 1
    train_samples = samples[:point]
    random.shuffle(train_samples)
    train_samples = train_samples[:int(len(train_samples) * proportion)]
    train_file = os.path.join(train_dir, "train.json")
    with open(train_file, "w") as f:
        json.dump(train_samples, f, indent=4)

    test_samples = samples[point:]
    test_file = os.path.join(test_dir, "test.json")
    with open(test_file, "w") as f:
        json.dump(test_samples, f, indent=4)

if __name__ == "__main__":
    main()