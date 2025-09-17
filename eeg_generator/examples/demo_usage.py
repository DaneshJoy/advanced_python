from eeg_generator import generate_eeg


print("===== Function Generator =====")
gen_samples = []
for i, sample in enumerate(generate_eeg(duration=2)):
    gen_samples.append(sample)
    if len(gen_samples) >= 3:
        break
print(f"Collected {len(gen_samples)} samples")
print(f"Samples:\n{gen_samples}")
print("Finished")