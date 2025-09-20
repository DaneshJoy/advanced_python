#%%
from eeg_generator import (
    generate_eeg,
    EEGStream,
    plot_eeg_samples,
    live_plot_stream
)
sampling_rate = 10

#%%
print("===== Function Generator =====")
gen_samples = []
for i, sample in enumerate(generate_eeg(n_channel=4, duration=1)):
    gen_samples.append(sample)
    print(f'Sample {i}: {sample}')
print(f"Collected {len(gen_samples)} samples")

#%%
print("\n===== Static Plot (Function Generator) =====")
fig1 = plot_eeg_samples(gen_samples, sampling_rate=sampling_rate)

#%%
print("===== Class Generator =====")
class_gen_samples = []
stream = EEGStream(n_channel=4, duration=1)
for i, sample in enumerate(stream):
    class_gen_samples.append(sample)
    print(f'Sample {i}: {sample}')
print(f"Collected {len(class_gen_samples)} samples")

#%%
print("\n===== Static Plot (Class Generator) =====")
fig2 = plot_eeg_samples(class_gen_samples, sampling_rate=sampling_rate)

#%%
print("\n===== Live Plot (Function Generator) =====")
live_stream = generate_eeg(n_channel=4, duration=3, sampling_rate=sampling_rate)
live_plot_stream(live_stream, n_channels=4, sampling_rate=sampling_rate)
# %%
