import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


human_path = "human.mp3"
ai_path = "ai.wav"

human_audio, sr = librosa.load(human_path, sr=16000)
ai_audio, sr = librosa.load(ai_path, sr=16000)

min_len = min(len(human_audio), len(ai_audio))
human_audio = human_audio[:min_len]
ai_audio = ai_audio[:min_len]

time = np.linspace(0, min_len/sr, min_len)

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(time, human_audio)
plt.title("Human Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")


plt.subplot(3, 1, 2)
plt.plot(time, ai_audio)
plt.title("AI Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")


plt.subplot(3, 1, 3)
plt.plot(time, human_audio, label="Human", alpha=0.7)
plt.plot(time, ai_audio, label="AI", alpha=0.7)
plt.title("Overlay Comparison (Human vs AI)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()

plt.savefig("waveform_comparison.png", dpi=300)

plt.show()

print("Image saved as waveform_comparison.png")
