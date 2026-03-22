import base64


with open("audio_base64.txt", "r") as f:
    audio_base64_string = f.read()

audio_base64_bytes = audio_base64_string.encode("utf-8")

audio_bytes = base64.b64decode(audio_base64_bytes)

with open("output2.mp3", "wb") as f:
    f.write(audio_bytes)

print("Base64 converted back to MP3 successfully")