import base64

def audio_to_base64(audio_path):
    """
    Convert audio file to Base64 string
    """
    with open(audio_path, "rb") as audio_file:
        base64_bytes = base64.b64encode(audio_file.read())
        base64_string = base64_bytes.decode("utf-8")

    return base64_string


audio_path = "/home/premkumar/Downloads/AI Buildathon/ai.wav"   # <-- put your audio path here
base64_audio = audio_to_base64(audio_path)

print("Base64 Audio:")
print(base64_audio)
