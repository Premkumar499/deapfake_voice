import base64
import sys
import os


def audio_to_base64(audio_path):
    
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found!")
        sys.exit(1)

    with open(audio_path, "rb") as audio_file:
        base64_string = base64.b64encode(audio_file.read()).decode("utf-8")

    return base64_string


if __name__ == "__main__":
    
    audio_path = "/home/premkumar/Downloads/AI Buildathon/human.mp3"

    print(f"Converting: {audio_path}")
    base64_audio = audio_to_base64(audio_path)

    
    output_file = "base64_output.txt"
    with open(output_file, "w") as f:
        f.write(base64_audio)

    print(f"Done! Base64 saved to {output_file} ({len(base64_audio)} characters)")
