import openai
from openai import OpenAI

openai.api_key = "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"

audio_file = open("speech.mp3", "rb")
transcript = openai.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"
)

print(transcript)