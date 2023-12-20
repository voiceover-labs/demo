import openai
import whisper

# OpenAI API 키
openai.api_key = "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"

def transcribe_whisper(audio_path):
    # Whisper 모델 로드
    model = whisper.load_model("base")

    # 오디오 로드 및 처리
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # 언어 감지
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # 오디오 디코딩
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    return result.text

