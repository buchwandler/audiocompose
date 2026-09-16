from utterrender import TTS

with TTS(default_voice="kokoro:af_heart") as tts:
    tts.to_wav("Hello from one interface.", "hello.wav", language="en-us")
