import os
import sys
import logging
from typing import Optional
import uuid

from pydantic import BaseModel
import torchaudio
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydub import AudioSegment

MATCHA_TTS_DIR = os.environ.get('MATCHA_TTS_DIR', '/opt/Matcha-TTS')
sys.path.append(MATCHA_TTS_DIR)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


ASSETS_DIR = os.environ.get('ASSETS_DIR', '/data/assets')
default_prompt_speech_16k = load_wav(os.path.join(ASSETS_DIR, 'zero_shot_prompt.wav'), 16000)


class Params(BaseModel):
    tts_text: str
    instruct_text: str
    prompt_wav: Optional[str] = None


@app.post("/instruct2")
async def inference_instruct2(params: Params):
    if params.prompt_wav:
        prompt_speech_16k = load_wav(params.prompt_wav, 16000)
    else:
        prompt_speech_16k = default_prompt_speech_16k

    request_id = uuid.uuid4().hex
    OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/data/output')
    os.makedirs(f'{OUTPUT_DIR}/{request_id}', exist_ok=True)

    audio_segments = []
    for i, j in enumerate(cosyvoice.inference_instruct2(params.tts_text, params.instruct_text, prompt_speech_16k)):
        file_path = f'{OUTPUT_DIR}/{request_id}/{i}.wav'
        torchaudio.save(file_path, j['tts_speech'], cosyvoice.sample_rate)
        audio_segments.append(AudioSegment.from_wav(file_path))

    if len(audio_segments) > 1:
        combined = sum(audio_segments)
        combined.export(f'{OUTPUT_DIR}/{request_id}/combined.wav', format='wav')
        result_file = f'{request_id}/combined.wav'
    else:
        result_file = f'{request_id}/0.wav'

    OUTPUT_URL_ROOT = os.environ.get('OUTPUT_URL_ROOT', 'http://localhost:8000')
    return {'result': f'{OUTPUT_URL_ROOT}/{result_file}'}   



if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    MODEL_DIR = os.environ.get('MODEL_DIR', '/data/models/CosyVoice2-0.5B')
    try:
        cosyvoice = CosyVoice2(MODEL_DIR)
    except Exception:
        raise TypeError('no valid model_type!')
    uvicorn.run(app, host="0.0.0.0", port=port)