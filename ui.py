import sys
import os
import json
import time
import numpy as np
import torch
import torchaudio
import gradio as gr
from torchaudio.compliance.kaldi import fbank
from pypinyin import lazy_pinyin
from train_pinyin import MMKWS2_Wrapper

# 设备与模型加载
# device = torch.device("cuda:4")
device = torch.device("cpu")
wrapper = MMKWS2_Wrapper.load_from_checkpoint(
    "stepstep=024500.ckpt",
    map_location=device,
)
wrapper.eval()

# 注册信息
registered = {"text": "", "audios": []}
enroll = None
enroll_text = None
last_wake_time = 0

def load_pinyin_index(save_path):
    """加载拼音索引映射"""
    with open(save_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["pinyin_to_index"], data["index_to_pinyin"]

pinyin_to_index, index_to_pinyin = load_pinyin_index("pinyin_index.json")

def add_audio(text, audio, audio_list):
    if not text:
        return audio_list, "请先输入唤醒词文本"
    if audio is None or audio[1] is None or len(audio[1]) == 0:
        return audio_list, "请上传或录制音频"
    audio_list = audio_list or []
    if len(audio_list) >= 5:
        return audio_list, "最多支持5条音频"
    audio_list.append(audio)
    return audio_list, f"已录入 {len(audio_list)} 条音频"

def register_keyword(text, audio_list):
    if not text:
        return gr.update(value="请先输入唤醒词文本")
    if not audio_list or len(audio_list) == 0:
        return gr.update(value="请至少上传或录制一条音频")
    registered["text"] = text
    registered["audios"] = audio_list
    global enroll_text
    enroll_text = text
    fused_feats = []
    for audio in audio_list:
        anchor_wave, _ = torchaudio.load(audio)
        anchor_text_embedding = torch.tensor([pinyin_to_index[p] + 1 for p in lazy_pinyin(text)])
        anchor_wave = anchor_wave.to(device)
        anchor_text_embedding = anchor_text_embedding.to(device).unsqueeze(0)
        with torch.no_grad():
            outputs = wrapper._hubert_model(anchor_wave.half())
            anchor_wave_embedding = outputs.last_hidden_state
        anchor_wave_embedding = anchor_wave_embedding.to(anchor_wave.dtype)
        fused_feat = wrapper.model.enrollment(
            anchor_wave_embedding,
            anchor_text_embedding
        )
        fused_feats.append(fused_feat)
    fused_feats = torch.cat(fused_feats, dim=0)
    fused_feats, _ = fused_feats.max(dim=0)
    fused_feats = fused_feats.unsqueeze(0)
    global enroll
    enroll = fused_feats
    return gr.update(value=f"注册完成，唤醒词：{text}，音频数：{len(audio_list)}")

def update_gallery(audio_list):
    if audio_list and len(audio_list) > 0:
        return gr.update(visible=True, value=audio_list[-1])
    else:
        return gr.update(visible=False, value=None)

def streaming_detect_handler(current_audio, state, audio_player):
    global last_wake_time, enroll_text, enroll
    if current_audio is None or current_audio[1] is None or len(current_audio[1]) == 0:
        return state, gr.update()
    if enroll_text is None:
        return state, gr.update()
    pad = len(enroll_text) * 5
    state = (state or []) + [current_audio]
    state = state[-pad:]
    if len(state) < pad:
        return state, gr.update()
    sr = state[0][0]
    audio_list = [x[1] for x in state]
    audio_concat = np.concatenate(audio_list, axis=0)
    audio_concat = audio_concat.astype(np.float32) / 32768.0
    audio_concat = torch.from_numpy(audio_concat).unsqueeze(0)
    audio_concat = torchaudio.functional.resample(audio_concat, sr, 16000)
    audio_concat = audio_concat / torch.max(torch.abs(audio_concat))
    compare_wave = fbank(audio_concat, num_mel_bins=80)
    compare_wave = compare_wave.to(device).unsqueeze(0)
    compare_lengths = torch.tensor([compare_wave.size(1)], device=compare_wave.device)
    if enroll is None:
        return state, None
    current_time = time.time()
    if current_time - last_wake_time <= 2:
        return state, gr.update()
    with torch.no_grad():
        preds = wrapper.model.verification(
            enroll,
            compare_wave,
            compare_lengths
        )
    preds = torch.sigmoid(preds).item()
    if preds >= 0.85:
        print(f"Wake up! {preds}")
        last_wake_time = current_time
        audio_path = "tts-2025-04-27@197cd135f2a2451b9cab9cf2add1c1ab.wav"
        state = []
        return state, gr.Audio(value=audio_path, visible=True, autoplay=True)
    if preds >= 0.6:
        print(f"Preds! {preds}")
    return state, None

with gr.Blocks() as demo:
    gr.Markdown("# 自定义关键词检测 Demo")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 注册唤醒词")
            text_input = gr.Textbox(label="唤醒词文本", placeholder="请输入唤醒词")
            audio_list_state = gr.State([])
            audio_input = gr.Audio(label="上传或录制音频", type="filepath")
            add_btn = gr.Button("添加音频")
            audio_status = gr.Textbox(label="音频状态", interactive=False)
            audio_gallery = gr.Audio(label="已添加音频", type="filepath", interactive=False, visible=False)
            register_btn = gr.Button("注册完成")
            register_status = gr.Textbox(label="注册状态", interactive=False)
            add_btn.click(
                add_audio,
                inputs=[text_input, audio_input, audio_list_state],
                outputs=[audio_list_state, audio_status]
            ).then(
                update_gallery,
                inputs=audio_list_state,
                outputs=audio_gallery
            )
            register_btn.click(
                register_keyword,
                inputs=[text_input, audio_list_state],
                outputs=register_status
            )
        with gr.Column(scale=2):
            gr.Markdown("## 实时检测")
            mic = gr.Audio(sources="microphone", streaming=True, label="实时监听")
            state = gr.State(value=[])
            audio_player = gr.Audio(label="唤醒提示", visible=False)
            mic.stream(
                streaming_detect_handler,
                inputs=[mic, state, audio_player],
                outputs=[state, audio_player],
                time_limit=1000,
                stream_every=0.05,
            )
if __name__ == "__main__":
    demo.launch()