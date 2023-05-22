from utils.hparams import hparams
from preprocessing.data_gen_utils import get_pitch_parselmouth, get_pitch_crepe
import numpy as np
import matplotlib.pyplot as plt
import utils
import librosa
import torchcrepe
from infer import *
import logging
from infer_tools.infer_tool import *

logging.getLogger('numba').setLevel(logging.WARNING)

# spk_id = 'bakAIyaroid'
# model_path = 'D:/WorkSpace/diff-svc-trains_and_result/checkpoints/bakAIyaroid/model_ckpt_steps_202000.ckpt'
# config_path = 'D:/WorkSpace/diff-svc-trains_and_result/checkpoints/bakAIyaroid/config.yaml'
spk_id = 'dio'
model_path = 'D:/WorkSpace/diff-svc-trains_and_result/checkpoints/dio/model_ckpt_steps_50000.ckpt'
config_path = 'D:/WorkSpace/diff-svc-trains_and_result/checkpoints/dio/config.yaml'
hubert_gpu = True
wav_input = 'D:/WorkSpace/diff-svc-trains_and_result/target/idol/yoasobi-idol.wav'
pitch_shift = 0 # 0~12의 값, 키를 올리기만 가능
speedup = 5
wav_output = 'D:/WorkSpace/diff-svc-trains_and_result/output/IDOL/dio-IDOL.wav'
add_noise_step = 500   # 높을수록 학습데이터가 우선, 500~1000 추천
threshold = 0.05
use_crepe = False   # True시 쉰 목소리
use_pe = True
use_gt_mel = True

model = Svc(spk_id, config_path, hubert_gpu, model_path)
f0_test, f0_pred, audio = run_clip(model, file_path=wav_input, key=pitch_shift, acc=speedup, use_crepe=use_crepe, use_pe=use_pe, thre=threshold, use_gt_mel=use_gt_mel, add_noise_step=add_noise_step, project_name=spk_id, out_path=wav_output)
print(f'spk_id = {spk_id}\nmodel_path = {model_path}\nconfig_path = {config_path}\nwav_input = {wav_input}\nwav_output = {wav_output}')
