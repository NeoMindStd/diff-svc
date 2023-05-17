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

spk_id = 'bakAIyaroid'
model_path = 'D:/WorkSpace/bakAIyaroid/checkpoints/bakAIyaroid/model_ckpt_steps_90000.ckpt'
config_path = 'D:/WorkSpace/bakAIyaroid/checkpoints/bakAIyaroid/config.yaml'
hubert_gpu = True
wav_input = 'D:/WorkSpace/bakAIyaroid/target/idol/yoasobi-idol.wav'
pitch_shift = 0
speedup = 5
wav_output = 'D:/WorkSpace/bakAIyaroid/output/output.wav'
add_noise_step = 500
threshold = 0.05
use_crepe = False
use_pe = True
use_gt_mel = True

model = Svc(spk_id, config_path, hubert_gpu, model_path)
f0_test, f0_pred, audio = run_clip(model, file_path=wav_input, key=pitch_shift, acc=speedup, use_crepe=use_crepe, use_pe=use_pe, thre=threshold, use_gt_mel=use_gt_mel, add_noise_step=add_noise_step, project_name=spk_id, out_path=wav_output)
