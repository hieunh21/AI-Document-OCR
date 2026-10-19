"""Build notebooks/finetune_ppocrv6_rec_vi.ipynb (cells kept as plain text here)."""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": text.strip("\n"),
        }
    )


md(r"""
# Fine-tune PP-OCRv6 medium rec cho tiếng Việt (VietDocAI)

**Vì sao cần:** bộ ký tự gốc của PP-OCRv6 rec (18,708 ký tự) **thiếu toàn bộ 90 chữ tiếng Việt có dấu thanh** (ẩ, ả, ổ, ờ, ộ, ị…), nên model không thể in ra chúng → "Câm Phå" thay vì "Cẩm Phả". Notebook này giữ backbone đã huấn luyện của PP-OCRv6 medium, **thay bộ ký tự bằng bộ tiếng Việt 237 ký tự** và fine-tune toàn bộ model.

**Công thức vòng 3** (cùng config đã train trên máy local, xem `docs/finetune_rec.md`):
- ảnh huấn luyện rộng **320 px** như config gốc. Backbone PPLCNetV4 khi train luôn pool về **40 bước thời gian**; vòng 1–2 dùng 640 px nên model học ký tự hẹp gấp đôi thực tế và **lặp ký tự** khi chạy ("119.000" → "1119.000");
- lớp đầu ra chép từ model gốc (`remap_rec_head.py`): ký tự có trong cả hai bộ giữ nguyên trọng số, chữ có dấu thanh lấy từ chữ gốc của nó ("ẩ" từ "â");
- chỉ dữ liệu thật, 15 epoch, batch 16, learning rate 1e-4 (cosine).

**Dữ liệu** (tạo trên máy bằng `python scripts/export_rec_dataset.py --zip`):
- `train_list.txt`: ~3.7k dòng cắt từ ảnh **train** của MC-OCR (đúng khung detector, đã dựng thẳng ảnh xoay);
- `val_list.txt`: ~0.8k dòng từ **dev** — chỉ để chọn checkpoint; **test không có ở đây**;
- dòng tổng hợp sinh trên Colab (`synth_rec_lines.py`) từ nhãn train + danh mục hành chính.

**Cách chạy:** upload `outputs/rec_dataset.zip` lên Google Drive (mặc định `MyDrive/vietdocai/`), chọn *Runtime → Change runtime type → GPU*, rồi chạy lần lượt từng ô. Checkpoint được lưu thẳng vào Drive, nên nếu Colab ngắt phiên thì chạy lại ô cài đặt + ô **Resume**.

**Kết quả cần mang về máy:** `MyDrive/vietdocai/ppocrv6_medium_rec_vi3_inference.zip` → giải nén vào `models/ppocrv6_medium_rec_vi3/` trong repo.
""")

md("## 1. GPU và đường dẫn")
code(r"""
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')

DRIVE_DIR = '/content/drive/MyDrive/vietdocai'      # nơi đặt rec_dataset.zip
DATA_ZIP  = f'{DRIVE_DIR}/rec_dataset.zip'
OUT_DIR   = f'{DRIVE_DIR}/ppocrv6_medium_rec_vi3'    # checkpoint vòng 3 (lưu trên Drive)
WORK      = '/content/work'
DATA      = f'{WORK}/rec_dataset'
import os
os.makedirs(WORK, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)
assert os.path.exists(DATA_ZIP), f'Chưa thấy {DATA_ZIP}: upload outputs/rec_dataset.zip lên Drive'
""")

md(r"""
## 2. Cài đặt
Cùng phiên bản Paddle với máy local (`paddlepaddle-gpu 3.3.1`, CUDA 12.6) để model export ra chạy được ở cả hai nơi. PaddleOCR repo (có `tools/train.py` và config PP-OCRv6) được clone và ghi lại commit.
""")
code(r"""
# Colab cài sẵn torch build cho NCCL mới; paddlepaddle-gpu kéo theo nvidia-nccl-cu12 cũ hơn, nên
# `import torch` (paddleocr tự import nếu có torch) lỗi "undefined symbol: ncclCommWindowDeregister".
# Notebook không dùng torch -> gỡ đi. pip có thể để lại thư mục torch/ rỗng một phần; khi đó
# `import torch` vẫn "thành công" (namespace package) và modelscope lỗi "No module named
# 'torch.multiprocessing'", nên xoá hẳn các thư mục còn sót.
!pip -q uninstall -y torch torchvision torchaudio
import site, shutil
for sp in site.getsitepackages():
    for d in ('torch', 'torchvision', 'torchaudio', 'torchgen', 'functorch'):
        shutil.rmtree(f'{sp}/{d}', ignore_errors=True)
!pip -q install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
!pip -q install paddleocr==3.7.0 fonttools
!apt-get -qq install -y fonts-noto-core fonts-dejavu-core fonts-liberation2 > /dev/null
%cd {WORK}
!test -d PaddleOCR || git clone -q --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git
!pip -q install -r PaddleOCR/requirements.txt
!git -C PaddleOCR rev-parse HEAD
import paddle; paddle.utils.run_check(); print(paddle.__version__)
""")

md("## 3. Giải nén dữ liệu thật và sinh dữ liệu tổng hợp")
code(r"""
!unzip -q -o {DATA_ZIP} -d {WORK}
!cat {DATA}/manifest.json
!wc -l {DATA}/train_list.txt {DATA}/val_list.txt {DATA}/dict_vi.txt
""")
code(r"""
N_SYNTH = 0   # vòng 3 chỉ dùng dữ liệu thật; đặt 30000 để thử thêm dòng tổng hợp (~3-6 phút CPU)
if N_SYNTH and not os.path.exists(f'{DATA}/synth/synth_list.txt'):
    get_ipython().system(
        f'cd {DATA} && python synth_rec_lines.py --out synth --n {N_SYNTH} '
        '--corpus corpus_train.txt --admin admin_names.txt --fonts /usr/share/fonts'
    )
!wc -l {DATA}/synth/synth_list.txt 2>/dev/null || echo 'no synthetic data'
""")
md(r"""
## 4. Danh sách huấn luyện và config
- dòng thật lặp `REAL_REPEAT` lần (1; vòng 1 dùng 3 và model học thuộc cả lỗi gõ trong nhãn train);
- ảnh huấn luyện rộng **320** px (đúng config gốc): khi train, backbone luôn pool về 40 bước thời gian, nên độ rộng khác 320 làm lệch giữa lúc train và lúc chạy. **Đừng tăng `WIDTH`.** Lúc chạy thật, ảnh dài vẫn được giữ đúng tỉ lệ (số bước tăng theo độ rộng);
- `max_text_length` 25 → **36**: 40 bước thời gian chỉ chứa được nhãn khoảng 36 ký tự; 2.6 % nhãn train dài hơn sẽ bị bỏ qua;
- `character_dict_path` = `dict_vi.txt`; lớp đầu ra (CTC + NRTR) có kích thước mới và được dựng từ lớp đầu ra gốc (mục cuối của phần này), nên mọi lớp đều bắt đầu từ trọng số đã huấn luyện;
- learning rate 1e-4 (vòng 1: 3e-4) vì không còn lớp nào học từ đầu.
""")
code(r"""
def rows(f): return [l.rstrip('\n').split('\t') for l in open(f, encoding='utf-8')]
real = rows(f'{DATA}/train_list.txt')
synth = rows(f'{DATA}/synth/synth_list.txt') if os.path.exists(f'{DATA}/synth/synth_list.txt') else []
REAL_REPEAT = 1
with open(f'{DATA}/train_all.txt', 'w', encoding='utf-8') as fh:
    for _ in range(REAL_REPEAT):
        fh.writelines(f'{p}\t{t}\n' for p, t in real)
    fh.writelines(f'{p}\t{t}\n' for p, t in synth)
n_train = REAL_REPEAT * len(real) + len(synth)
print('train samples per epoch:', n_train, f'(real {len(real)} x {REAL_REPEAT}, synthetic {len(synth)})')
""")
code(r"""
# Xem nhanh vài ảnh: thật (train) và tổng hợp
import random, matplotlib.pyplot as plt
from PIL import Image
sample = random.Random(0).sample(real, 6) + random.Random(1).sample(synth, min(6, len(synth)))
fig, axes = plt.subplots(len(sample), 1, figsize=(10, 1.1 * len(sample)))
for ax, (p, t) in zip(axes, sample):
    ax.imshow(Image.open(f'{DATA}/{p}')); ax.set_title(t, fontsize=9, loc='left'); ax.axis('off')
plt.tight_layout()
""")

code(r"""
import yaml, math
BATCH, EPOCHS, LR, WIDTH, MAX_LEN = 16, 15, 1e-4, 320, 36

cfg = yaml.safe_load(open(f'{WORK}/PaddleOCR/configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml'))
iters_per_epoch = n_train // BATCH
g = cfg['Global']
g.update(
    epoch_num=EPOCHS, save_model_dir=OUT_DIR, save_epoch_step=EPOCHS, eval_batch_step=[0, iters_per_epoch],
    pretrained_model=f'{WORK}/PP-OCRv6_medium_rec_vi_init', character_dict_path=f'{DATA}/dict_vi.txt',
    use_space_char=True, max_text_length=MAX_LEN, distributed=False, print_batch_step=50,
    d2s_train_image_shape=[3, 48, WIDTH], save_res_path=f'{OUT_DIR}/predicts.txt',
)
cfg['Optimizer']['lr'].update(learning_rate=LR, warmup_epoch=0.3)
for h in cfg['Architecture']['Head']['head_list']:
    if 'NRTRHead' in h: h['NRTRHead']['max_text_length'] = MAX_LEN

tr = cfg['Train']
tr['dataset'].update(data_dir=DATA, label_file_list=[f'{DATA}/train_all.txt'], ratio_list=[1.0])
for t in tr['dataset']['transforms']:
    if 'RecConAug' in t: t['RecConAug'].update(image_shape=[48, WIDTH, 3], max_text_length=MAX_LEN)
tr['sampler'].update(scales=[[WIDTH, 32], [WIDTH, 48], [WIDTH, 64]], first_bs=BATCH)
tr['loader'].update(batch_size_per_card=BATCH, num_workers=4)

ev = cfg['Eval']
ev['dataset'].update(data_dir=DATA, label_file_list=[f'{DATA}/val_list.txt'])
for t in ev['dataset']['transforms']:
    if 'RecResizeImg' in t: t['RecResizeImg']['image_shape'] = [3, 48, WIDTH]
ev['loader'].update(batch_size_per_card=64, num_workers=2)

CONFIG = f'{OUT_DIR}/PP-OCRv6_medium_rec_vi.yml'
yaml.safe_dump(cfg, open(CONFIG, 'w'), allow_unicode=True, sort_keys=False)
print(CONFIG, '| iters/epoch', iters_per_epoch, '| total iters', iters_per_epoch * EPOCHS)
""")
code(r"""
# Pretrained weights chính thức (cùng file PaddleX dùng để fine-tune)
!cd {WORK} && test -f PP-OCRv6_medium_rec_pretrained.pdparams || wget -q https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams
!ls -la {WORK}/*.pdparams
""")
code(r"""
# Dựng lớp đầu ra cho bộ ký tự mới từ lớp đầu ra gốc (thay vì khởi tạo ngẫu nhiên)
!python {DATA}/remap_rec_head.py --pretrained {WORK}/PP-OCRv6_medium_rec_pretrained.pdparams --stock-dict {WORK}/PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt --new-dict {DATA}/dict_vi.txt --out {WORK}/PP-OCRv6_medium_rec_vi_init.pdparams
# Kỳ vọng: "237 chars: 149 copied, 88 from a base letter, 0 mean"
""")

md(r"""
## 5. Huấn luyện
Log in ra `acc` (khớp nguyên dòng) và `norm_edit_dis` (1 − CER chuẩn hoá) trên dev sau mỗi epoch; checkpoint tốt nhất là `best_accuracy`. Khi bắt đầu **không** được có cảnh báo *"shape … not matched"*: mọi lớp đã khớp kích thước nhờ bước dựng lớp đầu ra. Nếu thấy cảnh báo đó thì dừng lại và kiểm tra ô `remap_rec_head.py`.

`BATCH` 16 là cấu hình đã train trên máy local (GPU 4 GB, ~1.5 phút/epoch); trên Colab cũng chỉ mất vài phút.
""")
code(r"""
%cd {WORK}/PaddleOCR
!python tools/train.py -c {CONFIG} 2>&1 | tee {OUT_DIR}/train.log | grep -E "not matched|cur metric|best metric|global_step: [0-9]*000,|Error|error"
""")
md("**Resume** (chỉ khi phiên bị ngắt giữa chừng): chạy lại mục 1–4 rồi ô này.")
code(r"""
%cd {WORK}/PaddleOCR
!python tools/train.py -c {CONFIG} -o Global.checkpoints={OUT_DIR}/latest 2>&1 | tee -a {OUT_DIR}/train.log | grep -E "best metric|Error|error"
""")

md(
    "## 6. Export model inference và đo CER trên dev bằng chính model export (như lúc chạy ở máy local)"
)
code(r"""
%cd {WORK}/PaddleOCR
INFER_DIR = f'{OUT_DIR}/inference'
!python tools/export_model.py -c {CONFIG} -o Global.pretrained_model={OUT_DIR}/best_accuracy Global.save_inference_dir={INFER_DIR}
!ls -la {INFER_DIR}
""")
code(r"""
# Ô này tự đủ: sau khi restart session chỉ cần chạy lại mục 1 rồi chạy ô này.
import unicodedata
from paddleocr import TextRecognition

INFER_DIR = f'{OUT_DIR}/inference'
def rows(f): return [l.rstrip('\n').split('\t') for l in open(f, encoding='utf-8')]

def digits(s): return ''.join(c for c in s if c.isdigit())

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def noacc(s):
    s = s.replace('đ', 'd').replace('Đ', 'D')
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

val = rows(f'{DATA}/val_list.txt')
models = {'PP-OCRv6 gốc': TextRecognition(model_name='PP-OCRv6_medium_rec'),
          'fine-tuned vi3': TextRecognition(model_name='PP-OCRv6_medium_rec', model_dir=INFER_DIR)}
for name, m in models.items():
    err = err_na = n = n_na = exact = dg_err = dg_n = 0
    preds = m.predict([f'{DATA}/{p}' for p, _ in val], batch_size=32)
    for (p, gt), r in zip(val, preds):
        pr = r['rec_text']
        g, q = ' '.join(gt.lower().split()), ' '.join(pr.lower().split())
        err += lev(g, q); n += len(g); exact += g == q
        err_na += lev(noacc(g), noacc(q)); n_na += len(noacc(g))
        if digits(g): dg_err += lev(digits(g), digits(q)); dg_n += len(digits(g))
    print(f'{name:14s} CER {err/n:.2%} | CER no-accent {err_na/n_na:.2%} | digit CER {dg_err/dg_n:.2%} | line exact {exact/len(val):.1%}')
""")
code(r"""
# Đóng gói để tải về máy: giải nén vào models/ppocrv6_medium_rec_vi3/ trong repo
INFER_DIR = f'{OUT_DIR}/inference'
!cd {INFER_DIR} && zip -q -r {DRIVE_DIR}/ppocrv6_medium_rec_vi3_inference.zip . && ls -la {DRIVE_DIR}/ppocrv6_medium_rec_vi3_inference.zip
!cp {OUT_DIR}/train.log {DRIVE_DIR}/ 2>/dev/null; git -C {WORK}/PaddleOCR rev-parse HEAD > {DRIVE_DIR}/paddleocr_commit.txt
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = Path("notebooks/finetune_ppocrv6_rec_vi.ipynb")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(out, len(cells), "cells")
