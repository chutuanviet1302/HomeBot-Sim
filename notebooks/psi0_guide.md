# Ψ₀ (Psi0) cho HomeBot-Sim: hướng dẫn từng bước

Đường đi: dữ liệu HomeBot → định dạng Ψ₀ (máy này) → fine-tune trên Kaggle 2× T4 (chạy nền, "Save & Run All") →
một phiên Kaggle ngắn chạy server suy luận + tunnel HTTPS → mô phỏng trên máy này gọi server (`--policy psi0`).

Máy này có GPU GTX 1650 4 GB. Ψ₀ cần ~6 GB chỉ để suy luận (VLM Qwen3-VL-2B 4,3 GB + action expert ~1 GB),
nên **server chạy trên Kaggle**; còn mô phỏng và đánh giá chạy trên máy này.

## 1. Dataset (đã upload: `chutunvit/homebot-can-psi0`, riêng tư)

- `data/psi0/homebot_can_psi0.zip` (9,3 MB) = `data/psi0/real/homebot_can`: 60 tập, 22 087 khung, `states` 30-D,
  `action` 34-D, video `observation.images.egocentric` 224×224 h264.
- Tạo lại: `python scripts/export_psi0.py` rồi nén thư mục `data/psi0/real/homebot_can`.
  (`meta/episodes_stats.jsonl`, LeRobot v2.1 cần, được notebook tự sinh.)

## 2. Huấn luyện (Kaggle, chạy nền)

Notebook `notebooks/kaggle_finetune_psi0.ipynb` (đã import vào `chutunvit/notebook00ad13c4bd`), input là dataset
trên, *Settings*: GPU T4 x2, Internet On. **Save Version → Save & Run All (Commit)**: chạy nền tới 12 giờ, không cần
giữ trình duyệt (phiên tương tác bị dừng khi máy/trình duyệt ngủ), output `/kaggle/working/psi0_homebot` (~6,5 GB)
được lưu cùng version. Khi chạy nền, các ô serve tự bỏ qua.

Thông số đã đo trên 2× T4 (2026-10-21):

| | fp16 | bf16 (dùng) |
|---|---|---|
| Loss | `nan` từ bước 1 (Qwen3-VL tràn số) | 44 → ~25–31 sau 19 bước |
| Tốc độ | ~3 s / micro-step (batch 2/GPU) | ~7,8 s / micro-step (bf16 giả lập trên T4) |
| Bộ nhớ đỉnh | 14,0 / 15,1 GB mỗi GPU | tương tự |

Thời gian: cài đặt ~10 phút, tải checkpoint 6,3 GB ~5 phút, probe 20 bước ~15 phút (khởi động ~10 phút), rồi
huấn luyện `TRAIN_HOURS = 3,8` giờ: batch toàn cục 16 (2 × 2 GPU × 4 tích lũy), ~440 bước ≈ 0,3 epoch. Probe tự tính
số bước từ thời gian thực. Tổng ~4,5 giờ. (Tác giả: 8 GPU bf16, batch 128, 40 000 bước - mô hình ở đây được huấn
luyện ít hơn nhiều.)

Các bản vá notebook áp lên Ψ₀ (`cda3fdf`) cho T4/Kaggle: `uv.lock` hỏng ở upstream (xóa, resolve từ PyPI);
`pyarrow==17` (Ψ₀ ghim numpy 1.x); `uv pip --python .venv-psi/bin/python` (trên Kaggle `uv pip` trỏ vào Python hệ
thống); `flash_attention_2` → `sdpa`; `.env` (`train.py`/server `assert load_dotenv()`); `PSI_HOME` ngoài repo (repo có
thư mục `real/`); `--train.optimizer-foreach=False`; với fp16 thêm: `torch.bfloat16` → `float16` và sửa GradScaler
(`unscale_` gọi một lần mỗi bước).

## 3. Server suy luận (phiên Kaggle tương tác, ~1–2 giờ)

1. Mở notebook → *Add Input* → *Your Work* → chọn output của version vừa chạy.
2. Ô đầu: `SERVE_ONLY = "auto"` (tự tìm `psi0_homebot/run_config.json` trong `/kaggle/input`). Tải 5,9 GB
   output vào phiên mất khoảng 1–1,5 giờ (~1 MB/s).
3. **Run All**: chỉ cài đặt (~10 phút) rồi chạy `serve_psi0_simple` (GPU 0, `--action-exec-horizon 10`), tự kiểm
   tra `/act`, mở tunnel cloudflared và in `PUBLIC URL: https://<...>.trycloudflare.com`. Giữ tab mở.

Đo được (2026-10-21): mỗi lần gọi `/act` mất 5–8 s với mặc định (10 bước flow, 10 hành động mỗi lần). Để E2 vừa
khoảng 1 giờ, server đã chạy với `--action-exec-horizon=20 --num-inference-steps=5` (~3 s mỗi lần gọi, 20 hành động).
Checkpoint trong `/kaggle/input` được chép sang `/tmp/run` trước (đọc mmap trực tiếp từ dataset làm server treo > 45
phút; chép tuần tự 5,9 GB mất 39 s).

## 4. Đánh giá trên máy này

```bash
curl https://<...>.trycloudflare.com/health
```

```bash
python scripts/eval_manip.py --policy psi0 --host https://<...>.trycloudflare.com --n 20 --workers 2
```

```bash
python scripts/eval_full.py --policy psi0 --host https://<...>.trycloudflare.com --n 10 --level 2 --workers 2
```

```bash
python -m web.backend.ingest
```

Kết quả: `results/e2_manip_psi0.csv`, `results/e3_full_level2_psi0.csv` (cột `policy` = `psi0`). Mỗi lần gọi trả
10 hành động (0,4 s mô phỏng); mỗi lượt gắp/thả tối đa 60 lần gọi. Server không giữ trạng thái giữa các request
(không RTC), nên nhiều worker song song là an toàn.

Kết quả 2026-10-21/22 (checkpoint 300 bước, `chutunvit/psi0-homebot-ckpt`): E2 gắp 0/10, thả 5/10; E3 0/5 (hỏng ở
bước gắp). Đường ống chạy trọn; mô hình được huấn luyện quá ít (≈ 0,2 epoch) để gắp được.

## 5. Các thành phần trong repo

- `scripts/export_psi0.py`: mapping HomeBot → Ψ₀ (xem docstring).
- `homebot/policy.py`: `psi0_state` (44-D + chiều cao chậu → 30-D) và `psi0_client(host)` (HTTP `POST /act`,
  định dạng của `serve_psi0_simple`). `LearnedSkillPolicy._apply` thực thi hành động 34-D trong không gian khớp:
  7 khớp tay phải `[19:26]` → `env.arm_q_goal` (giới hạn tốc độ như expert, không IK), 6 ngón `[6:12]` → `env.fingers`.
- `scripts/eval_manip.py`: `make_policy("psi0")`; `eval_full.py` dùng lại hàm này.
- `notebooks/make_notebooks.py`: sinh notebook (phần Ψ₀).
