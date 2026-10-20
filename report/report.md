# HomeBot-Sim — Báo cáo kết quả

Robot hình người **Unitree G1** (29 bậc tự do) với bàn tay 5 ngón **Inspire RH56** trong một căn hộ mô phỏng bằng
MuJoCo (khoảng 8 × 6 m). Nhiệm vụ chính: *"bring the can to the sink"* (mang lon tới bồn rửa): đi từ phòng khách tới bàn ăn,
tìm lon bằng camera, gắp bằng tiếp xúc ngón tay, mang sang bếp và thả vào bồn rửa trong khi có người (và thú cưng)
đi lại.

## 1. Hệ thống

| Thành phần | Hiện thực |
|---|---|
| Robot | Menagerie `unitree_g1` (bỏ bàn tay Dex3) + bàn tay Inspire RH56 từ `unitree_ros` (khớp mimic = equality, 6 actuator vị trí mỗi tay) |
| Đi bộ | Chính sách đi bộ đã huấn luyện của Unitree (`unitree_rl_gym`, `motion.pt`) + vòng PI bám vận tốc (bù lệch ngang ~9 cm/s của chính sách trên mô hình này) |
| Cảm biến | LiDAR 2D (`mj_multiRay`), camera RGB-D trên đầu, camera cổ tay |
| Điều hướng | A* 1 Hz → DWA 10 Hz → lớp an toàn 50 Hz → lệnh vận tốc |
| Nhận thức vật | RANSAC mặt đỡ → phân cụm → nhận diện (ICP so với lưới YCB) → tư thế 6D → theo dõi Kalman |
| Thao tác | Expert lập trình: căn chỉnh bằng bước chân → đứng + khóa thân → nắm kiểu bắt tay (power grasp) → mang → thả vào bồn |
| Dữ liệu | LeRobot v2.1 + `modality.json` (GR00T), trạng thái 44-D, hành động 17-D |

Chi tiết kiến trúc, an toàn và cách chạy: xem `README.md`. Video minh họa: `report/demo_full_task.mp4` (mức 2, seed 2, thành công trong 210 s).

## 2. Thí nghiệm

### E1 — Điều hướng A → B (100 lượt mỗi mức) và E4 — tắt lớp an toàn

Điểm đầu / đích ngẫu nhiên cách nhau ≥ 3 m (đích cách vật cản ≥ 0,5 m), ghế dịch ngẫu nhiên ±0,3 m mỗi lượt,
giới hạn 90 s. Mức 1: đồ đạc; mức 2: + một người đi lại (0,8–1,2 m/s, không nhường robot đang đi);
mức 3: + người + thú cưng.

| Điều kiện | Thành công | Va chạm đồ đạc | Lần chạm người (robot đang đi) | Lần chạm thú cưng | Ngã | Thời gian TB (s) |
|---|---|---|---|---|---|---|
| Mức 1 | **99%** | 0 | — | — | 0 | 26,0 |
| Mức 2 | **94%** | 0 | 16 (7) | — | 0 | 46,2 |
| Mức 3 | **80%** | 0 | 13 (11) | 7 | 0 | 54,3 |
| E4: mức 2, tắt an toàn | 99% | 0 | **27 (26)** | — | 0 | 39,6 |

- Không va chạm đồ đạc và không ngã ở mọi điều kiện (trước khi sửa: 698–1907 lần chạm đồ đạc / 100 lượt). Lượt
  không thành công đều là hết giờ (90 s), không phải va chạm.
- **E4**: tắt lớp an toàn, số lần chạm người tăng từ 16 lên 27, và số lần chạm *khi robot đang đi* tăng từ 7 lên 26
  (×3,7), dù tỷ lệ tới đích cao hơn. Lớp an toàn đổi thời gian (46 s so với 40 s) lấy an toàn.
- Chạm còn lại phần lớn do người mô phỏng *cố tình không nhường* robot đang đi và đi tới 1,2 m/s, trong khi dáng đi
  của G1 phanh chậm. Mức 3 thấp hơn chủ yếu vì hết giờ: robot dừng hoặc nhường nhiều khi có hai vật động.

### E2 — Gắp / thả (50 lượt mỗi kỹ năng, xuất phát lệch ±10 cm / ±15°)

Vị trí lon do **nhận thức camera** ước lượng (không đọc trạng thái mô phỏng); vật: lon súp YCB `005_tomato_soup_can`.
Gắp: robot bước căn chỉnh, đứng + khóa thân, nắm, nhấc 12 cm, đưa về tư thế mang, giữ 1 s. Thả: robot cầm lon (một lần
gắp thật ở bàn rồi dịch tới gần bồn), bước căn chỉnh, đưa lon lên trên miệng bồn và thả; thành công khi lon nằm trong
lòng bồn và tay đã nhả.

| Kỹ năng | Thành công | Sai số nhận thức (trung vị) | Lần chạm tay | Dừng tay (> 25 N) | Rơi vật | Ngã |
|---|---|---|---|---|---|---|
| Gắp | **50/50 (100%)** | 1,7 mm | 0 | 0 | 0 | 0 |
| Thả vào bồn | **50/50 (100%)** | 1,5 mm | 34 | 7 | 0 | 0 |

- Lần chạm tay khi thả là các chạm nhẹ (mép quầy / thân robot) lúc đưa tay ra vào. Bộ tìm đường cánh tay phải đi
  vòng ở mọi lượt thả (100 đường thẳng bị loại vì va chạm) và luôn tìm được đường không va chạm. Trước bản sửa
  đường đi cánh tay có một lượt tay tì vào thân robot với lực 104 N (332 lần dừng); sau sửa: 7 lần dừng trên 50 lượt.

### E3 — Toàn bộ nhiệm vụ có người đi lại (30 lượt, mức 2)

Lệnh *"bring the can to the sink"*: robot xuất phát ngẫu nhiên ở phòng khách, đi tới bàn ăn, tìm và gắp lon, mang
sang bếp, thả vào bồn. Ghế dịch ngẫu nhiên, một người đi lại suốt lượt, giới hạn 180 s cho mỗi chặng điều hướng.

| Thành công | Thời gian TB (s) | Lần chạm người (robot đang đi) | Ngã | Chạm tay khi thao tác (phải / trái) | Dừng tay |
|---|---|---|---|---|---|
| **26/30 (87%)** | 168 | 7 (7) | 0 | 11 / 18 | 5 |

- 4 lượt hỏng: 2 lượt lon tuột khỏi tay trong chặng mang dài (87–103 s, nhiều lần dừng/tránh người) nên tới bồn
  thì tay đã trống; 1 lượt thả không thành công; 1 lượt hết giờ trên đường tới bồn.
- 29 lần "chạm đồ đạc" của E3 trùng đúng với số lần chạm của hai tay khi đứng làm việc ở bàn / quầy (tay trái buông
  chạm mặt quầy khi đứng sát). Không có lần chạm đồ đạc nào khi đi (giống E1).
- Trước các bản sửa cuối (dáng tay, bù lượng tử hóa bản đồ LiDAR), E3 là 15/30 (50%), hỏng chủ yếu vì robot đứng
  yên sau khi gắp, không tìm được đường tới bồn.


### Dataset (LeRobot v2.1)

`data/homebot_can`: **60 tập** (30 "pick up the can from the table", 30 "drop the can into the sink"), **25 828 khung**
ở 25 Hz, video camera đầu + cổ tay 224×224, trạng thái 44-D (29 khớp G1, 6 khớp Inspire, tư thế lòng bàn tay 9-D),
hành động 17-D (tư thế lòng bàn tay 9-D, chiều cao chậu, cờ đứng yên, 6 lệnh ngón), kèm `meta/modality.json` cho GR00T.
Expert (dùng nhận thức camera) thành công 60/61 lần thử; chỉ giữ tập thành công. Seed của dataset tách biệt với seed
đánh giá E2 (≥ 200 000). Notebook Kaggle: `notebooks/kaggle_train_act.ipynb`, `notebooks/kaggle_finetune_gr00t.ipynb`.

## 3. Các lỗi đã tìm và sửa (tóm tắt)

| Hiện tượng | Nguyên nhân | Sửa |
|---|---|---|
| Gắp chỉ 25% dù bàn thử 100% | Khóa thân khi xương chậu đang lắc → weld mềm kéo trôi 13 cm; căn chỉnh theo vị trí tức thời (lắc ±3 cm) | Khóa khi vận tốc trung bình ổn định, weld cứng, căn theo vị trí trung bình nhịp bước |
| Ngón khép trên miệng vật | Nhiễu độ cao 8 mm + bù 2 cm | Độ cao lấy theo mặt đỡ đã biết (nhiễu 2 mm) |
| Cốc nhỏ tuột, mug vướng quai | Vật 13 g quá nhẹ; quai mug chắn ngón | Dùng lon súp YCB (mug giữ lại làm vật gây nhiễu cho nhận diện) |
| Đặt lon thì lon đổ | Lon nghiêng 20–35° trong tay khi đi | Thay "đặt" bằng "thả vào bồn" (bồn trũng 18 cm thật) |
| Căn chỉnh hết giờ khi dùng camera | Điều kiện dừng theo ước lượng vị trí có nhiễu trắng 2 cm | Dừng theo sai số đã lọc thông thấp |
| Camera không thấy lon khi đứng gần | Lon và bàn tay robot gộp thành một cụm | Bộ lọc tự thân (self-filter) theo mô hình + động học robot |
| Track lon "trôi" | Kalman vận tốc + nhiều khung cùng thời điểm | Mô hình bước ngẫu nhiên (vật đứng yên tới khi bị dời) |
| Robot cọ tường/ghế (698 lần / 100 lượt) | Bán kính robot 0,30 m < footprint thật 0,37 m; DWA tưởng robot bước ngang được 0,2 m/s (thật 0,08) | Bán kính theo đo đạc; DWA chỉ dùng bước ngang thật |
| Robot đứng chết ở khe hẹp | A* chấp nhận khe DWA không đi được; A* đổi lộ trình mỗi giây | A* có biên dư so với DWA; giữ lộ trình (hysteresis) |
| Ghế ngẫu nhiên chưa bao giờ được dịch | Điều kiện bố cục luôn sai → luôn dùng bố cục gốc | Sửa điều kiện; bếp bố trí lại với lối đi ≥ 0,9 m |
| Robot chạm người đi ra từ cửa | Người bị vách che, robot phanh chậm (~0,8 m/s²) | Tiêu chí thời gian-tới-va-chạm, phanh khẩn, giảm tốc ở góc khuất |
| Camera "không thấy" lon (gắp 82%) | RANSAC chọn mặt ghế (0,90 m) thay vì mặt bàn | Tìm vật trên mọi mặt phẳng ngang trong tầm nhìn |
| Sai số nhận thức 11 mm | Bộ lọc tự thân theo hình cầu bao xóa mất nửa trên của lon | Lọc theo hộp bao định hướng của từng geom |
| Tay tì vào thân robot 104 N khi rút tay | Đường đi khớp va chạm vẫn được thực hiện nếu đường vòng cũng va chạm | Tìm đường qua nhiều điểm trung gian, chỉ đi đường không va chạm |
| E3: robot kẹt sau khi gắp | Lùi thẳng về điểm "pre" bị ghế chắn; kiểm tra khoảng trống cấm cả lùi ra xa bàn | Lùi 0,35 m rồi để A* lập kế hoạch; cho phép đi ra xa vật cản |
| Bài đánh giá bị treo | Mỗi lượt tạo renderer OpenGL không giải phóng → hết RAM, worker chết | Đóng renderer, tái tạo worker định kỳ |
| Dáng đi thiếu tự nhiên | Tay phải giơ cứng ở tư thế "sẵn sàng" suốt quãng đi, tay trái cố định | Hai tay buông theo thân và đánh tay ngược pha với chân (theo góc hông đo được); chỉ nâng tay khi tới nơi làm việc, theo đường khớp đã kiểm tra va chạm |
| Bàn tay buông chạm mép bàn khi tới gần | Tay còn buông khi robot bước sát bàn | Nâng tay lên tư thế sẵn sàng *trước* khi bước sát bàn |
| E3: robot đứng yên sau khi gắp, không có đường tới bồn | Ô bản đồ LiDAR được phồng từ tâm ô → ghế "to" hơn thật ~3 cm, khe hẹp cạnh bàn bị đóng | Phồng ô nhớ LiDAR bớt nửa ô (bù lượng tử hóa) |
| Xem trong MuJoCo bị giật | Nhận thức camera (0,5 s) và tìm đường cánh tay (1,5 s) làm chậm vòng mô phỏng | `view.py` mô phỏng trước rồi phát lại đúng thời gian thực |

## 4. Giới hạn (cần đọc trước khi trích dẫn số liệu)

- **Không dùng GEAR-SONIC**: lệnh vận tốc đi vào chính sách đi bộ của Unitree; `sonic_bridge.py` chỉ giới hạn lệnh.
- **Khóa thân khi thao tác**: chính sách đi bộ không đứng yên được (luôn bước tại chỗ), nên khi tay làm việc thân robot
  được giữ cố định (weld) — giống dự án HomeHand.
- **GR00T N1.7 / ACT chưa được huấn luyện ở đây**: dataset, cấu hình và notebook Kaggle đã sẵn; số liệu thao tác là của
  expert (dùng nhận thức camera thật).
- **E2 phần thả**: bắt đầu bằng một lần gắp thật ở bàn rồi dịch robot + lon tới gần bồn (chỉ để dựng bài thử; E3 đi bộ cả
  quãng).
- **Người / thú cưng** là khối động học (mocap) với hành vi nhường đơn giản; người cố tình *không bao giờ* nhường robot
  đang đi (trường hợp xấu nhất); va chạm đo bằng khoảng cách footprint.
- Tư thế robot đưa cho bộ điều hướng là tư thế thật + nhiễu 2 cm (không có SLAM).
- **Dáng đi**: chính sách đi bộ được huấn luyện trên mô hình G1 12 bậc của Unitree; trên mô hình này (Menagerie G1 có
  quán tính rotor khớp, bàn tay Inspire) nó trôi ngang ~0,12 m/s và đi gối hơi khuỵu. Vòng PI vận tốc triệt tiêu độ
  trôi (robot đi đúng hướng) nhưng bước chân vẫn hơi lệch; muốn hết cần huấn luyện lại chính sách trên mô hình này.
- Lon súp đối xứng tròn: góc xoay quanh trục lon không xác định được bằng hình học.
