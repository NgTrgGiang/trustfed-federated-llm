# TrustFed — Hướng dẫn toàn diện (tiếng Việt)

> File này giúp bạn **hiểu tường tận** mọi thứ trong dự án: từ ý tưởng nghiên cứu,
> các khái niệm cốt lõi (Federated Learning, LoRA, Differential Privacy…), cho tới
> từng dòng logic trong code và cách đọc kết quả. Đọc xong file này bạn có thể tự
> tin trình bày dự án với giáo sư và trả lời câu hỏi phản biện.
>
> (File `README.md` là bản tiếng Anh, ngắn gọn, để **gửi cho giáo sư**. File này là
> để **cho bạn học**.)

---

## Mục lục

1. [Dự án này là gì và để làm gì](#1-dự-án-này-là-gì-và-để-làm-gì)
2. [Bức tranh lớn: vì sao đề tài quan trọng](#2-bức-tranh-lớn-vì-sao-đề-tài-quan-trọng)
3. [Cấu trúc repo — file nào làm gì](#3-cấu-trúc-repo--file-nào-làm-gì)
4. [Các khái niệm cốt lõi (giải thích từ đầu)](#4-các-khái-niệm-cốt-lõi-giải-thích-từ-đầu)
5. [Stage 0 — FedAvg trên MNIST](#5-stage-0--fedavg-trên-mnist)
6. [Stage 1 — Federated LoRA trên LLM (demo đinh)](#6-stage-1--federated-lora-trên-llm-demo-đinh)
7. [Đọc hiểu kết quả](#7-đọc-hiểu-kết-quả)
8. [Hai lỗi tinh vi đã sửa (rất đáng kể khi phỏng vấn)](#8-hai-lỗi-tinh-vi-đã-sửa)
9. [Cách chạy & các tham số](#9-cách-chạy--các-tham-số)
10. [Giáo sư có thể hỏi gì — và cách trả lời](#10-giáo-sư-có-thể-hỏi-gì--và-cách-trả-lời)
11. [Hạn chế trung thực & hướng mở rộng](#11-hạn-chế-trung-thực--hướng-mở-rộng)
12. [Bảng thuật ngữ](#12-bảng-thuật-ngữ)

---

## 1. Dự án này là gì và để làm gì

**Mục tiêu thực dụng:** tạo một bộ hồ sơ đủ mạnh để **thuyết phục giáo sư nhận bạn
làm Trợ lý Nghiên cứu (RA)** cho đề tài *TrustFed: Trustworthy Federated Large
Language Models* (Mô hình Ngôn ngữ Lớn Liên kết, Đáng tin cậy).

Hồ sơ gồm 2 phần bổ trợ nhau:

- **Phần nói** — một *research proposal* (`PROPOSAL.md`) trình bày bạn hiểu vấn đề,
  biết khoảng trống nghiên cứu, và có kế hoạch.
- **Phần làm** — một *demo chạy được* (`fed_lora.py`) chứng minh bạn **đã bắt tay
  code** đúng bài toán, không chỉ nói lý thuyết. Đây là thứ thuyết phục nhất với
  một giáo sư đang tuyển RA: *"em không chỉ thấy hay, em đã tự dựng vòng lặp lõi
  của bài toán rồi"*.

**Đề tài TrustFed** xoay quanh 3 trụ cột (chính là 3 vế trong đề bài gốc):

| Trụ cột | Nghĩa là gì |
|---|---|
| **Privacy-by-design** | Bảo mật ngay từ thiết kế: dữ liệu thô không rời khỏi máy client; hơn nữa còn có bảo đảm *hình thức* (differential privacy) chứ không chỉ "không gửi data" |
| **Học không giám sát hợp pháp** | Học từ dữ liệu **không nhãn** (self-supervised), theo cách tuân thủ quy định pháp lý về dữ liệu |
| **Chuyển giao tri thức hiệu quả** | Truyền tri thức **hai chiều** giữa mô hình global lớn và các mô hình local nhỏ, dưới ngân sách truyền thông eo hẹp |

Bạn đã chọn: bối cảnh **RA**, demo đinh là **Federated LoRA fine-tune một LM nhỏ**,
và proposal **cân bằng cả 3 trụ cột**. Toàn bộ tài liệu được viết theo đúng lựa chọn đó.

---

## 2. Bức tranh lớn: vì sao đề tài quan trọng

**Nghịch lý dữ liệu.** Các LLM được huấn luyện trên dữ liệu tập trung khổng lồ.
Nhưng dữ liệu *giá trị nhất* để tinh chỉnh chúng cho việc thực tế — bệnh án, hồ sơ
tài chính, văn bản pháp lý, log doanh nghiệp, tin nhắn trên điện thoại — lại chính
là dữ liệu **không thể gom về một chỗ**, vì kỳ vọng riêng tư và luật (GDPR ở EU,
HIPAA ở Mỹ, Nghị định 13/2023 PDPD ở Việt Nam, EU AI Act…). Ai có data quý nhất thì
lại ít có khả năng dùng nó nhất.

**Federated Learning (Học liên kết)** là lối thoát: nhiều bên cùng huấn luyện một
mô hình mà **dữ liệu thô không bao giờ rời khỏi máy**; chỉ có *cập nhật mô hình*
được chia sẻ và tổng hợp. Nhưng đưa ý tưởng này từ bộ phân loại nhỏ lên **LLM** thì
vấp 3 rào cản — trùng khớp đúng 3 trụ cột ở trên:

1. **Riêng tư không miễn phí.** "Không gửi data thô" ≠ "riêng tư". Bản thân *cập
   nhật mô hình* (gradient, trọng số, thậm chí adapter) vẫn **rò rỉ thông tin**, có
   thể bị tấn công suy luận thành viên (membership inference) hay tái dựng dữ liệu.
   → Cần bảo đảm hình thức (DP), không chỉ locality.
2. **Nhãn khan hiếm và nhạy cảm pháp lý.** Trong lĩnh vực bị quản lý, dữ liệu có
   nhãn vừa đắt vừa bị hạn chế sử dụng. → Cần học từ text **không nhãn**
   (self-supervised) một cách hợp pháp.
3. **Client dị biệt và nhỏ.** Một bệnh viện không thể chạy mô hình 70 tỷ tham số;
   điện thoại không chạy nổi 7 tỷ. → Cần **chuyển giao tri thức hiệu quả** giữa mô
   hình global lớn và mô hình local nhỏ, tiết kiệm băng thông.

Các "viên gạch" để giải quyết đều đã có (FedAvg, LoRA, DP, distillation) nhưng
**chưa được hợp nhất** cho bối cảnh F-LLM tiết kiệm tham số. TrustFed nhắm đúng
khoảng trống đó.

---

## 3. Cấu trúc repo — file nào làm gì

```
FL/
├── PROPOSAL.md        # Research proposal 2 trang (gửi giáo sư) — ĐỌC TRƯỚC
├── README.md          # Giới thiệu portfolio bằng tiếng Anh (cho giáo sư)
├── HUONG_DAN.md       # ← File này: giải thích cặn kẽ bằng tiếng Việt (cho bạn)
├── EMAIL_TO_PROF.md   # Mẫu email pitch ngắn để xin gặp/xin RA
├── RESULTS.md         # Bảng kết quả TỰ SINH mỗi lần chạy fed_lora.py
│
├── fed_lora.py        # ★ DEMO ĐINH: Federated LoRA trên BERT-tiny pretrained
│
├── federated.py       # Stage 0: FedAvg trên MNIST (dùng Flower)
├── centralized.py     # Stage 0: baseline huấn luyện tập trung (để so sánh)
│
├── requirements.txt   # Thư viện cần cài
└── results.png        # Biểu đồ (tự sinh nếu có matplotlib)
```

**Mạch câu chuyện portfolio** đi từ đơn giản → chuyên sâu:

> Stage 0 (*"em hiểu FedAvg"*) → Stage 1 (*"em chạy được Federated LoRA trên một LLM
> thật, chạm cả 3 trụ cột"*) → Proposal (*"và đây là kế hoạch nghiên cứu 12 tháng"*).

---

## 4. Các khái niệm cốt lõi (giải thích từ đầu)

### 4.1. Federated Learning (FL) & FedAvg

**Ý tưởng:** thay vì gom data về server rồi train, ta để **data ở nguyên tại client**.
Mỗi vòng (round):

1. Server gửi mô hình hiện tại xuống các client.
2. Mỗi client train **cục bộ** trên dữ liệu riêng của mình vài bước.
3. Client gửi **trọng số đã cập nhật** (không phải data) về server.
4. Server **tính trung bình** các trọng số → mô hình global mới. Lặp lại.

Thuật toán trung bình đó gọi là **FedAvg** (McMahan và cộng sự, 2017). Trung bình có
**trọng số theo lượng dữ liệu**: client nào nhiều data thì đóng góp nhiều hơn.

Công thức (đơn giản hóa): với `n_k` là số mẫu của client `k`, `w_k` là trọng số của nó:

```
w_global = Σ_k (n_k / Σ n) · w_k
```

### 4.2. Non-IID — vì sao FL khó

IID = "independent and identically distributed" (độc lập, cùng phân phối). Trong đời
thực, mỗi client có phân phối dữ liệu **rất khác nhau** (non-IID): bệnh viện A chuyên
tim mạch, bệnh viện B chuyên nhi… Khi đó mỗi client kéo mô hình về "miền" của riêng
nó (*client drift*), và trung bình các mô hình lệch nhau có thể **hội tụ kém**.

Trong demo, ta mô phỏng non-IID bằng **phân hoạch Dirichlet**: mỗi client bị *lệch*
về vài lớp/"miền" chủ đề. Tham số `alpha` càng **nhỏ** thì càng lệch (non-IID mạnh).

### 4.3. LoRA — Low-Rank Adaptation (tinh chỉnh tiết kiệm tham số)

Fine-tune toàn bộ một LLM = cập nhật hàng triệu/tỷ trọng số → tốn bộ nhớ, tốn băng
thông (nếu gửi qua mạng trong FL), và dễ overfit khi data ít.

**LoRA** (Hu và cộng sự, 2021) giải quyết bằng cách: **đóng băng** toàn bộ mô hình
gốc `W₀`, chỉ **thêm một cập nhật hạng thấp** nhỏ xíu bên cạnh:

```
y = W₀·x  +  (α/r)·B·A·x
```

- `W₀`: trọng số gốc (đóng băng, không train).
- `A` (kích thước `r × d_in`) và `B` (`d_out × r`): hai ma trận **nhỏ** cần train.
- `r` (rank) rất nhỏ (ví dụ 8) → `A`, `B` có rất ít tham số.
- `B` được **khởi tạo bằng 0** → lúc đầu adapter = 0 (không làm hỏng mô hình gốc),
  rồi học dần.

**Vì sao LoRA cực hợp với FL?** Vì chỉ cần gửi `A`, `B` (bé tí) qua mạng thay vì cả
mô hình → **tiết kiệm truyền thông ~100–500 lần**, và **giảm bề mặt tấn công** riêng
tư (kẻ tấn công thấy ít thông tin hơn). Đây là cầu nối giữa 3 trụ cột.

### 4.4. Vì sao backbone phải là mô hình ĐÃ pretrain và ĐÓNG BĂNG

LoRA chỉ *điều chỉnh nhẹ* một mô hình vốn đã "biết ngôn ngữ". Nếu backbone là ngẫu
nhiên (chưa học gì), thì việc điều chỉnh hạng thấp lên một nền vô nghĩa → không học
được gì mấy. Vì thế demo dùng **BERT-tiny** — một transformer nhỏ (~4,4 triệu tham
số) **đã được pretrain** (học Masked Language Modeling trên Wikipedia/sách) — rồi
**đóng băng** nó, chỉ train LoRA + đầu phân loại. (Xem [mục 8](#8-hai-lỗi-tinh-vi-đã-sửa)
để hiểu vì sao chi tiết "đóng băng + dùng chung backbone" lại là điểm mấu chốt.)

### 4.5. Differential Privacy (DP) & DP-FedAvg

Ngay cả khi chỉ gửi adapter, thông tin vẫn có thể rò. **Differential Privacy** cho
một bảo đảm *toán học*: kết quả gần như **không đổi** dù thêm/bớt dữ liệu của một cá
nhân/client → không thể suy ra ai đó có trong tập huấn luyện.

**DP-FedAvg mức người-dùng** (McMahan và cộng sự, 2018) làm 2 việc trước khi tổng hợp:

1. **Clip (cắt chuẩn):** giới hạn độ lớn update của mỗi client về ngưỡng `C` (chuẩn
   L2 ≤ C). → Không client nào ảnh hưởng quá mạnh.
2. **Thêm nhiễu Gauss:** cộng nhiễu ngẫu nhiên chuẩn vào bản tổng hợp. Độ lệch chuẩn
   nhiễu = `σ·C/N` (chia cho `N` client vì độ nhạy của **trung bình** N update đã
   clip là `C/N`). `σ` càng lớn → riêng tư càng mạnh nhưng độ chính xác giảm.

Đây chính là **đánh đổi privacy–utility**: bạn "mua" riêng tư bằng một ít độ chính xác.

### 4.6. Knowledge Transfer & Distillation

- **FedAvg trên adapter** đã là một dạng chuyển giao tri thức: tri thức của nhiều
  client local được **gộp** vào một adapter global, rồi phát ngược lại cho mọi
  client. Demo cho thấy điều này **giúp chính các client yếu nhất** (xem lift ở
  [mục 7](#7-đọc-hiểu-kết-quả)).
- **Knowledge Distillation (chưng cất):** mô hình lớn ("thầy") dạy mô hình nhỏ
  ("trò") bằng cách cho trò bắt chước *đầu ra* của thầy. Trong TrustFed, đây là cách
  để mô hình global lớn dạy các mô hình local nhỏ (global→local), và là hướng mở
  rộng trong proposal (RQ3).

---

## 5. Stage 0 — FedAvg trên MNIST

Đây là bài "khởi động" để nắm cơ chế FL, dùng bộ ảnh chữ số MNIST.

- **`centralized.py`** — train một CNN nhỏ theo kiểu thường (gom hết data). Đây là
  **mốc chuẩn (baseline)** về độ chính xác để so sánh.
- **`federated.py`** — chia MNIST cho **5 client ảo**, train bằng **FedAvg** qua công
  cụ mô phỏng của thư viện **Flower**. Mỗi vòng: client train cục bộ, server trung
  bình trọng số.

**Ba bài học rút ra (đã ghi trong code):**
- FedAvg tổng hợp update **có trọng số theo lượng data**.
- FL khó hơn train tập trung vì **chi phí truyền thông** và **non-IID**.
- **"Không gửi data thô" ≠ riêng tư** — update vẫn rò → dẫn tới Stage 1 và proposal.

> Lưu ý: Flower dùng Ray để mô phỏng, đôi khi "khó tính" trên Windows. Vì vậy ở
> Stage 1 (demo đinh) mình **tự viết vòng lặp FL** để chủ động và chạy ổn định.

---

## 6. Stage 1 — Federated LoRA trên LLM (demo đinh)

Toàn bộ nằm trong **`fed_lora.py`**, chạy được trong vài phút trên CPU, chạm cả 3 trụ
cột trong **một file duy nhất**. Dưới đây là "tua" qua từng phần của file.

### 6.1. Dữ liệu (Phần 1 trong code)

- Dùng bộ **ag_news** thật: tin tức 4 miền — *World, Sports, Business, Sci/Tech*.
  Bốn miền này rất hợp để mô phỏng "mỗi client một miền" (non-IID).
- Nếu không có mạng, tự động chuyển sang **bộ synthetic đa-miền** để demo *luôn chạy
  được* (kỹ thuật phòng hờ mà người review sẽ đánh giá cao).
- `dirichlet_partition(...)`: chia dữ liệu cho các client theo phân phối Dirichlet
  (non-IID). In ra "domain mix" của từng client để bạn thấy rõ độ lệch.

### 6.2. Tokenizer

Dùng **WordPiece tokenizer** thật của BERT (`bert-base-uncased`) — biến câu chữ thành
chuỗi id token mà mô hình hiểu. (BERT-tiny dùng chung bộ từ vựng này.)

### 6.3. Mô hình + LoRA tự viết (Phần 2 trong code)

- **`LoRALinear`** — lớp tự cài đặt công thức LoRA ở [mục 4.3](#43-lora--low-rank-adaptation-tinh-chỉnh-tiết-kiệm-tham-số):
  bọc một `Linear` gốc (đóng băng) + hai ma trận nhỏ `A`, `B` (train). **Tự viết,
  không dùng thư viện `peft`** → chứng minh bạn hiểu nội tại, không chỉ gọi API.
- **`inject_lora(...)`** — chèn LoRA vào đúng các lớp `query` và `value` của attention
  (đây là vị trí kinh điển hay gắn LoRA).
- **`LoRABert`** — nạp **BERT-tiny pretrained**, **đóng băng toàn bộ**, chèn LoRA vào
  Q/V, thêm một **đầu phân loại** (`head`) train được. Chỉ LoRA + head là train và
  **là thứ duy nhất được truyền đi** trong FL.

### 6.4. Trạng thái adapter, FedAvg & DP (Phần 3 trong code)

- `adapter_state(model)` — lấy **chỉ các tensor train được** (LoRA + head) = đúng gói
  tin sẽ truyền qua mạng.
- `local_train`, `evaluate` — train cục bộ / đánh giá.
- `fedavg_deltas(...)` — tổng hợp **độ chênh (delta)** của các client theo trọng số
  dữ liệu. Nếu bật DP thì gọi sang `_apply_user_level_dp`.
- `_apply_user_level_dp(...)` — **clip** chuẩn update mỗi client về `C`, rồi cộng
  **nhiễu Gauss** với độ lệch chuẩn `σ·C/N` (xem [mục 4.5](#45-differential-privacy-dp--dp-fedavg)).

### 6.5. Bốn "chế độ" thí nghiệm (Phần 4 trong code)

Script chạy 4 kịch bản để so sánh sòng phẳng:

| Chế độ | Ý nghĩa |
|---|---|
| **Centralized** | Gom hết data lại train (vi phạm riêng tư) → **trần** độ chính xác |
| **Local-only** | Mỗi client tự train một mình, **không** hợp tác → baseline "không chuyển giao tri thức" (non-IID làm hại nặng nhất) |
| **Federated LoRA** | FedAvg trên adapter — **data ở nguyên client** |
| **Federated LoRA + DP** | Như trên, thêm nhiễu DP → thấy **cái giá của riêng tư** |

Cuối cùng script in bảng tóm tắt + ghi `RESULTS.md` (và `results.png` nếu có matplotlib).

---

## 7. Đọc hiểu kết quả

Kết quả tiêu biểu (ag_news, 5 client, non-IID Dirichlet α=0.3, 6 vòng — số cụ thể xem
`RESULTS.md`):

| Chế độ | Độ chính xác | Đọc thế nào |
|---|---|---|
| Centralized (trần) | **0.862** | Nếu được gom hết data thì đạt tối đa cỡ này |
| Local-only (không hợp tác) | **0.560** | Mỗi client tự bơi → non-IID làm hại, thấp |
| **Federated LoRA** | **0.721** | **Hợp tác mà data không rời máy** → cao hơn hẳn local-only |
| Federated + DP (C=1.0, σ=0.02) | **0.636** | Thêm riêng tư hình thức → chỉ mất ~8.5 điểm, vẫn hội tụ |

**Ba con số "biết nói" (insight chính):**

1. **Lift chuyển giao tri thức = 0.721 − 0.560 = +0.161.** Đây là *bằng chứng định
   lượng* cho trụ cột #3: dưới non-IID, **hợp tác giúp chính các client yếu**, mà
   **dữ liệu thô không hề di chuyển**. Đây là câu chuyện đắt giá nhất của demo.
2. **Đường cong hội tụ** (cột "per round" trong RESULTS.md): federated leo đều
   0.44 → 0.72 qua 6 vòng → cho thấy FedAvg thực sự đang tổng hợp tri thức, không
   phải ăn may.
3. **Payload = 8.708 tham số/vòng** so với **4.394.628** của cả mô hình → **~505×
   nhỏ hơn**. Vừa tiết kiệm băng thông, vừa thu nhỏ bề mặt tấn công privacy → chạm
   trụ cột #1 và #3 cùng lúc.

**Cái giá của DP:** khi bật nhiễu, độ chính xác tụt từ 0.721 xuống **0.636** — chỉ
mất ~8.5 điểm và **vẫn hội tụ** (leo 0.56 → 0.64 qua các vòng). Đó **đúng như kỳ
vọng** và chính là "đánh đổi privacy–utility" mà đề tài muốn định lượng (RQ1): tăng
`σ` thì riêng tư mạnh hơn nhưng mất nhiều độ chính xác hơn. Với ít client (chỉ 5), DP
đặc biệt "đắt" vì nhiễu khó được san đều; thực tế các hệ thống dùng *hàng nghìn*
client để pha loãng nhiễu — đây là điểm bạn nên nêu thẳng thắn và biến thành hướng
nghiên cứu.

---

## 8. Hai lỗi tinh vi đã sửa

> Đây là phần **cực kỳ giá trị khi phỏng vấn**: nó cho thấy bạn không chỉ ghép API,
> mà thật sự *hiểu* và *gỡ được* những cái bẫy mà người mới hay mắc.

### Lỗi 1 — Adapter chỉ có nghĩa khi mọi client dùng CHUNG một backbone pretrained

Ban đầu, mỗi lần tạo mô hình lại khởi tạo một backbone **ngẫu nhiên khác nhau**. Khi
đó LoRA train trên backbone của client A **vô nghĩa** nếu đem ghép vào backbone khác
của global → trung bình adapter ra "rác", độ chính xác federated ≈ ngẫu nhiên (~0.25).

**Bản chất:** LoRA là "điều chỉnh *quanh* một điểm gốc `W₀`". Nếu `W₀` mỗi nơi một
khác, các điều chỉnh không cộng lại được. **Cách sửa:** mọi client nạp **cùng một
bộ trọng số BERT-tiny pretrained** rồi đóng băng → mọi adapter được tổng hợp trên
*cùng một nền*. Sau khi sửa, federated vượt hẳn local-only như mong đợi.

### Lỗi 2 — Nhiễu DP phải chia cho số client (`/N`)

Ban đầu cộng nhiễu độ lớn `σ·C` thẳng vào **trung bình** các update. Vì adapter có
8.708 chiều, chuẩn L2 của nhiễu ≈ `σ·√8708 ≈ σ·93` — **lấn át** tín hiệu (chỉ ~0.7)
→ DP không chỉ giảm chính xác mà còn **phân kỳ** (càng train càng tệ).

**Bản chất:** độ nhạy của **trung bình** N update đã clip là `C/N`, nên độ lệch chuẩn
nhiễu đúng phải là `σ·C/N`. **Cách sửa:** chia nhiễu cho `N` và chọn `σ` hợp lý → DP
giảm chính xác một cách *mượt mà* (đánh đổi thật), không sập.

> Mình còn *đo thực nghiệm* chuẩn update mỗi client (~0.62–1.0) để chọn ngưỡng clip
> `C` và `σ` cho đúng — đó là cách làm nghiên cứu tử tế: hiệu chỉnh dựa trên số đo,
> không phải đoán.

---

## 9. Cách chạy & các tham số

**Cài thư viện** (Stage 1 chỉ cần `torch`, `transformers`, `datasets` — máy bạn đã có):

```bash
pip install -r requirements.txt
```

**Chạy demo đinh:**

```bash
python fed_lora.py            # bản đầy đủ (tự tải BERT-tiny ~17MB lần đầu, rồi cache)
python fed_lora.py --quick    # bản nhỏ, nhanh để thử
python fed_lora.py --dp-noise 0.05   # tăng nhiễu DP để thấy riêng tư mạnh hơn
python fed_lora.py --rounds 10 --alpha 0.1   # nhiều vòng hơn, non-IID mạnh hơn
```

**Các tham số hữu ích** (xem thêm `python fed_lora.py --help`):

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `--clients` | Số client | 5 |
| `--rounds` | Số vòng liên kết | 6 |
| `--local-epochs` | Số epoch train cục bộ mỗi vòng | 2 |
| `--alpha` | Độ non-IID Dirichlet (nhỏ = lệch mạnh) | 0.3 |
| `--rank` | Hạng `r` của LoRA | 8 |
| `--dp-clip` / `--dp-noise` | Ngưỡng clip `C` / hệ số nhiễu `σ` của DP | 1.0 / 0.02 |
| `--n-train` / `--n-test` | Cỡ tập train/test lấy từ ag_news | 4000 / 2000 |
| `--quick` | Chế độ nhỏ + nhanh | tắt |

**Chạy Stage 0** (tùy chọn, cần cài thêm `torchvision`, `flwr[simulation]`):

```bash
python centralized.py    # baseline
python federated.py      # FedAvg
```

---

## 10. Giáo sư có thể hỏi gì — và cách trả lời

- **"Vì sao đóng băng backbone mà vẫn học được?"** → Vì BERT-tiny đã pretrain nên đặc
  trưng ngôn ngữ ở lớp ẩn đã hữu ích; LoRA + head chỉ cần *xoay* các đặc trưng đó cho
  bài phân loại. Đó là toàn bộ tinh thần PEFT.
- **"Federated 0.72 vẫn thua Centralized 0.86 mà?"** → Đúng, và đó là *khoảng cách
  do non-IID + ngân sách vòng hữu hạn* — chính là thứ đề tài muốn thu hẹp (aggregator
  nhận biết dị biệt, nhiều vòng hơn, distillation). Điểm mấu chốt là federated **cao
  hơn local-only** trong khi **không lộ data**.
- **"DP làm tụt chính xác nhiều thế?"** → Với chỉ 5 client, nhiễu khó pha loãng; hệ
  thống thật dùng hàng nghìn client. Định lượng *biên privacy–utility* theo số client,
  rank, `C`, `σ` chính là RQ1 của em.
- **"Đây mới là phân loại, chứ đâu phải LLM sinh văn bản?"** → Đúng, em chọn phân loại
  để có thước đo sạch trên CPU. Phương pháp *độc lập với checkpoint*: đổi sang
  DistilBERT/GPT-2 và mục tiêu sinh/MLM chỉ là thay vài dòng — đó là bước đầu của
  lộ trình.
- **"Vì sao không dùng `peft`/Flower?"** → Em *cố ý* tự viết LoRA và vòng lặp FL để
  (a) hiểu nội tại, (b) chủ động cài DP + tổng hợp adapter, (c) chạy ổn định trên
  Windows. Stage 0 đã cho thấy em dùng được Flower.

---

## 11. Hạn chế trung thực & hướng mở rộng

**Hạn chế (nên chủ động nêu — sự trung thực tạo uy tín):**
- Backbone rất nhỏ (BERT-tiny) và bài toán là phân loại 4 lớp; chưa phải LLM sinh văn.
- Chạy *mô phỏng* trên một máy, chưa phải hệ phân tán thật.
- DP theo kiểu minh họa cơ chế; chưa kèm *kế toán (ε, δ)* chính thức.
- Số client nhỏ (5) khiến DP đắt.

**Lộ trình mở rộng (khớp proposal):**
1. Nâng backbone lên DistilBERT/GPT-2; thêm baseline nhận biết dị biệt.
2. **RQ1:** DP + secure aggregation trên adapter; vẽ *biên privacy–utility* + kế toán (ε, δ).
3. **RQ2:** học **self-supervised** (MLM/CLM) trên text **không nhãn** trong FL.
4. **RQ3:** **chưng cất hai chiều** global↔local dưới non-IID mạnh.

---

## 12. Bảng thuật ngữ

| Thuật ngữ | Giải nghĩa nhanh |
|---|---|
| **F-LLM** | Federated Large Language Model — LLM huấn luyện/tinh chỉnh kiểu liên kết |
| **FedAvg** | Thuật toán trung bình có trọng số các cập nhật client |
| **Non-IID** | Dữ liệu các client phân phối khác nhau (thực tế, gây khó cho FL) |
| **Dirichlet(α)** | Cách chia dữ liệu tạo độ lệch non-IID; α nhỏ = lệch mạnh |
| **LoRA** | Low-Rank Adaptation — tinh chỉnh bằng hai ma trận nhỏ, đóng băng mô hình gốc |
| **PEFT** | Parameter-Efficient Fine-Tuning — họ phương pháp tinh chỉnh tiết kiệm tham số (LoRA thuộc nhóm này) |
| **Adapter** | Phần trọng số nhỏ được thêm/ train (ở đây = LoRA + head) |
| **Backbone** | Mô hình nền (BERT-tiny) — được đóng băng |
| **DP** | Differential Privacy — bảo đảm riêng tư hình thức bằng clip + nhiễu |
| **Clip (C)** | Cắt chuẩn L2 của update về ≤ C để giới hạn ảnh hưởng một client |
| **σ (noise)** | Hệ số nhiễu DP; lớn hơn = riêng tư mạnh hơn, chính xác thấp hơn |
| **Distillation** | Chưng cất tri thức: mô hình lớn "dạy" mô hình nhỏ bắt chước đầu ra |
| **Self-supervised** | Học từ dữ liệu **không nhãn** (vd. đoán từ bị che — MLM) |
| **MLM / CLM** | Masked / Causal Language Modeling — hai mục tiêu tự giám sát kinh điển |

---

*Tài liệu này đi kèm `PROPOSAL.md` (bản nghiên cứu) và `fed_lora.py` (mã nguồn demo).
Có gì chưa rõ, cứ mở đúng phần code tương ứng theo [mục 6](#6-stage-1--federated-lora-trên-llm-demo-đinh) để đối chiếu.*
