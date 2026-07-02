# TrustFed — Mô hình Ngôn ngữ Lớn Liên kết, Đáng tin cậy

[🇬🇧 English](README.md) · **🇻🇳 Tiếng Việt**

Một *portfolio + research proposal* để xin vị trí **Trợ lý Nghiên cứu (RA)** về đề
tài **tinh chỉnh mô hình ngôn ngữ theo kiểu liên kết, bảo vệ quyền riêng tư**. Dự án
được xây theo hai tầng, đi từ *"tôi hiểu học liên kết"* đến *"tôi đã chạy được một
pipeline thích ứng LLM liên kết chạm tới cả ba trụ cột của đề tài"*.

- **Tầm nhìn nghiên cứu:** [`PROPOSAL.md`](PROPOSAL.md) — proposal 2 trang cho
  *TrustFed*: các F-LLM riêng tư ngay từ thiết kế, học được từ dữ liệu không nhãn,
  và chuyển giao tri thức hiệu quả giữa mô hình global lớn và mô hình local nhỏ.
- **Lời chào hàng:** [`EMAIL_TO_PROF.md`](EMAIL_TO_PROF.md) — mẫu email ngắn xin gặp/xin RA.
- **Bằng chứng:** hai tầng code bên dưới và [`RESULTS.md`](RESULTS.md).
- **Hiểu sâu (tiếng Việt):** [`HUONG_DAN.md`](HUONG_DAN.md) — giải thích cặn kẽ từng
  khái niệm và từng phần code.

---

## Tầng 0 — FedAvg từ nguyên lý cơ bản (MNIST)

Một demo Học Liên kết tối giản để nắm chắc cơ chế client/server.

- [`centralized.py`](centralized.py) — train một CNN nhỏ theo cách thường (mốc chuẩn).
- [`federated.py`](federated.py) — chia MNIST cho 5 client ảo và train bằng
  **FedAvg** qua công cụ mô phỏng của thư viện Flower; chỉ chia sẻ *cập nhật mô hình*.

```bash
python centralized.py    # độ chính xác mốc chuẩn
python federated.py      # bản liên kết với FedAvg
```

**Điều rút ra:** FedAvg tổng hợp cập nhật client *có trọng số theo lượng dữ liệu*
như thế nào; vì sao dữ liệu *non-IID* và chi phí truyền thông khiến FL khó hơn train
tập trung; và vì sao *"không gửi dữ liệu thô" ≠ riêng tư* — cập nhật vẫn rò rỉ, điều
này dẫn động cho Tầng 1 và proposal.

## Tầng 1 — Federated LoRA trên một LLM đã pretrain (demo đinh)

[`fed_lora.py`](fed_lora.py) là hiện vật bắc cầu từ Tầng 0 sang proposal. Nó tinh
chỉnh một **BERT-tiny đã pretrain, được đóng băng** bằng **LoRA (tự viết từ đầu,
không dùng `peft`)** trên các **client non-IID**, và thể hiện cả ba trụ cột của
TrustFed trong một script chạy vài phút trên CPU:

| Trụ cột | Demo thể hiện thế nào |
|---|---|
| **Riêng tư từ thiết kế** | chỉ truyền các adapter LoRA tí hon (~500× nhỏ hơn cập nhật cả mô hình); tùy chọn **DP-FedAvg mức người-dùng** (clip + nhiễu Gauss) làm "núm vặn" riêng tư hình thức |
| **Chuyển giao tri thức hiệu quả** | FedAvg trên adapter truyền tri thức giữa các client; ta *đo* được **mức tăng của federated so với local-only** dưới lệch miền |
| **Tính thực dụng** | mô hình nền đóng băng + adapter tiết kiệm tham số + vòng lặp liên kết tự viết, minh bạch (không Ray), chạy ổn trên Windows |

```bash
pip install -r requirements.txt
python fed_lora.py            # so sánh đầy đủ trên ag_news (lần đầu tải BERT-tiny)
python fed_lora.py --quick    # bản nhỏ, chạy nhanh để thử
python fed_lora.py --dp-noise 0.05   # tăng cường differential privacy
```

Script chạy bốn chế độ — **centralized** (trần), **local-only** (không hợp tác),
**federated LoRA**, và **federated LoRA + DP** — rồi ghi bảng kết quả vào
[`RESULTS.md`](RESULTS.md) (và `results.png` nếu đã cài `matplotlib`). Xem proposal
để biết chúng khớp với các câu hỏi nghiên cứu ra sao.

### Điều gì khiến đây là một prototype F-LLM đáng tin (không phải đồ chơi)
- **Backbone pretrain thật**: LoRA chỉ có nghĩa khi đặt trên một mô hình đã pretrain,
  nên ta đóng băng trọng số BERT-tiny thật — việc trung bình adapter trên một backbone
  *dùng chung* chính là điều kiện đúng đắn tinh vi mà đa số demo ngây thơ làm sai.
- **Non-IID có chủ đích**: phân hoạch Dirichlet cho mỗi client một "khẩu phần miền"
  khác nhau — đúng bối cảnh F-LLM thực tế nơi FL ngây thơ hay đuối.
- **Thước đo trung thực**: ta báo cáo *cái giá riêng tư* của DP và *chi phí truyền
  thông* một cách tường minh, chứ không chỉ một con số độ chính xác.

## Bản đồ repo

| File | Là gì |
|---|---|
| [`PROPOSAL.md`](PROPOSAL.md) | Research proposal (đọc trước tiên) |
| [`HUONG_DAN.md`](HUONG_DAN.md) | Hướng dẫn tiếng Việt chuyên sâu toàn dự án (để bạn học) |
| [`fed_lora.py`](fed_lora.py) | **Demo đinh**: federated LoRA trên một LLM pretrain |
| [`federated.py`](federated.py) / [`centralized.py`](centralized.py) | Tầng 0: FedAvg vs tập trung trên MNIST |
| [`RESULTS.md`](RESULTS.md) | Kết quả tự sinh từ `fed_lora.py` |
| [`EMAIL_TO_PROF.md`](EMAIL_TO_PROF.md) | Mẫu email xin làm RA |
| [`requirements.txt`](requirements.txt) | Thư viện phụ thuộc (Tầng 0 + Tầng 1) |

## Lộ trình (theo proposal)
Nâng backbone (DistilBERT/GPT-2) → vẽ *biên đánh đổi riêng tư–độ chính xác* của DP →
thích ứng **tự giám sát** (không nhãn) theo kiểu liên kết → **chưng cất hai chiều**
cho bối cảnh global-lớn ↔ local-nhỏ.
