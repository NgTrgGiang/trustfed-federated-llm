# TrustFed: Mô hình Ngôn ngữ Lớn Liên kết, Đáng tin cậy

**Đề cương nghiên cứu / thư bày tỏ nguyện vọng cho vị trí Trợ lý Nghiên cứu (RA)**

*Người chuẩn bị: [Tên của bạn], VinUniversity — [ngày]*

[🇬🇧 English](PROPOSAL.md) · **🇻🇳 Tiếng Việt**

---

## 1. Vấn đề và động lực

Các Mô hình Ngôn ngữ Lớn (LLM) được huấn luyện trên những kho dữ liệu tập trung ngày
càng lớn. Thế nhưng dữ liệu khiến chúng *hữu ích nhất* trong thực tế — bệnh án, hồ sơ
tài chính, văn bản pháp lý, log doanh nghiệp, văn bản người dùng trên thiết bị — lại
chính là dữ liệu **không thể tập trung hóa**, vì kỳ vọng riêng tư và các quy định ràng
buộc (GDPR, HIPAA, Nghị định bảo vệ dữ liệu cá nhân PDPD 2023 của Việt Nam, EU AI
Act). Hệ quả là một khoảng cách ngày càng lớn: tổ chức nào có dữ liệu chuyên ngành giá
trị nhất thì lại *ít có khả năng dùng nó nhất* để thích ứng các mô hình nền.

**Học Liên kết (Federated Learning – FL)** mở một lối ra: nhiều bên cùng huấn luyện một
mô hình trong khi dữ liệu thô không bao giờ rời khỏi cơ sở của họ; chỉ *cập nhật mô
hình* được chia sẻ. Nhưng đi từ các bộ phân loại nhỏ lên *LLM Liên kết (F-LLM)* không
phải là một sự thay thế đơn giản. Có ba rào cản, và chúng ánh xạ trực tiếp lên ba mục
tiêu của dự án này:

1. **Riêng tư không miễn phí.** "Không chia sẻ dữ liệu thô" không đồng nghĩa với riêng
   tư — cập nhật mô hình (gradient, trọng số, thậm chí adapter) vẫn rò rỉ thông tin và
   dễ bị tấn công suy luận thành viên (membership inference) và tái dựng dữ liệu. F-LLM
   cần *riêng tư từ thiết kế*, với bảo đảm hình thức, không chỉ là "dữ liệu ở tại chỗ".
2. **Nhãn khan hiếm và nhạy cảm pháp lý.** Trong các lĩnh vực bị quản lý, dữ liệu có
   nhãn vừa đắt vừa thường bị hạn chế sử dụng. F-LLM thực dụng phải học từ văn bản
   **không nhãn** qua tự giám sát (self-supervision), theo cách *tuân thủ pháp lý*.
3. **Client dị biệt và nhỏ.** Một bệnh viện không thể chứa mô hình 70 tỷ tham số; một
   chiếc điện thoại không chạy nổi 7 tỷ. Ta cần **chuyển giao tri thức hiệu quả** giữa
   một mô hình global lớn và các mô hình local nhỏ đã thích ứng theo miền — theo *cả
   hai chiều* — dưới một ngân sách truyền thông eo hẹp.

**Mục tiêu.** Dự án này (tên tạm *TrustFed*) hướng tới phát triển các phương pháp giúp
F-LLM **thực dụng và đáng tin cậy** đúng trong những bối cảnh đó: riêng tư từ thiết kế,
học được mà không cần nhãn, và đủ hiệu quả để chạy trên các client dị biệt, hạn chế tài
nguyên.

## 2. Vì sao là lúc này, và khoảng trống nằm ở đâu

Các "viên gạch" đã có sẵn nhưng chưa được hợp nhất:

- **FedAvg** (McMahan và cộng sự, 2017) khiến huấn luyện hợp tác trở nên khả thi, nhưng
  suy giảm nặng dưới dữ liệu **non-IID / lệch miền** (Zhao và cộng sự, 2018) — vốn là
  điều bình thường ở các client F-LLM thật.
- **Tinh chỉnh tiết kiệm tham số (LoRA;** Hu và cộng sự, 2021) khiến việc thích ứng một
  mô hình nền đóng băng trở nên rẻ — rất hợp với FL, vì chỉ cần truyền các adapter tí
  hon (FedIT / OpenFedLLM; Zhang và cộng sự, 2023; Ye và cộng sự, 2024). Nhưng việc
  liên kết adapter LoRA một cách ngây thơ lại tương tác kém với tính dị biệt của client
  và với nhiễu riêng tư.
- **Differential Privacy** (DP-SGD, Abadi và cộng sự, 2016; DP-FedAvg, McMahan và cộng
  sự, 2018) và **tổng hợp an toàn – secure aggregation** (Bonawitz và cộng sự, 2017)
  cho bảo đảm riêng tư hình thức, nhưng đánh đổi riêng tư–độ chính xác trên adapter LLM
  vẫn ít được hiểu rõ.
- **Chưng cất tri thức liên kết** (FedDF, Lin và cộng sự, 2020; FedKD, Wu và cộng sự,
  2022) cho phép hợp tác giữa các mô hình khác kiến trúc, nhưng chủ yếu mới nghiên cứu
  trên thị giác, chưa phải bối cảnh LLM global-lớn ↔ local-nhỏ.

**Khoảng trống TrustFed nhắm tới:** một khung *thống nhất* nơi riêng-tư-từ-thiết-kế, tự
giám sát không nhãn, và chuyển giao tri thức global↔local được **đồng thiết kế** cho
F-LLM tiết kiệm tham số — và nơi các đánh đổi giữa chúng được *đo đạc*, chứ không phải
giả định. Kairouz và cộng sự (2021) liệt kê thẳng những điều này là các bài toán mở
của FL.

## 3. Câu hỏi nghiên cứu

- **RQ1 (Riêng tư).** Ta "mua" được bao nhiêu riêng tư hình thức (DP) và bao nhiêu mức
  giảm bề mặt tấn công (chỉ truyền adapter, secure aggregation) đổi lấy bao nhiêu độ
  chính xác, khi tinh chỉnh một mô hình nền *đóng băng* bằng LoRA trong bối cảnh liên
  kết?
- **RQ2 (Không giám sát, hợp pháp).** Thích ứng **tự giám sát** theo kiểu liên kết (mô
  hình hóa ngôn ngữ che mặt nạ / nhân quả trên văn bản client không nhãn) có thể lấy
  lại phần lớn lợi ích của tinh chỉnh có nhãn, đồng thời giữ một câu chuyện *nguồn gốc
  dữ liệu và tuân thủ* rõ ràng hay không?
- **RQ3 (Chuyển giao tri thức).** Ta có thể chuyển giao tri thức **hai chiều** — chưng
  cất một mô hình global lớn vào các mô hình local nhỏ đã thích ứng miền, và gộp chuyên
  môn local ngược trở lại mô hình global — tốt hơn FedAvg dưới lệch miền mạnh hay không?

## 4. Cách tiếp cận đề xuất — khung TrustFed

TrustFed giữ một **mô hình nền pretrain đóng băng** trên mọi client và chỉ huấn luyện
các mô-đun nhẹ, có thể truyền đi. Một quyết định này dẫn động cả ba trụ cột: nó thu nhỏ
gói tin ~100–500×, thu nhỏ bề mặt tấn công riêng tư, và khiến việc chưng cất giữa các
mô hình khác kiến trúc trở nên khả thi.

**Trụ cột 1 — Riêng tư từ thiết kế.** Chỉ truyền adapter LoRA, không bao giờ truyền dữ
liệu thô hay trọng số đầy đủ. Chồng thêm (a) *DP-FedAvg mức người-dùng* (clip cập nhật
mỗi client + nhiễu Gauss hiệu chỉnh) để có bảo đảm hình thức `(ε, δ)`, và (b) *tổng hợp
an toàn* để server chỉ thấy tổng các cập nhật. Nghiên cứu *biên đánh đổi riêng tư–độ
chính xác* theo hạng `r`, ngưỡng clip `C`, và nhiễu `σ`.

**Trụ cột 2 — Học không giám sát tuân thủ pháp lý.** Thay mục tiêu có nhãn bằng thích
ứng **tự giám sát** liên kết (MLM/CLM) trên văn bản client không nhãn, để client đóng
góp mà không bao giờ để lộ nhãn hay tài liệu. Đi kèm một *sổ cái nguồn gốc dữ liệu và
đồng thuận* cho từng client, để quá trình huấn luyện có thể kiểm toán được theo yêu cầu
quy định.

**Trụ cột 3 — Chuyển giao tri thức global↔local hiệu quả.** Dùng FedAvg trên adapter
làm bộ tổng hợp cơ sở, rồi cải tiến bằng **chưng cất hai chiều**: mô hình global lớn
dạy các mô hình local nhỏ trên dữ liệu công khai/thay thế không nhãn (global→local), và
adapter local được gộp (có nhận biết tính dị biệt) trở lại mô hình global (local→global).
Điều này hỗ trợ cả những client không thể chứa nổi mô hình đầy đủ — họ chỉ giữ một "học
trò" đã chưng cất.

## 5. Công việc sơ bộ (đã hoàn thành — xem repo này)

Tôi đã dựng một pipeline đầu-cuối chạy được, giúp *giảm rủi ro* cho phần cơ chế lõi và
chứng minh tôi có thể thực thi chương trình nghiên cứu này.

**Tầng 0 — FedAvg từ nguyên lý** (`federated.py`, Flower + MNIST). Một mô phỏng FedAvg
5 client tái tạo mốc chuẩn tập trung, để nắm chắc cơ chế client/server và thách thức
non-IID.

**Tầng 1 — Federated LoRA trên một LLM đã pretrain** (`fed_lora.py`). Hiện vật đinh,
chạm cả ba trụ cột trong một script chạy vài phút trên CPU:

- Một mô hình nền **BERT-tiny pretrain, đóng băng** được thích ứng bằng **LoRA tự viết
  từ đầu** (không dùng `peft`), trên **ag_news** với phân hoạch **Dirichlet non-IID**
  qua 5 client — đúng bối cảnh thực tế "mỗi client một miền khác nhau".
- Chỉ **8.708 tham số adapter** được truyền mỗi vòng, so với **4.394.628** nếu truyền
  cả mô hình — gói tin **nhỏ hơn ~505×** và bề mặt tấn công riêng tư nhỏ tương ứng.
- **DP-FedAvg mức người-dùng** (clip cập nhật + nhiễu Gauss) như một "núm vặn" riêng tư
  bật/tắt được, để đo trực tiếp đánh đổi riêng tư–độ chính xác.

**Kết quả sơ bộ nổi bật** (`RESULTS.md`, độ chính xác trên tập test ag_news):

| Chế độ | Độ chính xác |
|---|---|
| Centralized LoRA (trần, không có riêng tư) | **0.862** |
| Local-only LoRA (không hợp tác) | 0.560 |
| **Federated LoRA (dữ liệu thô ở nguyên client)** | **0.721** |
| Federated LoRA + DP mức người-dùng (C=1.0, σ=0.02) | 0.636 |

Phát hiện then chốt là **mức tăng do chuyển giao tri thức +0.161** từ local-only (0.560)
lên federated (0.721): dưới lệch miền, sự hợp tác giúp các client yếu nhất một cách *đo
được*, trong khi dữ liệu thô không hề di chuyển. Thêm DP mức người-dùng chỉ tốn ~8.5
điểm (0.721 → 0.636) ở mức nhiễu này và *vẫn hội tụ* — một điểm cụ thể đầu tiên trên
biên đánh đổi riêng tư–độ chính xác mà TrustFed đề xuất lập bản đồ.

## 6. Kế hoạch nghiên cứu (dự kiến, ~12 tháng làm RA)

| Giai đoạn | Trọng tâm | Sản phẩm |
|---|---|---|
| 1 (th1–2) | Tái tạo + mở rộng bộ khung lên DistilBERT/GPT-2; thêm baseline nhận biết dị biệt | Bộ benchmark tái lập được |
| 2 (th3–5) | **RQ1**: DP + secure aggregation trên adapter; lập bản đồ biên riêng tư–độ chính xác | Nghiên cứu thực nghiệm + kế toán `(ε,δ)` |
| 3 (th5–8) | **RQ2**: thích ứng tự giám sát liên kết (MLM/CLM) trên văn bản không nhãn | Kết quả thích ứng không nhãn |
| 4 (th8–11) | **RQ3**: chưng cất hai chiều global↔local dưới lệch miền | Phương pháp tổng hợp mới |
| 5 (th11–12) | Tích hợp, khảo sát cắt bỏ (ablation), viết bài | Bản thảo bài hội thảo/hội nghị |

## 7. Đóng góp kỳ vọng

1. Một **benchmark F-LLM mở, tái lập được** biến thiên *đồng thời* riêng tư, mức sẵn có
   của nhãn, và tính dị biệt của client — đa số nghiên cứu trước chỉ đổi một yếu tố.
2. Một **biên đánh đổi riêng tư–độ chính xác–truyền thông được đo đạc** cho federated LoRA.
3. Một phương pháp **chưng cất hai chiều** cho bối cảnh global-lớn ↔ local-nhỏ, với các
   client không bao giờ cần chứa mô hình đầy đủ.
4. Một **giao thức huấn luyện định hướng tuân thủ** (nguồn gốc + sổ đồng thuận) làm cho
   tuyên bố "học không giám sát hợp pháp" trở nên cụ thể.

## 8. Vì sao là tôi (mức phù hợp với vị trí RA)

Tôi không chỉ thấy bài toán này thú vị — tôi đã *tự cài đặt vòng lặp lõi* của nó. Bắt
đầu từ một repo trống, tôi đã dựng một mô phỏng FedAvg đúng, rồi một bộ tinh chỉnh
federated LoRA trên một transformer pretrain thật với LoRA tự viết, phân hoạch non-IID,
DP mức người-dùng, và hạch toán truyền thông — đồng thời gỡ được những chế độ hỏng tinh
vi trên đường đi (ví dụ: việc trung bình adapter chỉ có nghĩa trên một mô hình nền *dùng
chung, đã pretrain*). Tôi thành thạo PyTorch và hệ sinh thái HuggingFace, tôi viết code
mà người khác đọc và chạy được, và tôi có động lực với các bối cảnh thực tế, bị quản lý,
nơi F-LLM thật sự quan trọng. Tôi rất sẵn lòng bắt đầu bằng việc tái tạo một kết quả do
thầy/cô chọn từ công trình gần đây của nhóm.

## Tài liệu tham khảo

*(Giữ nguyên tiếng Anh theo quy ước học thuật.)*

1. McMahan et al. *Communication-Efficient Learning of Deep Networks from
   Decentralized Data* (FedAvg). AISTATS, 2017.
2. Zhao et al. *Federated Learning with Non-IID Data.* arXiv:1806.00582, 2018.
3. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022.
4. Zhang et al. *Towards Building the Federated GPT: Federated Instruction Tuning
   (FedIT).* 2023.
5. Ye et al. *OpenFedLLM: Training LLMs on Decentralized Private Data via Federated
   Learning.* KDD, 2024.
6. Abadi et al. *Deep Learning with Differential Privacy* (DP-SGD). CCS, 2016.
7. McMahan et al. *Learning Differentially Private Recurrent Language Models*
   (DP-FedAvg). ICLR, 2018.
8. Bonawitz et al. *Practical Secure Aggregation for Privacy-Preserving Machine
   Learning.* CCS, 2017.
9. Lin et al. *Ensemble Distillation for Robust Model Fusion in Federated Learning*
   (FedDF). NeurIPS, 2020.
10. Wu et al. *Communication-Efficient Federated Learning via Knowledge
    Distillation* (FedKD). Nature Communications, 2022.
11. Kairouz et al. *Advances and Open Problems in Federated Learning.* Foundations
    and Trends in ML, 2021.
