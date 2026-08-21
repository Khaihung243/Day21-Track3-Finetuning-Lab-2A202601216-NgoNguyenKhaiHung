# Lab 21 — Evaluation Report

**Họ tên**: Ngô Nguyễn Khải Hưng  **MSSV**: 2A202601216  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6 GB usable)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage 4 trường (mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)*, tier đặt 1024 để đảm bảo an toàn tuyệt đối không cắt cụt chuỗi |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*: `verdict: reasoning preserved — safe to train on traces`. Chat template của Qwen3.5 bảo toàn khối `<think>` trong chuỗi render, không bị xóa bỏ, đảm bảo an toàn khi huấn luyện trên dữ liệu suy luận.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.758 | 0.000 | 3179.2 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.000 | 1020.9 |
| (c) LoRA fine-tune | 0.970 | 0.522 | 1.000 | 1408.6 |

**(b) có thật sự mạnh hơn (a) không?** Có — target tăng vọt từ 0.000 lên 0.765 (+76.5%), định dạng JSON đạt chuẩn 100% (format=1.000) và độ trễ giảm hơn 3 lần từ 3179.2ms xuống 1020.9ms do prompt tối ưu ép model dừng sinh sau khi hoàn thành JSON.
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao? Không sửa, giữ nguyên `OPTIMIZED_PROMPT` mặc định của lab (SHA: `719e74d3b6232053`) để đảm bảo tính liêm chính và tính công bằng của mốc đánh giá đối chứng.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6263 | 0.9700 | 918.2 | 12.01 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 | 0.5374 | 0.9700 | 790.0 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5704 | 0.0000 | 927.9 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.9400 | 994.7 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**

Run `attn_only` được nâng rank lên r=283 để khớp chính xác ngân sách tham số (~32.46M tham số huấn luyện, sai lệch <0.03% so với `correct`). Trên tập target thực tế, `attn_only` đạt 0.9700 (hoà điểm số với `correct`), nhưng theo cột `final_loss` huấn luyện ở NB4 thì `attn_only` lại có loss thấp hơn rõ rệt (0.5374 so với 0.6263 của `correct`). Sự bất đồng thứ tự giữa train loss và task target cho thấy rank cực lớn dồn vào ít vị trí (q,v) chỉ giúp ép loss ghi nhớ trên tập train tốt hơn chứ không tạo ra sự vượt trội về năng lực suy luận thực tế. Điều này khẳng định rằng vị trí adapter bao phủ toàn bộ các tầng biến đổi tuyến tính (`text-linear`) mới là đòn bẩy căn bản để tối ưu biểu diễn, thay vì cố gắng tăng rank tại các module attention.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

Run `wrong_lr` giữ nguyên mọi siêu tham số ngoại trừ việc hạ Learning Rate xuống 10 lần ($10^{-5}$ thay vì $10^{-4}$), mô phỏng thang LR của full fine-tune. Kết quả là loss huấn luyện gần như phẳng, chỉ đạt 1.5704 sau 30 step (cao gấp 2.5 lần so với `correct`), dẫn đến điểm target rớt về 0.0000 vì mô hình chưa kịp học cấu trúc JSON mong muốn. Nếu chỉ nhìn đường loss mà không biết nguyên nhân do LR quá thấp, người làm rất dễ kết luận sai lầm rằng phương pháp LoRA bị lỗi, mô hình không thể hội tụ hoặc dữ liệu không đủ chất lượng. Thực tế, các ma trận rank thấp của LoRA đòi hỏi Learning Rate lớn hơn xấp xỉ 10 lần so với full fine-tuning để gradient có thể cập nhật hiệu quả các trọng số adapter trong số step hạn chế.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**

Run `qlora` giúp tiết kiệm VRAM đáng kể khi giảm mức chiếm dụng bộ nhớ từ 12.01 GB xuống còn 7.09 GB (giảm ~41%), cho phép mô hình chạy vừa vặn trên các GPU nhỏ hơn. Tuy nhiên, cái giá phải trả là thời gian huấn luyện tăng lên 994.7s (chậm hơn so với 918.2s của `correct` do chi phí giải lượng tử dequantize on-the-fly) và độ chính xác target bị suy giảm nhẹ từ 0.9700 xuống 0.9400. Kết quả đo đạc thực nghiệm này hoàn toàn ủng hộ khuyến nghị của nhà sản xuất rằng không nên dùng QLoRA 4-bit cho dòng Qwen3.5 nếu VRAM phần cứng vẫn đủ chứa bản fp16/bf16 LoRA, bởi lượng tử hoá làm mất mát biểu diễn mà không đem lại ưu thế về tốc độ.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.236` · `valid_trace_rate = 0.0000`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?

Phán quyết của cổng hồi quy ghi nhận kết quả `FAILED` do xảy ra hiện tượng suy giảm năng lực tổng quát (general capability regression hay catastrophic forgetting). Mặc dù bản fine-tune LoRA đạt mức tăng trưởng rất tốt trên tác vụ mục tiêu với target Δ = +0.205 (tăng từ 0.765 của baseline prompt tối ưu lên 0.970), điểm số trên tập kiểm thử kiến thức tổng quát `eval_regression` đã bị sụt giảm -0.236 (từ 0.7578 xuống 0.5222), vượt qua ngưỡng dung sai cho phép là 0.020. Nguyên nhân trực tiếp là do toàn bộ 225 mẫu huấn luyện đều chỉ bao gồm các ticket chăm sóc khách hàng với định dạng trả về thuần JSON, khiến mô hình bị chuyên môn hoá quá mức (overfitting vào cấu trúc JSON) và làm suy giảm khả năng trả lời các câu hỏi chỉ dẫn ngôn ngữ tự nhiên thông thường. Kết quả FAILED này phản ánh trung thực bài toán thực tế: khi fine-tune một LLM trên dữ liệu đơn nhiệm hẹp, ta bắt buộc phải áp dụng biện pháp phòng ngừa như trộn 1–5% dữ liệu replay tổng quát (deck §14.3) để bảo vệ năng lực nền của mô hình.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | Khớp nhãn | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` | ✅ FT thắng (Score 1.0: Trích xuất chính xác tuyệt đối 4 trường, JSON chuẩn không giải thích thừa). |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình. | `hoan_tien`, `trung_binh`, `ốp lưng điện thoại`, `tieu_cuc` | Khớp nhãn | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}` | ✅ FT thắng (Score 1.0: Nhận diện chuẩn xác sentiment tiêu cực và urgency trung bình). |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | Khớp nhãn | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | ❌ **FT thua** (Score 0.75: Đoán nhầm urgency thành `trung_binh` thay vì `thap` do bỏ sót sắc thái "Khi nào tiện"). |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi. | `san_pham_loi`, `thap`, `nồi chiên không dầu`, `trung_tinh` | Khớp nhãn | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | ❌ **FT thua** (Score 0.75: Đoán nhầm urgency thành `trung_binh` thay vì `thap`). |
| 5 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều. | `san_pham_loi`, `thap`, `áo khoác gió`, `tich_cuc` | Khớp nhãn | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"}` | ❌ **FT thua** (Score 0.75: Đoán nhầm urgency thành `trung_binh` thay vì `thap`). |

**Có mẫu chung nào ở các ca FT thua không?**

Tất cả các ca mô hình Fine-tune bị mất điểm (đều đạt 0.75/1.0) đều có chung một mẫu hình: trong câu ticket của khách hàng xuất hiện cụm từ biểu thị mức độ không khẩn cấp *"Khi nào tiện"*, ứng với nhãn ground truth là `urgency: "thap"`. Tuy nhiên, do trong tập huấn luyện đa số các ticket khiếu nại hoàn tiền (`hoan_tien`) hoặc lỗi sản phẩm (`san_pham_loi`) đều gắn với mức độ khẩn cấp `trung_binh` hoặc `cao`, mô hình fine-tune đã hình thành thiên kiến (bias) và dự đoán mặc định là `trung_binh`. Trong khi đó, prompt tối ưu ở Baseline (b) được định nghĩa ngữ nghĩa chi tiết nên vẫn phân loại đúng trường này.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?

Bản fine-tune LoRA `correct` đã chứng minh được hiệu quả vượt trội trên tác vụ chuyên biệt phân loại ticket CSKH khi nâng độ chính xác target lên 97.0% (vượt xa 76.5% của prompt tối ưu và 0.0% của prompt ngây thơ), đồng thời đảm bảo 100% tuân thủ định dạng JSON với độ trễ tối ưu. Tuy nhiên, chúng ta **chưa nên deploy trực tiếp** bản adapter này như một mô hình tổng quát độc lập vào hệ thống production, vì mô hình đã mắc lỗi suy giảm năng lực nền (regression delta -0.236 trên tập kiểm thử phổ thông). Để triển khai an toàn, ta có hai lựa chọn kiến trúc: (1) Chỉ định tuyến các request phân loại ticket đến một worker riêng gắn adapter LoRA này (theo mô hình phục vụ adapter chuyên biệt ở NB6), hoặc (2) Huấn luyện lại adapter bằng cách bổ sung 1–5% dữ liệu đa mục đích (general replay data) nhằm khắc phục hiện tượng quên thảm họa. Đòn bẩy thực sự trong lab này là sự kết hợp chặt chẽ giữa **Loss Mask chính xác** (ngăn model học vẹt câu hỏi), **Learning Rate đúng thang đo LoRA (~10x full-FT)** và **Vị trí adapter bao phủ toàn diện `text-linear`**.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Loss Mask và Chat Template là nền tảng cốt tử**: Nếu không kiểm tra mask proof mà để loss tính trên toàn bộ chuỗi (`everything`), mô hình sẽ chỉ học cách viết tiếp câu hỏi của khách hàng; chỉ khi mask đúng trên phần câu trả lời của assistant thì gradient mới tập trung tối ưu đúng mục tiêu.
2. **Đừng bao giờ đánh giá mô hình chỉ bằng Train Loss**: Thí nghiệm đối chứng `attn_only` r=283 có loss huấn luyện thấp hơn `correct` (0.5374 vs 0.6263) nhưng điểm số target thực tế lại không hề vượt trội. Train loss thấp có thể chỉ là ghi nhớ cục bộ (memorization); đánh giá trên task metric và cổng hồi quy mới là thước đo chân thực.
3. **Prompt Engineering là một mốc chuẩn bắt buộc phải vượt qua**: Việc thiết lập Baseline (b) với prompt tối ưu trước khi huấn luyện giúp xác lập một ranh giới chi phí - hiệu quả thực tế; fine-tune chỉ thực sự có giá trị khi chứng minh được nó vượt qua năng lực tối đa của prompt engineering.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Trộn thêm 3–5% dữ liệu đàm thoại tổng quát (như tập ShareGPT/Alpaca tiếng Việt) vào tập huấn luyện SFT để triệt tiêu hiện tượng quên thảm họa (đưa `regression_delta` về trong ngưỡng $\ge -0.02$ nhằm đạt phán quyết `PASSED`), đồng thời thử nghiệm kỹ thuật DoRA (Weight-Decomposed Low-Rank Adaptation) để kiểm tra xem việc phân rã độ lớn và hướng của trọng số có cải thiện thêm độ chính xác ở các ca biên hay không.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap (`results/merge_check.json` — delta = 0.0000, hoán đổi thành công adapter)
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/hungjd243/qwen-ticket-lora-lab21
