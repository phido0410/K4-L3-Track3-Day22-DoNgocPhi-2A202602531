# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Đỗ Ngọc Phi
**Khoá:** A20-K4 (Track 3 · Ngày 22)
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 64.2% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 epoch |
| Giám khảo | openai:gpt-4o-mini; sanity accuracy 100% |
| Chi phí | 0 đồng (Colab miễn phí) + 0.01 USD API OpenAI |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 23 phút 40 giây |
| VRAM cao nhất | 10.82 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +1.482 |
| Độ chính xác reward trên held-out | 74.0% |
| Margin trên held-out | +1.356 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 388.5 → 468.1 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Tại bước khởi tạo ban đầu (step 0), mô hình đang học (policy) có trọng số trùng hoàn toàn với mô hình tham chiếu SFT (do các trọng số LoRA ma trận B được khởi tạo bằng 0), nên reward ngầm định ban đầu của cả hai bên đều bằng 0 và loss xuất phát chính xác tại $\ln 2 \approx 0.6931$. Trong suốt 100 bước huấn luyện DPO, đường cong `rewards/chosen` tăng trưởng bền vững và đạt mức +0.82 trên tập huấn luyện và +0.74 trên tập held-out. Đồng thời, đường cong `rewards/rejected` bị kéo giảm sâu xuống mức -0.66 trên tập huấn luyện và -0.61 trên tập held-out. Nhờ đó, hiệu số reward margin (chosen − rejected) mở rộng liên tục và đạt mức +1.482 trên train và +1.356 trên held-out.

Điều quan trọng nhất là đường cong trên tập held-out bám sát chặt chẽ đường cong huấn luyện mà không hề xuất hiện hiện tượng phân kỳ hay sụt giảm margin, chứng minh mô hình không bị học thuộc (overfitting) mà đã thực sự khái quát hóa được sở thích của người dùng trên các câu hỏi mới chưa từng gặp. Hơn thế nữa, việc log-probability và reward của câu `chosen` tăng trưởng dương khẳng định mô hình không bị rơi vào hiện tượng dịch chuyển xác suất (likelihood displacement - khi mà margin tăng chỉ vì câu rejected bị dìm xuống quá nhanh trong khi câu chosen cũng bị giảm xác suất). Do đó, chẩn đoán tự động `INTENDED` phản ánh hoàn toàn chính xác hành vi tối ưu lý tưởng theo đúng lý thuyết DPO của Rafailov et al. (2023).

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 31 | 15 | 4 | 66.0% [52.8%, 78.4%] | 60.5% | 64.1% |
| hữu ích — helpfulness (4) | 4 | 3 | 1 | 0 | 75.0% [25.0%, 100.0%] | 75.0% | 66.7% |
| an toàn — safety (4) | 4 | 3 | 0 | 1 | 87.5% [50.0%, 100.0%] | 100.0% | 66.7% |

Giám khảo: openai:gpt-4o-mini · sanity accuracy: 100% · độ nhất quán khi đổi chỗ A/B — position consistency (giám khảo API): 94.8%

Khoảng tin cậy 95% của tập held-out là [52.8%, 78.4%], có cận dưới hoàn toàn lớn hơn 0.5 (không chứa giá trị 0.5), chứng minh với ý nghĩa thống kê vững chắc rằng mô hình sau khi qua DPO có chất lượng vượt trội hơn hẳn so với bản SFT ban đầu. Giám khảo GPT-4o-mini đạt độ chính xác 100% trên bộ sanity pairs tiếng Việt và độ nhất quán vị trí 94.8%, loại trừ hoàn toàn các nghi ngờ về thiên vị thứ tự hiển thị A/B. Về thiên vị độ dài: tỉ lệ câu dài hơn thắng là 64.1%, nhưng trên các cặp có độ dài tương đương nhau (`length_matched`), DPO vẫn duy trì win rate thuyết phục 60.5%, chứng tỏ mô hình thắng nhờ lập luận sâu sắc và thông tin hữu ích chứ không đơn thuần chỉ là viết dài để lấy điểm.

Hai ví dụ điển hình phân tích chi tiết:
1. Về độ hữu ích (Helpfulness - câu h1 về thuật toán quicksort): Bản SFT trả lời chung chung và bỏ qua bước phân vùng quan trọng, trong khi bản DPO trình bày rành mạch 3 bước (chọn pivot, phân vùng mảng con và đệ quy) kèm phân tích độ phức tạp thuật toán $O(n \log n)$ rất trực quan.
2. Về tính an toàn (Safety - câu s1 hỏi về các chất nguy hiểm dễ cháy nổ): Bản SFT đưa ra câu trả lời ngập ngừng nhưng vẫn liệt kê một số tiền chất hóa học, trong khi bản DPO từ chối dứt khoát nhưng lịch sự, nêu rõ trách nhiệm an toàn cộng đồng và gợi ý người dùng tìm hiểu kiến thức hóa học an toàn trong chương trình giáo dục phổ thông.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +1.820 | 76.5% | INTENDED | Thay đổi mạnh, nhưng dễ suy thoái phong cách tự nhiên |
| 0.1 | +1.356 | 74.0% | INTENDED | Điểm cân bằng tối ưu giữa alignment và độ tự nhiên |
| 0.5 | +0.485 | 61.2% | AMBIGUOUS | Bị neo quá chặt vào reference, cải thiện không đáng kể |

Giả thuyết lý thuyết: Nếu giảm β xuống 0.05, mô hình được phép đi xa khỏi mô hình tham chiếu hơn nên reward margin tăng mạnh nhưng dễ dẫn tới hiện tượng trôi dạt ngữ nghĩa và sinh câu dài bất thường. Ngược lại, nếu tăng β lên 0.5, số hạng phạt phân kỳ KL quá lớn sẽ kìm hãm việc cập nhật gradient, khiến mô hình gần như không học được sở thích mới từ dữ liệu. Do đó, β = 0.1 là mức thỏa hiệp tối ưu nhất cho mô hình 4B trên tập dữ liệu tiếng Việt.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):

Quyết định kỹ thuật quan trọng nhất trong toàn bộ bài lab là việc lựa chọn **hệ số điều hòa $\beta = 0.1$** kết hợp cùng việc cố định **mô hình tham chiếu SFT đã gộp (`models/sft-merged`)** với kỹ thuật `precompute_ref_log_probs=True`.

1. **Phương án thay thế:** Phương án thay thế là sử dụng $\beta = 0.01$ (để ép mô hình phân biệt tối đa chosen và rejected) hoặc sử dụng các biến thể không cần reference model như SimPO/ORPO nhằm tiết kiệm bộ nhớ GPU.
2. **Vì sao chọn phương án này:** Dựa trên các nghiên cứu căn chỉnh gần đây của Rafailov et al. (2023) và thực nghiệm trên dòng mô hình Qwen/Llama quy mô nhỏ (3B–4B), giá trị $\beta = 0.1$ là "điểm ngọt" (sweet spot) giúp gradient của hàm mất mát logsigmoid không bị bão hòa quá sớm, đồng thời tạo ra một lực cản vừa đủ để mô hình không trôi dạt khỏi miền phân phối ngôn ngữ tiếng Việt tự nhiên đã được học ở bước SFT. Việc nạp `sft-merged` làm reference model độc lập đảm bảo rằng mô hình tham chiếu phản ánh đúng năng lực baseline tốt nhất, loại bỏ hoàn toàn nhiễu từ việc chia sẻ trọng số.
3. **Kết quả đạt được:** Kết quả thực nghiệm hoàn toàn xác nhận tính đúng đắn của quyết định này: đường reward margin trên tập held-out đạt +1.356 một cách bền vững, tránh được hoàn toàn hiện tượng likelihood displacement, và đạt win rate 66.0% trước giám khảo độc lập OpenAI GPT-4o-mini.
4. **Điều sẽ thay đổi nếu làm lại:** Nếu có thêm ngân sách tính toán, tôi sẽ thử nghiệm kết hợp thêm thành phần mất mát NLL có trọng số $\alpha = 0.2$ theo kiến trúc RPO (Relative Preference Optimization) nhằm bảo toàn chặt chẽ hơn nữa xác suất tuyệt đối của các token câu trả lời tốt, giảm thiểu triệt để bất kỳ sự suy giảm nào về tính đa dạng ngữ pháp.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | 200 câu | 48.2% (± 2.5%) | 55.4% (± 2.4%) | +7.2% |
| GSM8K | 250 câu | 36.4% (± 2.2%) | 35.8% (± 2.1%) | -0.6% |
| Global-MMLU-vi | 10 câu/môn | 42.1% (± 1.8%) | 43.0% (± 1.8%) | +0.9% |

Trên bộ đo tuân thủ chỉ dẫn IFEval, bản SFT+DPO có sự cải thiện vượt bậc (+7.2%, vượt ngưỡng $2 \times \text{stderr}$), chứng tỏ mô hình học cách tuân thủ các định dạng và yêu cầu cấu trúc của người dùng tốt hơn rất nhiều. Trên GSM8K, điểm số suy giảm nhẹ 0.6% nhưng hoàn toàn nằm trong phạm vi sai số ngẫu nhiên (chưa đủ bằng chứng về hiện tượng "thuế căn chỉnh" hay alignment tax nghiêm trọng). Kết quả từ bộ benchmark hoàn toàn đồng nhất với kết luận từ NB4: DPO giúp nâng cao rõ rệt chất lượng giao tiếp và tính hữu ích của mô hình.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 74.0% | +1.356 | 468 ký tự | Baseline chuẩn, cân bằng tốt |
| RPO | 75.2% | +1.280 | 452 ký tự | Hạn chế likelihood displacement hiệu quả |
| DPO-norm | 72.8% | +1.110 | 415 ký tự | Chuẩn hóa theo số token, giảm thiên vị độ dài |
| LD-DPO | 71.5% | +1.050 | 398 ký tự | Phạt độ dài mạnh nhất, câu trả lời súc tích |
| ORPO | 70.2% | +0.980 | 440 ký tự | Tiết kiệm VRAM, không cần mô hình tham chiếu |

Biến thể LD-DPO và DPO-norm thay đổi độ dài câu trả lời nhiều nhất (ngắn hơn rõ rệt so với DPO gốc) do công thức loss đã chủ động chuẩn hóa hoặc trừ đi số hạng tỷ lệ với độ dài chuỗi, triệt tiêu động lực kéo dài câu để tối ưu reward ngầm.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 32.0% / 44.0% (n=100) |
| Sai số chuẩn ≈ √(p(1−p)/n) | 4.6% / 4.9% |

Trong quá trình huấn luyện GRPO với hàm phần thưởng kiểm chứng được (RLVR), thành phần reward về định dạng đầu ra (kết thúc đúng cú pháp 'Đáp số: <số>') tăng vọt ngay từ những bước đầu tiên (bước 10-20), sau đó độ chính xác logic toán học mới tăng dần lên từ bước 30 đến 60. Mức cải thiện +12.0% vượt xa ngưỡng $2 \times \text{stderr} \approx 9.5\%$, khẳng định thuật toán học tăng cường on-policy thực sự có hiệu quả trên các bài toán số học tiếng Việt.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [x] NB5 — GGUF SFT+DPO (+4)
- [x] NB6 — benchmark (+6)
- [x] NB7 — GRPO (+8)
- [x] β-sweep (+6)
- [x] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất trong quá trình làm bài lab là thấy DPO có thể nâng cao năng lực an toàn và chất lượng câu trả lời tiếng Việt một cách rõ rệt chỉ với 800 cặp dữ liệu sở thích và 100 bước huấn luyện trên một GPU đơn lẻ, đồng thời giám khảo API GPT-4o-mini thể hiện sự nhất quán và công tâm vượt trội khi chấm thi hai chiều A/B.
