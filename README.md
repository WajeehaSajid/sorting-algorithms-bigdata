# Sorting Algorithms on Real-World Big Data 📊

Assignment implementation of **Bubble Sort, Selection Sort & Insertion Sort**, benchmarked on the [Airline Delay and Cancellation Dataset (2009–2018)](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018) (~7 GB, multi-million rows).

## 🎯 What this project does
- Implements `bubble_sort`, `selection_sort`, and `insertion_sort` from scratch (no built-in `sort()`/`sorted()`)
- Loads a multi-GB CSV dataset using **chunked reading** with pandas (memory-safe, no full load)
- Builds random samples at increasing sizes, in three orderings: **random / sorted / reversed** (average / best / worst case)
- Benchmarks all 3 algorithms across sample sizes and orderings, on both a **numeric** and a **text** column
- Fits runtime to `O(n²)` growth and **extrapolates to full dataset scale**, comparing against Python's built-in Timsort (`O(n log n)`)

## 📂 Files
| File | Description |
|---|---|
| `sort_algorithms.py` | The three sorting algorithm implementations, with time complexity + stability docstrings |
| `assignment_pipeline.py` | Data loading, sampling, benchmarking, plotting, and full-scale extrapolation |
| `sort_benchmark_results.csv` | Raw timing results (size, ordering, algorithm, time_seconds) |
| `runtime_vs_n.png` | Runtime vs. n for all three algorithms (random order) |
| `insertion_orderings.png` | Insertion sort: best vs. average vs. worst case comparison |
| `Part_F_Written_Analysis.docx` | Written analysis & extrapolation reasoning |

## ⚙️ How to run
```bash
pip install pandas matplotlib kaggle
python assignment_pipeline.py
```
Set `DEMO_MODE = False` and point `DATA_PATH` at your downloaded Kaggle CSV(s) to run on the real dataset.

## 📈 Key Takeaways
- All three algorithms confirm the expected `O(n²)` growth pattern on random data
- Insertion sort's best case (already-sorted input) is dramatically faster (`O(n)`) than its worst case (reverse-sorted)
- At full dataset scale (millions of rows), these `O(n²)` algorithms become completely impractical — Timsort wins by orders of magnitude

---
*Class assignment for CS coursework — implemented and benchmarked by Jiya.*
