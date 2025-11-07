N = 500 # 1000
TRAIN_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/train.txt"
VAL_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/val.txt"

# src = TRAIN_FILE
# dst = "test_data/train.txt"

src = VAL_FILE
dst = "test_data/val.txt"

with open(src, "r", encoding="utf-8") as fin, \
     open(dst, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        if i >= N:
            break
        fout.write(line)