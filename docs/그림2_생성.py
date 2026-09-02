"""그림 2 (노드 통과 수 분포) 재현 스크립트.

원자료: challenge set 80건의 노드 통과 수 분포.
※ 아래 counts는 기존 히스토그램 이미지에서 판독한 값이다.
   evaluation 파이프라인 실행 로그가 확보되면 로그에서 직접 집계하도록 교체할 것.
   합계 80, 평균 7.9625(원고 기재 7.96), 최빈 8, 중앙값 8로 원고 서술과 일치함을 확인.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트 (Windows에서는 "Malgun Gothic" 등으로 교체)
FP = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    font_manager.fontManager.addfont(FP)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FP).get_name()
except Exception:
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

counts = {4: 15, 6: 10, 7: 1, 8: 24, 10: 23, 11: 1, 12: 3, 13: 1, 14: 2}
assert sum(counts.values()) == 80

xs = list(range(4, 15))
ys = [counts.get(x, 0) for x in xs]

fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=300)
ax.bar(xs, ys, width=0.68, color="#4a4a4a", edgecolor="black", linewidth=0.6)
for x, y in zip(xs, ys):
    if y:
        ax.text(x, y + 0.5, str(y), ha="center", va="bottom", fontsize=8.5)

ax.set_xlabel("노드 통과 수", fontsize=10, labelpad=7)
ax.set_ylabel("빈도(건)", fontsize=10, labelpad=7)
ax.set_xticks(xs)
ax.set_ylim(0, 27)
ax.tick_params(labelsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, color="#cccccc", linewidth=0.5)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("그림2_노드통과수분포.png", bbox_inches="tight", facecolor="white")

n = sum(counts.values())
mean = sum(k * v for k, v in counts.items()) / n
print(f"n={n} mean={mean:.4f} mode={max(counts, key=counts.get)} max={max(counts)}")
