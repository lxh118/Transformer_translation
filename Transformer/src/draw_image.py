# _*_ coding : utf-8 _*_
# @Time : 2023/12/12 18:24
# @Author : 娄星华
# @File : drawImage
# @Project : Transformer


from collections import Counter
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from src import cn2en_lib, en2cn_lib

# 绘图
plt.switch_backend('TkAgg')  # Use TkAgg as the backend


def plot_loss(train_loss, val_loss):
    plt.figure()
    plt.plot(train_loss, c="red", label="train_loss")
    plt.plot(val_loss, c="blue", label="val_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropyLoss")
    plt.title("CrossEntropyLoss of Train and Validation in each Epoch")
    plt.savefig("image/fig_loss.png")


def plot_acc(train_acc, val_acc):
    plt.figure()
    plt.plot(train_acc, c="red", label="train_bleu")
    plt.plot(val_acc, c="blue", label="val_bleu")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("BLEU")
    plt.title("BLEU of Train and Validation in each Epoch")
    plt.savefig("image/fig_bleu.png")


def word_frequency():
    # 计算每个频次对应的单词数
    frequency_counter = Counter(en2cn_lib.cn_ws.count.values())
    frequencies, word_counts = zip(*frequency_counter.items())
    print(frequencies)
    print(word_counts)
    # 绘制频次分布直方图
    nums = 1000  # 统计的数量
    plt.bar(frequencies[-nums:], word_counts[-nums:], color='skyblue')
    plt.xlabel('word frequency')
    plt.ylabel('Number of words')
    plt.title('Word frequency distribution histogram')
    plt.savefig("../image/Word_frequency_distribution_histogram.png")
    plt.show()


def word_Cloud():
    # 绘制词云图
    # 设置中文字体路径
    font_path = r"C:\Windows\Fonts\simhei.ttf"
    yelpWordCloud = WordCloud(width=800, height=400, background_color='white',
                              font_path=font_path).generate_from_frequencies(
        en2cn_lib.cn_ws.count)
    plt.figure(figsize=(12, 6))
    plt.imshow(yelpWordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word cloud')
    plt.savefig("../image/Word_cloud.png")
    plt.show()


if __name__ == "__main__":
    word_frequency()
    word_Cloud()
    pass
