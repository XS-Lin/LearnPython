import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import jieba
import numpy as np
from PIL import Image
abel_mask = np.array(Image.open(r"D:\LearnAI\visualization\love_is_returned.png"))
#print(abel_mask.dtype);
text_from_file_with_apath = open(r"D:\LearnAI\visualization\birthday.txt",encoding="utf-8").read()
wordlist_after_jieba = jieba.cut(text_from_file_with_apath, cut_all=True)
wl_space_split = " ".join(wordlist_after_jieba)
my_wordcloud = WordCloud(width=600, height=400, background_color='white', mask=abel_mask,
                         max_words=400, stopwords=STOPWORDS, font_path=r"D:\LearnAI\visualization\chinese.msyh.ttf",
                         max_font_size=100,prefer_horizontal=0.8,margin=2,random_state=30,scale=1.5).generate(wl_space_split)
image_colors=ImageColorGenerator(abel_mask)
#print(image_colors)
plt.imshow(my_wordcloud)
plt.axis("off")
plt.show()