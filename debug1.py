from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 示例文本
text = "This is an example sentence. Note that some stopwords are removed."
stop_words = set(stopwords.words('english'))

# 分词
words = word_tokenize(text)

# 删除停用词
filtered_sentence = [word for word in words if not word.lower() in stop_words]

print("Original sentence:", text)
print("Filtered sentence:", " ".join(filtered_sentence))