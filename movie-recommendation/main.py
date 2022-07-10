from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
cv = CountVectorizer()

txt = ["HELLO WORLD", "WORLD HELLO WORLD","HELLO HELLO"]
x = cv.fit_transform(txt)
print(x.toarray())