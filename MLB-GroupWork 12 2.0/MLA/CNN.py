import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd
import numpy as np

from nltk.corpus import wordnet

df=pd.read_csv("C:\\Users\\lenovo\\Desktop\\研一\\machine learningb\\group project\\MLB-GroupWork\\MLB-GroupWork\\原始数据集\\final.csv",encoding='unicode_escape')
print(len(df))
length=len(df)

label_txt=[]
texts=[]

for i in range(length):
    texts.append(df.iloc[i][2].replace('\xa0',' ').replace('\n',' ').replace('ï\x82',' ').replace('&',' ').replace('\x80',' ').replace('â',' ').replace('\x9c',' ').replace('\x9d',' ').replace('\x93',' '))
    label_txt.append(df.iloc[i][1])

length=len(texts)

# experiment_texts=[]#取出来做实验的
# experiment_labels=[]
# for i in range(100):
#     experiment_texts.append(texts[i*12])
#     experiment_labels.append(label_txt[i*12])
# for i in range(100):
#     del texts[i*12-i]
#     del label_txt[i*12-i]

length=len(texts)
#数据增强
import random
for i in range(10000):

    random_data = random.randint(0, length)
    words=[]
    if random_data%6==0:
        continue

    # 随机交换20个单词

        #random_data = random.randint(0, length)
    label_empty = label_txt[random_data]

    for k in range(200):
        words = texts[random_data].split(" ")
        words_length = len(words) - 1
        random_changeword_1 = random.randint(0, words_length)
        random_changeword_2 = random.randint(0, words_length)
        words[random_changeword_1], words[random_changeword_2] = words[random_changeword_2], words[random_changeword_1]


    # 同义词替换
        #random_data = random.randint(0, length)
    for k in range(600):
        words_length = len(words) - 1
        random_changeword = random.randint(0, words_length)
        synonyms = []

        for syn in wordnet.synsets(words[random_changeword]):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())

        synonyms_length = len(synonyms)

        if synonyms_length >= 1:
            random_word = random.randint(0, synonyms_length) - 1
            words[random_changeword] = synonyms[random_word]


    # 随机删除单词
        #random_data = random.randint(0, length)

    for k in range(20):
        words_length = len(words) - 1
        if words_length >= 1:
            random_delete = random.randint(0, words_length)
            del words[random_delete]

    sentence = ' '.join(words)
    texts.append(sentence)
    label_txt.append(label_empty)

    #合并两篇相同类型的文章
num_change_word=0
for i in range(10000):

    random_data_2 = random.randint(0, length-1)
    words_1 = []
    words_2 = []
    if random_data_2 % 6 == 0:
        continue
    if (random_data_2+1) % 6==0:
        continue
    if label_txt[random_data_2]!=label_txt[random_data_2+1]:
        continue
    else:
        num_change_word+=1
        label_empty = label_txt[random_data_2]
        words_1 = texts[random_data_2].split(" ")
        words_2 = texts[random_data_2+1].split(" ")
        words_length_1 = len(words_1) - 1
        words_length_2 = len(words_2) - 1

        for k in range(400):
            random_changeword_1 = random.randint(0, words_length_1)
            random_changeword_2 = random.randint(0, words_length_2)
            words_1[random_changeword_1]=words_2[random_changeword_2]

        for k in range(600):
            words_length = len(words_1) - 1
            random_changeword = random.randint(0, words_length)
            synonyms = []
            for syn in wordnet.synsets(words_1[random_changeword]):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
            synonyms_length = len(synonyms)
            if synonyms_length >= 1:
                random_word = random.randint(0, synonyms_length) - 1
                words_1[random_changeword] = synonyms[random_word]

        sentence = ' '.join(words_1)
        texts.append(sentence)
        label_txt.append(label_empty)

length=len(texts)
print("数据交换完成")
print("一共交换的文章数",num_change_word)
print(length)
dict_label={}
n=-1
for i in range(length):
    if label_txt[i] not in dict_label:
        n=n+1
        dict_label[label_txt[i]]=n
print(dict_label)

labels=[]
#experiment_labels_num=[]

for i in range(length):
    labels.append(dict_label[label_txt[i]])
print("labels:",labels)

# for i in range(100):
#     experiment_labels_num.append(dict_label[experiment_labels[i]])

#数据处理
maxlen=2000
training_samples=10000
validation_samples=3
max_words=10000
texting_samples=200

from keras.utils.np_utils import to_categorical
one_hot_labels=to_categorical(labels)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences=tokenizer.texts_to_sequences(texts)

word_index=tokenizer.word_index
print('Found %s unique tokens.' %len(word_index))

data=pad_sequences(sequences,maxlen=maxlen)

labels=np.asarray(labels)
print('Shape of data tensor:',data.shape)
print('Shape of label tensor:',labels.shape)

experiment_texts=[]#取出来做实验的
experiment_labels=[]
for i in range(200):
    experiment_texts.append(data[i*6])
    experiment_labels.append(one_hot_labels[i*6])
texts_tests=np.asarray(experiment_texts)
labels_tests=np.asarray(experiment_labels)
for i in range(200):
    data=np.delete(data,i*6-i,axis=0)
    one_hot_labels=np.delete(one_hot_labels,i*6-i,axis=0)
print('验证:',data.shape)

indices=np.arange(data.shape[0])
np.random.shuffle(indices)
data=data[indices]
one_hot_labels=one_hot_labels[indices]
print("one_hot_labels:",one_hot_labels)

x_train=data[:training_samples]
y_train=one_hot_labels[:training_samples]
x_val=data[training_samples:training_samples+validation_samples]
y_val=one_hot_labels[training_samples:training_samples+validation_samples]
x_test=data[training_samples+validation_samples:training_samples+validation_samples+texting_samples]
y_test=one_hot_labels[training_samples+validation_samples:training_samples+validation_samples+texting_samples]

#卷积神经网络设计
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras.layers import SimpleRNN
from keras import layers
from keras import regularizers

model=Sequential()
model.add(Embedding(10000,32,input_shape=(2000,)))
#model.add(SimpleRNN(32))
#model.add(Dense(25,activation='softmax'))
model.add(layers.Conv1D(256,5,activation='relu'))#原来是256
model.add(layers.MaxPooling1D(2))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(256,5,activation='relu'))#原来是256
model.add(layers.MaxPooling1D(64))
model.add(layers.Dropout(0.5))
model.add(layers.GlobalAveragePooling1D())
model.add(Dense(17,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=10,batch_size=200,validation_data=(texts_tests,labels_tests))

#画图
import matplotlib.pyplot as plt

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig('acc+validation.jpg')
plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss+validation.jpg')
plt.show()

results=model.evaluate(texts_tests,labels_tests)
model.save('my_model3.h5')
print(results)

import tensorflow as tf
tf.keras.utils.plot_model(model, to_file='model_visualize.png', show_shapes=True)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.imshow(mpimg.imread('model_visualize.png'))
plt.show()