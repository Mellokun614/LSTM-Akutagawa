import sys
import re
paths = ["viyon.txt", "ningen_shikkaku.txt", "shayo.txt", "hashire_merosu.txt","otogi_zoshi.txt","joseito.txt"]
for path in paths:
    bindata = open("./" + path, "rb")
    lines = bindata.readlines()
    for line in lines:
        text = line.decode('Shift_JIS')
        text = re.split(r'\r', text)[0]
        text = re.split(r'底本', text)[0]
        text = text.replace('｜', '')
        text = re.sub(r'《.+?》', '', text)
        text = re.sub(r'［＃.+?］', '' , text)
        #print(text)
        with open('./data_dazai.txt' , 'a', encoding='utf-8') as f: #path
            f.write(text + '\n')
            
            
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

path = "./data_dazai.txt"
bindata = open(path, "rb").read()
text = bindata.decode("utf-8")
print("Size of text: ",len(text))
chars = sorted(list(set(text)))
print("Total chars :",len(chars))
#辞書を作成する
char_indices = dict((c,i) for i,c in enumerate(chars))
indices_char = dict((i,c) for i,c in enumerate(chars))
#40文字の次の1文字を学習させる. 3文字ずつずらして40文字と1文字というセットを作る
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text)-maxlen, step):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])
X = np.zeros((len(sentences),maxlen,len(chars)),dtype=np.bool)
y = np.zeros((len(sentences),len(chars)),dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t ,char in enumerate(sentence):
        X[i,t,char_indices[char]] = 1
    y[i,char_indices[next_chars[i]]] = 1
    #テキストのベクトル化
    X = np.zeros((len(sentences),maxlen,len(chars)),dtype=np.bool)
    y = np.zeros((len(sentences),len(chars)),dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t ,char in enumerate(sentence):
        X[i,t,char_indices[char]] = 1
    y[i,char_indices[next_chars[i]]] = 1

#LSTMを使ったモデルを作る
model = Sequential() #連続的なデータを扱う
model.add(LSTM(128, input_shape=(maxlen,len(chars))))
model.add(Dense(len(chars)))
model.add(Activation("softmax"))
optimizer = RMSprop(lr = 0.01)
model.compile(loss="categorical_crossentropy",optimizer=optimizer)
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)
#生成する
for iteration in range(1): #学習回数にあたる
    print()
    print("-"*50)
    print("繰り返し回数: ",iteration)
    model.fit(X, y, batch_size=128, epochs=1)
    start_index = random.randint(0, len(text)-maxlen-1)
for diversity in [0.2, 0.5, 1.0, 1.2]:
    print()
    print("-----diversity", diversity)
    generated =""
    sentence = text[start_index: start_index + maxlen ]
    generated += sentence
    print("-----Seedを生成しました: " + sentence + '"')
    sys.stdout.write(generated)

    #次の文字を予測して足す
    for i in range(400):
        x = np.zeros((1,maxlen,len(chars)))
        for t,char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1
    
        preds = model.predict(x, verbose =9)[0] #次の文字を予測
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
model.save('dazai_model.h5')
file = open('dazaigentext.txt','w+',encoding='utf-8').write(generated)
