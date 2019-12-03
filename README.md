# -Image-classification

import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
from keras import layers, models, optimizers, callbacks
from keras.layers import Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

■写真一括取り込み
X = []
# 画像群の読み込み
images = glob.glob(os.path.join(r"C:\***\input_gazou\*")) #フォルダ名
# 読み込んだ画像を順に拡張
for i in range(len(images)):
    img = np.array(Image.open(images[i]))
    X.append(img/255.)

#　取り込めてるか確認
plt.imshow(X[0])

■複数画像認識＆顔画像トリミング＆画像出力
in_dir = r"C:\***\input_gazou\*" #inputフォルダ
out_dir = r"C:\***\out_gazou" #ouputフォルダ
in_jpg = glob.glob(in_dir)
in_fileName = os.listdir(r"C:*****")

for num in range(len(in_jpg)):
    #ファイル読み込み
    image = cv2.imread(str(in_jpg[num]))
    if image is None:
        print("Not open:",line)
        continue
        
    #顔認識
    cascade = cv2.CascadeClassifier(r"C:***\haarcascades\haarcascade_frontalface_alt.xml") #顔認識用特徴量ファイルを読み込む
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # グレースケールに変換
    face_list = cascade2.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64,64)) #顔写真抽出
    #顔が１つ以上検出された時
    if len(face_list) > 0:
        for rect in face_list:
            x,y,w,h = rect #顔写真部分の座標軸
            image = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]] #顔写真部分の座標軸からトリミングした新たな写真を作る
            if image.shape[0] < 64:
                continue
            image = cv2.resize(image, (64,64)) #新たに作った顔写真をリサイズ
            
    #顔が検出されなかったとき
    else:
        print("no face")
        continue
    
    print(image.shape)
    
    #保存
    fileName = os.path.join(out_dir, str(in_fileName[num])+".jpg")
    cv2.imwrite(str(fileName), image)

■トリミング画像インプット＆シャッフル
in_dir2 = r"C:\***\out_gazou" #ouputフォルダ
in_jpg2 = glob.glob(os.path.join(in_dir2, "*.jpg"))
img_file_name_list = os.listdir(r"C:\***\out_gazou")
#img_file_name_listをシャッフル、そのうち半分をtest_imageディテクトリに入れる
random.shuffle(in_jpg2)
import shutil
for i in range(len(in_jpg2)//2):
    shutil.move(str(in_jpg2[i]), r"C:\***\test_image") #テストフォルダ
    
■画像の水増し
#水増しの関数の定義(keras.preprocessing.image.ImageDataGenerator使ってもいいかも)
def scratch_image(img, flip=True, thr=True, filt=True):
    # 水増しの手法を配列にまとめる
    methods = [flip, thr, filt]
    # ぼかしに使うフィルターの作成
    filter1 = np.ones((3, 3))
    # オリジナルの画像データを配列に格納
    images = [img]
    # 手法に用いる関数
    scratch = np.array([
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1],
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),
    ])
    # 加工した画像を元と合わせて水増し
    doubling_images = lambda f, imag: np.r_[imag, [f(i) for i in imag]]
    
    for func in scratch[methods]:
        images = doubling_images(func, images)
    return images


# 画像の読み込み
in_dir2 = r"C:\***\out_gazou"
in_jpg2 = glob.glob(os.path.join(in_dir2, "*.jpg"))
img_file_name_list = os.listdir(r"C:\***\out_gazou")

# 水増し画像生成＆書き出し
for i in range(len(in_jpg2)):
    print(str(in_jpg2[i]))
    img = cv2.imread(str(in_jpg2[i]))
    scratch_face_images = scratch_image(img)
    for num, im in enumerate(scratch_face_images):
        fn, ext = os.path.splitext(img_file_name_list[i])
        file_name = os.path.join(r"C:\***\out_gazou",str(fn+"."+str(num)+".jpg"))
        cv2.imwrite(str(file_name), im)

■モデル
flg = ["1", "0"]

###日本語読み込み用imread定義
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


#####################
# 学習データのラベル付け
X_train = []
Y_train = []

for i in range(len(flg)):
    img_file_name_list = os.listdir(os.path.join(r"C:\***\out_gazou", flg[i]))
    print(len(img_file_name_list))
    for j in range(0, len(img_file_name_list)):
        n = os.path.join(r"C:\***\out_gazou", flg[i], img_file_name_list[j])
        img = imread(n)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        X_train.append(img)
        Y_train.append(i)

# テストデータのラベル付け
X_test = [] # 画像データ読み込み
Y_test = [] # ラベル

for i in range(len(flg)):
    img_file_name_list = os.listdir(os.path.join(r"C:\***\test_image", flg[i]))
    print(len(img_file_name_list))
    for j in range(0, len(img_file_name_list)):
        n = os.path.join(r"C:\***\test_image", flg[i], img_file_name_list[j])
        img = imread(n)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        X_test.append(img)
        # ラベルは整数値
        Y_test.append(i)

##確認
print(len(Y_test))
print(len(X_test))
print(len(X_train))
print(len(Y_train))

X_train = np.array(X_train)
X_test = np.array(X_test)
print(X_test.shape)
print(X_train.shape)


#####################################model
###DeepLearning
X_train = np.array(X_train)
X_test = np.array(X_test)
X_train2 = X_train/255
X_test2 = X_test/255

# モデルの定義(ドロップアウト追加)
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3), activation="relu", input_shape=(64,64,3))) #imput_shapeはX_train.shapeに合わせる
model.add(Conv2D(16, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(16, (3,3), activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="sigmoid"))


###############validation_splitで実施
# 準備
model_link = os.path.join(r"***\code", "model_file_keras1_male.hdf5")
early_stopping = EarlyStopping(patience=5, verbose=1)
TB_link = r"***\code"

#モデルのコンパイル
model.compile(loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4), metrics=["acc"])

#学習
history = model.fit(X_train2, Y_train, epochs=15, verbose=1, validation_split=0.1, 
                          callbacks=[ModelCheckpoint(model_link, save_best_only=True), TensorBoard(log_dir=TB_link)])

#lossの最も少ないEpoch
model1 = load_model(r"***\code\model_file_keras1_male.hdf5")
predict_y1 = model1.predict_classes(X_test2)
#混合行列
confusion_matrix(Y_test, predict_y1)
print(classification_report(Y_test, predict_y1, target_names=["False", "True"]))


############ほかのモデル
X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0], -1) #4次元を2次元行列に変換


###ロジスティック回帰
logreg = LogisticRegression()
logreg_model = logreg.fit(X_train, Y_train)
Y_pred = logreg_model.predict(X_test)
#混合行列
confusion_matrix(Y_test, Y_pred)

################LightGBM
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
lgbm_params = {
    "objective": "binary",
    "num_iterations" : 500
}
lgb_model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval)
Y_pred_proba_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
Y_pred_lgb = np.where(Y_pred_proba_lgb > 0.5, 1, 0)
#混合行列
confusion_matrix(Y_test, Y_pred_lgb)

