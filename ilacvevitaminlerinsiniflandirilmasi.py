#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Dropout 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#belirli koşullar gerçekleştiğinde yani los düştüğünde veya accuracy istenilen seviyeye çıktığında traine devam etmeye gerek yok
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
#mobil cihazlarda optimize edilmiş hafif ve verimli derin öğrenme modeli.Transfer learning.
from tensorflow.keras import Model
from tensorflow.keras.layers import Normalization, Rescaling, Resizing

from pathlib import Path
import os.path

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
# %% load data
dataset = "Drug Vision/Data Combined"
image_dir =Path(dataset)
#image_dir kullanarak detaylı inceleyelim.

#filepaths= list(image_dir.glob(r"**/*.jpg"))+list(image_dir.glob(r"**/*.png"))+list(image_dir.glob(r"**/*.JPG"))
filepaths= list(image_dir.glob(r"**/*.jpg"))+list(image_dir.glob(r"**/*.png"))

labels = list(map(lambda x: os.path.split(x)[0][1],filepaths))

filepaths=pd.Series(filepaths,name ="filepath").astype(str)
labels=pd.Series(labels,name="labels")

image_df =pd.concat([filepaths,labels],axis=1)
#jpgleri aynı olarak aldı yani JPG ile jpg aynı aldı.


# %% data visualization 
random_index=np.random.randint(0,len(image_df),16)
# 0 ile 8440 arasında 16 adet rastgele sayı üret.Ynai rastgele 16 görüntüyü seçip görselleştiricez.
fig,axes =plt.subplots(nrows=4,ncols=4,figsize=(11,11))

for i,ax in enumerate(axes.flat):
#eksenleri flat ile düzleştirdik.
    ax.imshow(plt.imread(image_df.filepath[random_index[i]]))
    ax.set_title(image_df.labels[random_index[i]])
plt.tight_layout()
#Random olarak aldığımız için her çalıştırdığımızda farklı görseller geliyor.
#Görseller sentetik arka plan değiştirilerek veri çeşitlendirilmiş.


# %% data preprocessing(veri ön işleme):train-test split,data augmentation(veri artırımı),resize(veri boyutu değiştirimi),rescaling(veri normalizasyonu)
#data preprocessing
train_df,test_df = train_test_split(image_df,test_size=0.2,shuffle=True,random_state=42)
#shuflle veriyi karışık olarak alıp bölme işlemi yapıyor.

#data augmentation(veri artırımı)
train_generator = ImageDataGenerator(
                  preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
                  validation_split=0.2)
#mobilenet_v2 modeline göre girdiler normalize edilir.validation_split de train setinin ikiye arayacak %80 train %20 validation olacak.
test_generator=ImageDataGenerator(
                preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

train_images=train_generator.flow_from_dataframe(
                    dataframe=train_df,
                    x_col ="filepath", #ındependent->goruntu
                    y_col="labels", #dependent ->target variable ->etiket
                    target_size =(224,224),
                    color_mode="rgb",
                    class_mode="categorical",
                    batch_size=64,# 6752 veriyi 64lü paketler halinde işleyecek.
                    shuffle=True,
                    seed=42,
                    subset="training") #goruntulerin boyutu
val_images=train_generator.flow_from_dataframe(
                 dataframe=train_df,
                 x_col="filepath",
                 y_col="labels",
                 target_size=(224,224),
                 color_mode="rgb",
                 class_mode="categorical",
                 batch_size=64,
                 shuffle=True,
                 seed=42,
                 subset="validation")
test_images=train_generator.flow_from_dataframe(
                 dataframe=test_df,
                 x_col="filepath",
                 y_col="labels",
                 target_size=(224,224),
                 color_mode="rgb",
                 class_mode="categorical",
                 batch_size=64,
                 shuffle=False,
           )                            
#dataframeden gelen bilgileri kullanarak train_images kullanıyoruz.görüntüyü içe aktarmak yerine işlemleri toplu olarak flow_from_dataframe ile yapıyor

#resize,rescale
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(224, 224, 3)),  # Giriş katmanını ekleyin
        tf.keras.layers.Resizing(224, 224),
        tf.keras.layers.Rescaling(1./255),
    ]
)
# 1./255 ->1/255 nokta ile float oldu


# %%  transfer learning modeli(MobileNetV2),training
pretrained_model=tf.keras.applications.MobileNetV2(
                    input_shape=(224,224,3),  #girdi yani görüntülerin boyutu
                    include_top=False,    #mobilenet'in sınıflandırma katmanı(false)
                    weights="imagenet",   #hangi veri setiyle eğitimi
                    pooling="avg")
#include_top =önceden eğitilmiş veri setinin son katmanını dahil etmemek için false yaptık bu katmanı biz eğiteceğiz.
#weights önceden eğitilen verilerin hangi veri seti ile eğitildi?
#pooling katmanını yapılandırdık.
pretrained_model.trainable =False
#yeniden train etmeyeceğiz.Sınıflandırma katmanını kullanarak transfer learaning yapıyoruz. 

#create checkpoint callback
checkpoint_path ="checkpoint.weights.h5"
checkpoint_callback=ModelCheckpoint(checkpoint_path,
                                    save_weights_only=True,
                                    monitor="val_accuracy",
                                    save_best_only=True)

#check_point gerçekleştiği zaman sadece weightleri kaydedecek.
#save_best_only:En iyi modeli kaydet.En iyi validasyon accuracy değerine göre.

early_stopping=EarlyStopping(monitor="val_loss",
                             patience=5,
                             restore_best_weights=True)
#neye göre durduracak? monitor =val_loss
#patience : eğitim sırasında val_loss 5 epouch zamanı boyunca iyileşmezse eğitimi durdur.Öğrenemiyor.
#restore_best_weights=True  :Eğitim bittiğinde modelin en başarılı olduğu zamanki weightleri.

#training model-classification blok

def resize_and_rescale(inputs):
    # Görüntüyü 224x224 boyutlarına yeniden boyutlandırıyoruz.
    x = Resizing(224, 224)(inputs)
    # Görüntüyü 0-1 aralığına normalleştiriyoruz.
    x = Rescaling(1./255)(x)
    return x
inputs = pretrained_model.input
x = resize_and_rescale(inputs)  # resize_and_rescale fonksiyonu kullanımı
x = Dense(256,activation="relu")(pretrained_model.output)
x = Dropout(0.2)(x)
x = Dense(256,activation="relu")(x)
x = Dropout(0.2)(x)
#Binary Classification yapmıyorsak activation softmax olmalı.
outputs = Dense(10, activation="softmax")(x) 
model =Model(inputs =inputs,outputs=outputs)

batch_size=32
# Eğer sadece iki sınıf varsa
model = Model(inputs=inputs, outputs=Dense(1, activation='sigmoid')(x))  # 1 çıkış ve sigmoid aktivasyonu
model.compile(optimizer=Adam(0.0001), loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_images,
    steps_per_epoch=len(train_images) // batch_size,
    validation_data=val_images,
    validation_steps=len(val_images) // batch_size,
    epochs=10,
    callbacks=[early_stopping, checkpoint_callback]
)
#epoch zamanını değiştirerek overfittingi engelle.

# %% model evulation
# model sonuçları değerlendirme

results = model.evaluate(test_images,verbose = 1)
print("Test loss: ",results[0])
print("Test accuracy: ",results[1])
#0 loss ,1 accuracy

epochs = range(1,len(history.history["accuracy"])+1)
hist=history.history


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs,hist["accuracy"],"bo-",label="Training Accuracy")
plt.plot(epochs,hist["val_accuracy"],"r^-",label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()


plt.subplot(1,2,2)
plt.plot(epochs,hist["loss"],"bo-",label="Training loss")
plt.plot(epochs,hist["val_loss"],"r^-",label="Validation loss")
plt.title("Training and Validation loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%
pred = model.predict(test_images)
pred =np.argmax(pred,axis=1)

labels = (train_images.class_indices)
labels = dict((v,k)for k,v in labels.items())
pred =[labels[k] for k in pred]

random_index=np.random.randint(0,len(test_df)-1 , 15)
fig,axes =plt.subplots(nrows=5,ncols=3,figsize=(11,11))

for i,ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.filepath[random_index[i]]))
    if test_df.labels.iloc[random_index[i]]==pred[random_index[i]]:
        color="green"
    else:
        color="red"
    ax.set_title(f"True : {test_df.labels.iloc[random_index[i]]}\n predicted:{pred[random_index[i]]}",color =color)
plt.tight_layout()

y_test=list(test_df.labels)
print(classification_report(y_test,pred))