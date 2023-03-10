import keras
import numpy as np
from keras import backend as K
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.optimizers import Adam
from PIL import Image

import cv2
from random import shuffle
import tensorflow as tf
from utils.metrics import f_score

from net_conv5.pspnet import mobilenet_pspnet
K.set_learning_phase(True)
if tf.__version__>= '2.0.0':
    tf.compat.v1.disable_eager_execution()

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def letterbox_image(image, label , size):
    label = Image.fromarray(np.array(label))
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    label = label.resize((nw,nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w-nw)//2, (h-nh)//2))
    return new_image, new_label


#-------------------------------------------------------------#
#   定义了一个生成器，用于读取datasets2文件夹里面的图片与标签
#-------------------------------------------------------------#
def generate_arrays_from_file(lines, batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            #-------------------------------------#
            #   读取输入图片并进行归一化和resize
            #-------------------------------------#
            name = lines[i].split()[0]
            # 从文件中读取图像
            # img = Image.open("./dataset2/jpg/" + name)

            # 从文件中读取图像
            #img = Image.open(r"F:\wen\zhang\文豪\zzzzz刘元博\Dataset\all5020\jpg" + '/' + name + ".jpg")
            img = Image.open(r"F:\wen\zhang\文豪\zzzzz刘元博\Dataset\Heavy clouds3000\jpg" + '/' + name + ".jpg")
            img = img.resize((WIDTH, HEIGHT), Image.BICUBIC)
            img = np.array(img)/255
            X_train.append(img)

            #-------------------------------------#
            #   读取标签图片并进行resize
            #-------------------------------------#

            # name = lines[i].split(';')[1].split()[0]
            # label = Image.open("./dataset2/png/" + name)
            #label = Image.open(r"F:\wen\zhang\文豪\zzzzz刘元博\Dataset\all5020\png" + '/' + name + ".png")
            label = Image.open(r"F:\wen\zhang\文豪\zzzzz刘元博\Dataset\Heavy clouds3000\png" + '/' + name + ".png")
            label = label.resize((int(WIDTH),int(HEIGHT)), Image.NEAREST)
            if len(np.shape(label)) == 3:
                label = np.array(label)[:,:,0]
            label = np.reshape(np.array(label), [-1])
            one_hot_label = np.eye(NCLASSES)[np.array(label, np.int32)]
            Y_train.append(one_hot_label)

            i = (i+1) % n
        yield (np.array(X_train), np.array(Y_train))

if __name__ == "__main__":
    #---------------------------------------------#
    #   定义输入图片的高和宽，以及种类数量
    #---------------------------------------------#128 240 256 272 288  304 320 336 352 368 384 400 512
    HEIGHT = 320
    WIDTH = 320
    #---------------------------------------------#
    #   背景 + 云 = 2
    #---------------------------------------------#
    NCLASSES = 2

    log_dir = "F:\wen\zhang\文豪\zzzzz刘元博\pspnet_Mobile\logsbaihou\\"
    model = mobilenet_pspnet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
    #---------------------------------------------------------------------#
    #   这一步是获得主干特征提取网络的权重、使用的是迁移学习的思想
    #   如果下载过慢，可以复制连接到迅雷进行下载。
    #   之后将权值复制到目录下，根据路径进行载入。
    #   如：
    #   weights_path = "xxxxx.h5"
    #   model.load_weights(weights_path,by_name=True,skip_mismatch=True)
    #---------------------------------------------------------------------#
    # BASE_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/'
    # model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)
    # weight_path = BASE_WEIGHT_PATH + model_name
    # weights_path = keras.utils.get_file(model_name, weight_path)
    # model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    # 打开训练集的txt
    with open(r"F:\wen\zhang\文豪\zzzzz刘元博\Dataset\Heavy clouds3000/train.txt","r") as f:
        train_lines = f.readlines()

    # 打开验证集的txt
    with open(r"F:\wen\zhang\文豪\zzzzz刘元博\Dataset\Heavy clouds3000/val.txt","r") as f:
        val_lines = f.readlines()
    #---------------------------------------------#
    #   打乱的数据更有利于训练
    #   90%用于训练，10%用于估计。
    #---------------------------------------------#
    # np.random.seed(10101)
    # np.random.shuffle(lines)
    # np.random.seed(None)
    # num_val = int(len(lines)*0.1)
    # num_train = len(lines) - num_val

    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=False, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=60, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)

    #-------------------------------------------------------------------------------#
    #   这里使用的是迁移学习的思想，主干部分提取出来的特征是通用的
    #   所以我们可以不训练主干部分先，因此训练部分分为两步，分别是冻结训练和解冻训练
    #   冻结训练是不训练主干的，解冻训练是训练主干的。
    #   由于训练的特征层变多，解冻后所需显存变大
    #-------------------------------------------------------------------------------#
    # trainable_layer = 60
    # for i in range(trainable_layer):
    #     model.layers[i].trainable = False
    # print('freeze the first {} layers of total {} layers.'.format(trainable_layer, len(model.layers)))

    if True:
        lr = 1e-3
        batch_size = 16
        model.compile(loss = 'categorical_crossentropy',
                optimizer = Adam(lr=lr),
                metrics = ['accuracy', f_score()])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), batch_size))

        gen = generate_arrays_from_file(train_lines, batch_size)
        gen_val =generate_arrays_from_file(val_lines, batch_size)

        model.fit(gen,
                steps_per_epoch=max(1, len(train_lines)//batch_size),
                validation_data=gen_val,
                validation_steps=max(1, len(val_lines)//batch_size),
                epochs=400,
                initial_epoch=0,
                callbacks=[checkpoint, tensorboard, early_stopping])
        model.save_weights(log_dir+'last1.h5')
