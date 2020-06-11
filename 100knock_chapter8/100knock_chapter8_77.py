import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# 単層ニューラルネットワークのモデル作成
def create_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(output_size, activation='softmax', input_shape=(input_size,)))
    return model

def main():
    # 学習データの読み込み
    load_array = np.load('train.vectorizer.npz')
    x_train = load_array['arr_0']
    y_train = load_array['arr_1']

    load_array = np.load('valid.vectorizer.npz')
    x_valid = load_array['arr_0']
    y_valid = load_array['arr_1']

    # モデルの作成
    model = create_model(300, 4)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # コールバックに TensorBoard と ModelCheckpoint を用います
    filepath = 'model.h5'
    callbacks = [
        TensorBoard(log_dir='logs'),
        ModelCheckpoint(filepath)
    ]

    # batch_size を 1,2,4,8 でそれぞれ学習をさせ、その様子を観察します
    batch_size_list = [1, 2, 4, 8]
    for batch_size in batch_size_list:
        model.fit(x_train, y_train,
                  validation_data=(x_valid, y_valid),
                  epochs=1,
                  batch_size=batch_size,
                  callbacks=callbacks)

if __name__ == '__main__':
    main()