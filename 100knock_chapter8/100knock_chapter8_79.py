import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# ４層のニューラルネットワークのモデル作成
def create_model(input_size, output_size, hidden_size1=100, hidden_size2=10):
    model = Sequential()
    model.add(Dense(hidden_size1, activation='relu', input_shape=(input_size,)))
    model.add(Dense(hidden_size2, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
    return model

def main():
    # 学習データの読み込み
    load_array = np.load('train.vectorizer.npz')
    x_train = load_array['arr_0']
    y_train = load_array['arr_1']

    load_array = np.load('valid.vectorizer.npz')
    x_valid = load_array['arr_0']
    y_valid = load_array['arr_1']

    load_array = np.load('test.vectorizer.npz')
    x_test = load_array['arr_0']
    y_test = load_array['arr_1']

    # モデルの作成
    model = create_model(300, 4)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # コールバックに TensorBoard と ModelCheckpoint を用います
    filepath = 'model2.h5'
    callbacks = [
        TensorBoard(log_dir='logs2'),
        ModelCheckpoint(filepath)
    ]

    # バッチサイズ 8 で学習させます
    model.fit(x_train, y_train,
              validation_data=(x_valid, y_valid),
              epochs=100,
              batch_size=8,
              callbacks=callbacks)

    # 評価データで正答率から性能を見ます
    test_accuracy = model.evaluate(x_test, y_test)
    print('accuracy:', test_accuracy)

    # 1336/1336 [==============================] - 0s 105us/sample - loss: 0.8917 - accuracy: 0.9154
    # accuracy: [0.8916766409334186, 0.91541916]

if __name__ == '__main__':
    main()