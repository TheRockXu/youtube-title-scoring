import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import  quantile_transform
import numpy as np
import tensorflow_hub as hub

def _num_map(d, is_date=False):
    mapping = {
        'K views': 1000,
        'M views': 1000000,
        'B views': 1000000000
    }
    if is_date:
        mapping = {
        ' years ago': 365,
        ' months ago': 30,
        ' weeks ago': 7
    }
    total = 1
    try:
        for k in mapping.keys():
            if k in d:
                d = d.replace(k, '')
                total = float(d) * mapping[k]
    except :
        total=1
        pass
    return total


def data_preprocess():
    df = pd.read_csv('data.csv')
    df.columns = ['id', 'title', 'date', 'views']
    views= df.pop('views').map(_num_map)
    days= df.pop('date').map(lambda x: _num_map(x, is_date=True))
    avg_views = (views/(days+100)).astype(int)
    avg_views = quantile_transform(avg_views.values.reshape(-1, 1), n_quantiles=2, random_state=0, copy=True)
    avg_views = avg_views.round(1)
    categories = pd.Categorical(np.squeeze(avg_views)).categories
    df['avg_views'] = pd.Categorical(np.squeeze(avg_views), categories=categories)
    return df

def tokenize_dataset(df):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50000,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(df['title'].values)
    train_seqs = tokenizer.texts_to_sequences(df["title"].values)
    #
    train_vectors = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    return train_vectors, tokenizer

def base_model(vocab_size):
    # embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, 128),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ]

    )
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError())

    return model

def rnn_model(vocab_size):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, 128),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ]

    )
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError())


    return model

def plot_results(history, save_name='plot.png'):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.show()
    plt.savefig(save_name)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    df = data_preprocess()
    train_vectors, tokenizer = tokenize_dataset(df)
    dataset = tf.data.Dataset.from_tensor_slices((train_vectors[:-1000], df['avg_views'].values.astype(int)[:-1000])).shuffle(1000).batch(64).repeat()
    validate_dataset = tf.data.Dataset.from_tensor_slices((train_vectors[-1000:], df['avg_views'].values.astype(int)[-1000:])).shuffle(1000).batch(64).repeat()
    model = rnn_model(50000)
    print(model.summary())
    history = model.fit(dataset, validation_data=validate_dataset, validation_steps=200, epochs=10, steps_per_epoch=200)
    model.save_weights('checkpoints/rnn.m')
    plot_results(history, 'rnn.png')




