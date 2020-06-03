from train import rnn_model, tokenize_dataset, data_preprocess
import pandas as pd
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Score titles')
    parser.add_argument('texts', metavar='N', type=str, nargs='+',
                        help='submit list of texts for scoring')
    # Test model
    args = parser.parse_args()
    df = data_preprocess()
    _, tokenizer = tokenize_dataset(df)
    texts = args.texts
    seqs = tokenizer.texts_to_sequences(texts)
    vectors = tf.keras.preprocessing.sequence.pad_sequences(seqs, padding='post')
    model = rnn_model(50000)
    model.load_weights('checkpoints/rnn.m')
    pred = model.predict(vectors)
    pred = pd.Series(-np.squeeze(pred)).rank()
    res_df = pd.concat([pred,pd.Series(texts)], axis=1)
    res_df.index = pred
    print(res_df[1].sort_index())