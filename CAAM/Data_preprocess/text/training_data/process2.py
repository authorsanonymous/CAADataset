import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
embed = hub.Module("./universal-sentence-encoder-large_3")

files = os.listdir('./')

all_participants = []

for file in files:
    all_participants.append(np.load('./' + file,allow_pickle=True).tolist())
    print(file)
tf.logging.set_verbosity(tf.logging.ERROR)
maxim = -1
sentence_embeddings = []

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    for i in range(0, len(files)):
        print(len(files) - i, end='\r')
        x = session.run(embed(all_participants[i]))
        np.save('./individual_embeddings/' + files[i], x)

        if (maxim < x.shape[0]):
            maxim = x.shape[0]

print(maxim)