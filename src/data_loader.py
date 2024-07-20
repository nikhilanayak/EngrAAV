#from bert import BertModelLayer
from tensorflow import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tqdm import tqdm
import math
from sklearn.utils import shuffle
from sklearn import preprocessing
import random
import json
import os

WILD_BEFORE = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMIT"
WILD_AFTER = "QAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"

uniq = []

MAX_SEQ_LEN = 40

def data_augmentation(sequence):
    CIRCULAR = len(sequence)

    news = []
    this = sequence

    this = list(this)

    news.append("".join(this))
    for pos in range(1, CIRCULAR):
        last = this.pop(-1)
        this = list(last) + this
        news.append("".join(this))

    return news


def eval_get(sequence):
    X = np.array([uniq.index(char) for char in sequence.ljust(MAX_SEQ_LEN, " ")])
    return X

def get_data(path, val=False, rand=True, augment=False, mutation=None, vs=False, type=None, save_mutation=False):
    global uniq, MAX_SEQ_LEN

    data = pd.read_csv(path, low_memory=False)
    if type != None:
        if type == "upper":
            data = data[data.mutation_sequence.str.strip().str.isupper()] 
        elif type == "lower":
            data = data[data.mutation_sequence.str.strip().str.islower()] 
    if mutation != None:
        data = data[data.num_mutations == mutation]
    
    data = data[["sequence", "is_viable", "num_mutations", "viral_selection"]]
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if rand:
        data = data.sample(frac=1)

    data = data[~data.sequence.str.contains("\*")]
    data = data[data.sequence.str.len() <= MAX_SEQ_LEN]

    sequences = data.sequence
    vibs = data.is_viable if not val else None

    more_sequences = []
    more_vibs = []
    if not val and augment:
        for seq, vib in tqdm(zip(sequences,vibs)):
            augmented = data_augmentation(seq)
            for new_sequence in augmented:
                more_sequences.append(new_sequence)
                #print(len(augmented))
                #_ = input()
                more_vibs.append(vib)
    
        more_sequences = np.array(more_sequences)
        more_vibs = np.array(more_vibs)
    else:
        more_sequences = sequences
        more_vibs = vibs

    if os.path.exists("config/uniq.json"):
        config = json.load(open("config/uniq.json"))
        MAX_SEQ_LEN = config["MAX"]
        uniq = config["uniq"]
    else:
        uniq.append(" ")
        for i in tqdm(sequences):
            for letter in i:
                if letter not in uniq:
                    uniq.append(letter)
        uniq = list(uniq)
        with open("config/uniq.json", "w") as fp:
            json.dump({"uniq": uniq, "MAX": MAX_SEQ_LEN}, fp)            

    X_seq = np.array([
        ([uniq.index(char) for char in seq] + [0] * (40 - len(seq)))for seq in tqdm(more_sequences)
    ])

    if val:
        if rand:
            return shuffle(X_seq, random_state=0)
        else:
            return X_seq

    Y = []
    for i in more_vibs:
        if i:
            Y.append([0, 1])
        else:
            Y.append([1, 0])
    Y = np.array(Y)
    
    if rand:
        X_seq, Y, viral = shuffle(X_seq, Y, data.viral_selection, random_state=0)
    else:
        viral = data.viral_selection

    returning = [X_seq, Y]

    if vs:
        returning.append(viral)

    if save_mutation:
        returning.append(data.mutation_sequence.apply(lambda i: i.replace("_", "") + str(i.find(next(filter(str.isalpha, i))))))

    return (*returning,)