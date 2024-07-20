import bisect
import collections
import json
import random
import heapq
import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

import data_loader
from data_loader import get_data


class TQDMPredictCallback(keras.callbacks.Callback):
    def __init__(self, custom_tqdm_instance=None, tqdm_cls=tqdm, **tqdm_params):
        super().__init__()
        self.tqdm_cls = tqdm_cls
        self.tqdm_progress = None
        self.prev_predict_batch = None
        self.custom_tqdm_instance = custom_tqdm_instance
        self.tqdm_params = tqdm_params

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        self.tqdm_progress.update(batch - self.prev_predict_batch)
        self.prev_predict_batch = batch

    def on_predict_begin(self, logs=None):
        self.prev_predict_batch = 0
        if self.custom_tqdm_instance:
            self.tqdm_progress = self.custom_tqdm_instance
            return

        total = self.params.get('steps')
        if total:
            total -= 1

        self.tqdm_progress = self.tqdm_cls(total=total, **self.tqdm_params)

    def on_predict_end(self, logs=None):
        if self.tqdm_progress and not self.custom_tqdm_instance:
            self.tqdm_progress.close()





physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

model = keras.models.load_model("model")
x, _, y = get_data("data/everything.csv", uniq_path="config/uniq.json", vs=True)


as_df = pd.read_csv("data/everything.csv").sample(frac=1)

#acc = model.evaluate(x, y)
#print("Accuracy:", acc[2])



with open("config/uniq.json", "r") as fp:
    json_data = json.load(fp)
    uniq_list = json_data["uniq"]

    lindices = []
    uindices = []
    for index in range(len(uniq_list)):
        if uniq_list[index].islower():
            lindices.append(index)
        if uniq_list[index].isupper():
            uindices.append(index)

    lower_vocab = lindices
    upper_vocab = uindices

WT = "DEEEIRTTNPVATEQYGSVSTNLQRGNR"
WT = [data_loader.uniq.index(i) for i in WT]
WT = WT# + [0] * (40 - len(WT))



def remove_padding(sequence_list):
    sequence_list = list(sequence_list)

    while len(sequence_list) > 0 and sequence_list[0] == 0:
        sequence_list.pop(0)
    while len(sequence_list) > 0 and sequence_list[-1] == 0:
        sequence_list.pop(-1)
    return sequence_list

def mutate(sequences, population_size, num_mutations):
    sequences = list(map(remove_padding, sequences))
    
    if len(sequences) == 0:
        print("E: 0 sequences")
        return []
    size_per_seq = population_size // len(sequences)
    for curr_sequence in sequences:

        insertion_count = size_per_seq // 2 + size_per_seq % 2
        subsitution_count = size_per_seq // 2

        #print(insertion_count)
        #print(subsitution_count)
        #quit()

        ny = 0

        sequence = list(curr_sequence)

        yield np.array(sequence + [0] * (40 - len(sequence)))

        for _ in range(subsitution_count):
            positions = random.sample(range(len(sequence)), random.choice(num_mutations))
            mutated = sequence.copy()
            for p in positions:
                mutated[p] = random.choice(upper_vocab)
            
            if len(mutated) == 40:
                yield np.array(mutated)
                ny += 1
            elif len(mutated) <= 40:
                yield np.array(mutated + [0] * (40 - len(mutated)))
                ny += 1
            else:
                pass
        
        sequence = list(sequence)
        for _ in range(insertion_count):
            mutated = sequence.copy()
            for p in range(random.choice(num_mutations)):
                position = random.randint(0, len(sequence))
                v = random.choice(lower_vocab)
                #mutated = mutated[:position] + v + mutated[position:]
                mutated.insert(position, v)
            if len(mutated) == 40:
                yield np.array(mutated)
                ny += 1
            elif len(mutated) <= 40:
                yield np.array(mutated + [0] * (40 - len(mutated)))
                ny += 1
            else:
                pass
        
        #print("\nNY", ny)
        #quit()



def batch(gen, batch_size):
    while True:
        out = []
        for _ in range(batch_size):
            try:
                out.append(next(gen))
            except StopIteration:
                yield np.array(out)
                return
        yield np.array(out)


class TopHeap:
    def __init__(self, size):
        self.size = size
        self.sequences = []
        self.values = []
        self.hashes = set()
    
    #@profile
    def insert(self, sequence, value):
        if len(self.values) < self.size:
            self.sequences.append(sequence)
            self.values.append(value)
            return
        lower_index = None

        check = True
        if hash(tuple(sequence)) not in self.hashes:
            check = False
            

        for cmp_index, (cmp_sequence, cmp_value) in enumerate(zip(self.sequences, self.values)):
            if check:
                if (cmp_sequence == sequence).all():
                    return
                
            if cmp_value < value:
                lower_index = cmp_index
        
        if lower_index != None:
            #self.data[lower_index] = (sequence, value)
            self.sequences[lower_index] = sequence
            self.values[lower_index] = value
            self.hashes.add(hash(tuple(sequence)))



#@profile
def get_best(population, pop_size, num, yield_num=128, ):
    best = TopHeap(num)
    
    seen = 0

    with tqdm(total=pop_size) as pbar:
        while True:
            try:
                sequences = next(population)
                seen += len(sequences)
                pbar.update(len(sequences))

                outs = model.predict(sequences)

                for s, p in zip(sequences, outs):
                    best.insert(s, p[0])
            except StopIteration:
                break
    
    return best, seen




def best_of_mutation(df, mutation_count, number):
    if mutation_count == None:
        best = df[df.num_mutations == mutation_count].sort_values("viral_selection")[::-1][:number]
    best = df.sort_values("viral_selection")[::-1][:number]
    sequences = np.array([[data_loader.uniq.index(j) for j in i] + [0] * (40 - len(i)) for i in best.sequence])

    return sequences


def search_mut(df, mutation_count, seed_size, pop_size, cutoff_size, yield_num=128):
    x = best_of_mutation(df, mutation_count, seed_size)

    population = batch(mutate(x, pop_size, [1, 2, 3]), yield_num)

    best, num_tested = get_best(population, pop_size, cutoff_size, yield_num=yield_num)

    #best = list(zip(best.sequences, best.values))

    #best_best = sorted(best, key=lambda i: i[1], reverse=True)

    #return best_best[:50]
    return best, num_tested



BATCH_SIZE = 128


bests = [None for i in range(20)]

SEED_POP = 1000
GEN_POP = 25000


first_round = True

for sequence in as_df.sequence:
    if len(sequence) > 40:
        continue
    prediction = model.predict(np.array([[data_loader.uniq.index(i) for i in sequence] + [0] * (40 - len(sequence))]))[0][0]
    best_vs = 0

    pop = batch(mutate([
        [data_loader.uniq.index(i) for i in sequence] + [0] * (40 - len(sequence))
    ], GEN_POP, [1, 2, 3]), BATCH_SIZE)
    new_best, num_tested = get_best(pop, GEN_POP, SEED_POP, BATCH_SIZE)
    #bests[mutation_count] = new_best

    avg_vs = sum(new_best.values) / len(new_best.values)
    max_vs_index = max(range(len(new_best.values)), key=new_best.values.__getitem__)
    min_vs_index = min(range(len(new_best.values)), key=new_best.values.__getitem__)

    max_vs_sequence = ("".join(data_loader.uniq[i] for i in new_best.sequences[max_vs_index])).strip()
    min_vs_sequence = ("".join(data_loader.uniq[i] for i in new_best.sequences[min_vs_index])).strip()

    max_vs_vs = new_best.values[max_vs_index]

    best_vs = max(max_vs_vs, best_vs)

    min_vs_vs = new_best.values[min_vs_index]
    
    #print(f"Looked at {num_tested} sequences (250000)")
    #print(f"Best Sequence: {max_vs_sequence}, Viral Selection: {max_vs_vs}")
    #print(f"Worst Sequence: {min_vs_sequence}, Viral Selection: {min_vs_vs}")
    #print(f"Average Viral Selection: {avg_vs}")

    #print(f"Best: {len(new_best.sequences)} (50)")
    best_sequences = new_best.sequences

    best_str_sequences = ["".join([data_loader.uniq[j] for j in i]).strip() for i in new_best.sequences]
    best_virals = new_best.values

    dfdf = pd.DataFrame({"sequence": best_str_sequences, "pred": best_virals})
    dfdf = dfdf.sort_values("pred")[::-1]

    #dfdf.to_csv(f"designed_sequences/{round_num}_{mutation_count}.csv", index=False)

    print(sequence)
    print(prediction, best_vs)






ROUNDS = 100
INITIAL_POPULATION = 37 * 1000 * 1000
KEPT_POPULATION = 100 * 1000 #1000 * 10 # 10 thousand


"""
number of mutations to search = 16
seed per mutation = 1000
population per sequence = 250
population per mutation = 250,000


best per mutation = 50



"""

best_sequences = [WT]
for r in range(ROUNDS):
    print("\n")
    print(f"Mutating from {len(best_sequences)} sequences")
    population = batch(mutate(best_sequences, INITIAL_POPULATION, [1, 2, 3, 4, 5]), BATCH_SIZE)#mutate_to(best_sequences, INITIAL_POPULATION, yield_num=BATCH_SIZE)

    new_best, num_tested = get_best(population, INITIAL_POPULATION, KEPT_POPULATION, yield_num=BATCH_SIZE)

    if len(new_best.values) == 0:
        quit()
    avg_vs = sum(new_best.values) / len(new_best.values)
    max_vs_index = max(range(len(new_best.values)), key=new_best.values.__getitem__)
    min_vs_index = min(range(len(new_best.values)), key=new_best.values.__getitem__)

    max_vs_sequence = ("".join(data_loader.uniq[i] for i in new_best.sequences[max_vs_index])).strip()
    min_vs_sequence = ("".join(data_loader.uniq[i] for i in new_best.sequences[min_vs_index])).strip()

    max_vs_vs = new_best.values[max_vs_index]
    min_vs_vs = new_best.values[min_vs_index]
    
    print(f"Directed Evolution Round {r}:")
    print(f"Looked at {num_tested} sequences ({INITIAL_POPULATION})")
    print(f"Best Sequence: {max_vs_sequence}, Viral Selection: {max_vs_vs}")
    print(f"Worst Sequence: {min_vs_sequence}, Viral Selection: {min_vs_vs}")
    print(f"Average Viral Selection: {avg_vs}")

    print(f"Best: {len(new_best.sequences)} ({KEPT_POPULATION})")
    best_sequences = new_best.sequences

