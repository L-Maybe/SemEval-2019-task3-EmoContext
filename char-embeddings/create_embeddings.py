import numpy as np
import os

file_path = "G:/vector/glove.840B.300d.txt"

vectors = {}
with open(file_path, 'rb') as f:
    for line in f:
        line = line.decode('utf8').strip()
        line_split = line.split(" ")
        vec = np.array(line_split[1:], dtype=float)
        word = line_split[0]

        for char in word:
            if ord(char) < 128:
                if char in vectors:
                    print(char)
                    print(np.shape(vectors[char][0]))
                    print(np.shape(vec))
                    vectors[char] = (vectors[char][0] + vec, vectors[char][1] + 1)
                else:
                    vectors[char] = (vec, 1)

base_name = os.path.splitext(os.path.basename(file_path))[0] + '-char.txt'
with open(base_name, 'wb') as f2:
    for word in vectors:
        avg_vector = np.round(
            (vectors[word][0] / vectors[word][1]), 6).tolist()
        f2.write(word + " " + " ".join(str(x) for x in avg_vector) + "\n")
