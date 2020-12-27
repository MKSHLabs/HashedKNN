import numpy as np
from tqdm import tqdm

class HashedKNN:
  def __init__(self,bucket_size,n_universes):
      self.bucket_size = bucket_size
      self.n_universes=n_universes
      self.hash_tables=[]
      self.id_tables=[]

  def nearest_neighbor(self,vector, candidates, k=1):
      similarity = []
      for row in candidates:
          similarity.append(np.dot(vector,row)/(np.linalg.norm(vector)*np.linalg.norm(row)))
      sorted_ids = np.argsort(similarity)
      return sorted_ids[-k:]

  def hash_value_of_vector(self,vector,planes):
      h = np.dot(vector,planes)>=0
      h = np.squeeze(h)
      hash_value = 0
      for i in range(planes.shape[1]):
          hash_value += np.power(2,i)*h[i]

      hash_value = int(hash_value)
      return hash_value

  def make_hash_table(self, corpus,planes):
    hash_table = {i:[] for i in range(2**planes.shape[1])}
    id_table = {i:[] for i in range(2**planes.shape[1])}
    for i, v in enumerate(corpus):
        h = self.hash_value_of_vector(v,planes)
        hash_table[h].append(v)
        id_table[h].append(i)
    return hash_table, id_table

  def knn(self,doc_id, v, k=1):
    vecs_to_consider = []
    ids_to_consider = []
    ids_to_consider_set = set()

    for universe_id in range(self.n_planes):
        planes = self.planes[universe_id]
        hash_value = self.hash_value_of_vector(v, planes)
        hash_table = self.hash_tables[universe_id]
        document_vectors = hash_table[hash_value]
        id_table = self.id_tables[universe_id]
        new_ids_to_consider = id_table[hash_value]
        if doc_id in new_ids_to_consider:
            new_ids_to_consider.remove(doc_id)
        for i, new_id in enumerate(new_ids_to_consider):
            if new_id not in ids_to_consider_set:
                document_vector_at_i = document_vectors[i]
                vecs_to_consider.append(document_vector_at_i)
                ids_to_consider.append(new_id)
                ids_to_consider_set.add(new_id)
    print("Considering %d vecs" % len(vecs_to_consider))
    vecs_to_consider_arr = np.array(vecs_to_consider)
    nearest_neighbor_idx = self.nearest_neighbor(v, vecs_to_consider_arr, k=k)
    return [ids_to_consider[idx] for idx in nearest_neighbor_idx]

  def fit(self,corpus):
      self.n_planes=int(np.log2(corpus.shape[0]/self.bucket_size))
      self.planes=[np.random.normal(size=(corpus.shape[1], self.n_planes)) for _ in range(self.n_planes)]

      if self.n_universes>self.n_planes:
        self.n_universes=self.n_planes

      for universe_id in tqdm(range(self.n_universes)):
          planes = self.planes[universe_id]
          hash_table, id_table = self.make_hash_table(corpus,planes)
          self.hash_tables.append(hash_table)
          self.id_tables.append(id_table)

  def find(self,vectorIdx,corpus,k=1):
      return self.knn(vectorIdx, corpus[vectorIdx], k=k)
