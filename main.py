import re
import pandas as pd
import numpy as np

def get_data():
    doc_num = input().strip()
    docs = []
    for i in range(doc_num+2):
        doc = input().strip()
        docs.append(doc)
    return docs

docs = get_data()

all_docs = docs[:-1]
k = int(docs[-1])
all_docs = [re.sub(r"[^\w\s]", "", doc.lower()).split() for doc in all_docs]

uniq_terms = set()
for doc in all_docs:
    for term in doc:
        uniq_terms.add(term)

rows = []
for term in uniq_terms:
    result = {}
    for doc_index, doc in enumerate(all_docs):
        result[f"doc_{doc_index}"] = 1 if term in doc else 0
    rows.append(result)

C = pd.DataFrame(rows).to_numpy()
U, Sigma, VT = np.linalg.svd(C[...,:-1].T, full_matrices=False)
k = 2
U_k = U[:, :k]
Sigma_k = np.diag(Sigma[:k])
VT_k = VT[:k, :]
query = C[..., -1] @ VT_k.T @ np.linalg.inv(Sigma_k)
docs_reduced = U_k @ Sigma_k
print([round(np.dot(query, doc_reduced) / (np.linalg.norm(query) * np.linalg.norm(doc_reduced)), 2) for doc_reduced in docs_reduced])
