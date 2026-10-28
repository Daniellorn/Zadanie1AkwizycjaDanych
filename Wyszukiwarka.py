import re
import sys
import heapq
import string
import sqlite3
import numpy as np
from collections import Counter
from contextlib import closing


words_to_remove = {
    "i", "oraz", "a", "ale", "lecz", "lub", "albo", "bądź",
    "ani", "ni", "czy", "zaś", "natomiast", "jednak", "więc",
    "zatem", "toteż", "dlatego", "przeto", "czyli", "mianowicie",
    "bowiem", "ponieważ", "bo", "gdyż", "albowiem",
    "że", "żeby", "ażeby", "aby", "by", "iż", "aż",
    "choć", "chociaż", "aczkolwiek", "jakkolwiek", "mimo",
    "pomimo", "podczas", "zanim", "nim", "gdy",
    "kiedy", "ilekroć", "skoro", "jeśli", "jeżeli",
    "gdyby", "jeśliby", "jeżeliby", "chyba",
    "tak",
    "zarówno", "jak", "nie", "lecz", "także",
    "również", "bądź", "też",
    "tymczasem", "są", "jest", "to"
}

def CleanQuery(text):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', '')
    pattern = re.compile(f"[{re.escape(string.punctuation)}]")
    text = pattern.sub('', text)

    return text

def Read_docs_vectors(query_vector, db_path="Baza.db"):

    vectors = []

    with closing(sqlite3.connect(db_path)) as db,db:
        cursor = db.cursor()

        cursor.execute("SELECT * FROM document_vectors")

        query_vector_norm = np.linalg.norm(query_vector)

        for row in cursor:
            doc_id = row[0]
            filename = row[1]
            doc_vector = np.array(row[2:], dtype=np.float32)

            doc_norm = np.linalg.norm(doc_vector)

            if query_vector_norm == 0 or doc_norm == 0:
                dot_product= 0.0
            else:
                dot_product = np.dot(query_vector / query_vector_norm, doc_vector / doc_norm)

            vectors.append((doc_id, filename, dot_product))

        result = heapq.nlargest(5, vectors, key=lambda x: x[2])

    return  result

def Create_query_vector(query, db_path="Baza.db"):
    with closing(sqlite3.connect(db_path)) as db,db:

        rows = db.execute(
            "SELECT word, idf FROM word_idf"
        ).fetchall()

    idf_values = dict(rows)

    words = [
        word for word in query
        if word not in words_to_remove
    ]

    query_vector = {}
    counts = Counter(words)
    total_words = len(words)

    for word, idf in idf_values.items():
        tf = counts[word] / total_words if total_words != 0 else 0
        query_vector[word] = tf * idf

    result = dict(sorted(query_vector.items()))


    return np.array(list(result.values()), dtype=np.float32)

#ai
def PrintResult(result_vectors):
    print("\nBest matches:\n")
    print(f"{'No.':<5} {'ID':<38} {'Filename':<20} {'Score':>10}")
    print("-" * 78)

    for i, (doc_id, filename, score) in enumerate(result_vectors, start=1):
        print(f"{i:<5} {doc_id:<38} {filename:<20} {float(score):>10.4f}")



def main():

    words = ""
    query = []
    
    
    for arg in sys.argv[1:]:
        words += ' ' + arg
    
    words = CleanQuery(words)
    
    for word in words.split():
        if word not in words_to_remove:
            query.append(word)
    
    query_vector = Create_query_vector(query)
    docs = Read_docs_vectors(query_vector)
    
    PrintResult(docs)


if __name__ == "__main__":
    main()