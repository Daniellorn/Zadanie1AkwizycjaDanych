import math
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from contextlib import closing

import Indekser


class TestCleanText(unittest.TestCase):
    def test_lowercase_punctuation_and_basename(self):
        result = Indekser.CleanText(
            ("doc-1", "/folder/test.txt", "ALA, ma Kota!\nDrugi wiersz.")
        )

        self.assertEqual(result, ("doc-1", "test.txt", "ala ma kota drugi wiersz"))

    def test_empty_text(self):
        self.assertEqual(
            Indekser.CleanText(("doc-2", "pusty.txt", "")),
            ("doc-2", "pusty.txt", ""),
        )


class TestSaveToSQLite(unittest.TestCase):
    def setUp(self):
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "Baza.db")
        os.chdir(self.temp_dir.name)

    def tearDown(self):
        os.chdir(self.original_cwd)
        self.temp_dir.cleanup()

    def test_saves_document_vectors_and_zero_for_missing_words(self):
        documents = [
            (("id-1", "a.txt"), {"kot": 0.5, "pies": 0.2}),
            (("id-2", "b.txt"), {"kot": 0.1}),
        ]

        Indekser.save_to_sqlite(documents)

        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            cursor = conn.cursor()
            cursor.execute('PRAGMA table_info("document_vectors")')
            columns = [row[1] for row in cursor.fetchall()]

            self.assertEqual(
                columns,
                ["ID", "NAZWA DOKUMENTU", "kot", "pies"],
            )

            cursor.execute(
                'SELECT "ID", "NAZWA DOKUMENTU", "kot", "pies" '
                'FROM "document_vectors" ORDER BY "ID"'
            )
            rows = cursor.fetchall()

        self.assertEqual(rows, [
            ("id-1", "a.txt", 0.5, 0.2),
            ("id-2", "b.txt", 0.1, 0.0),
        ])

    def test_empty_documents_creates_table(self):
        Indekser.save_to_sqlite([])

        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            cursor = conn.cursor()
            cursor.execute('PRAGMA table_info("document_vectors")')
            columns = [row[1] for row in cursor.fetchall()]
            cursor.execute('SELECT COUNT(*) FROM "document_vectors"')
            count = cursor.fetchone()[0]

        self.assertEqual(columns, ["ID", "NAZWA DOKUMENTU"])
        self.assertEqual(count, 0)


class TestSaveIdfToSQLite(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "Baza.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_saves_idf_values(self):
        idf_items = [("kot", 0.0), ("pies", math.log(2))]

        Indekser.save_idf_to_sqlite(idf_items, self.db_path)

        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            rows = conn.execute(
                "SELECT word, idf FROM word_idf ORDER BY word"
            ).fetchall()

        self.assertEqual(rows[0], ("kot", 0.0))
        self.assertEqual(rows[1][0], "pies")
        self.assertAlmostEqual(rows[1][1], math.log(2))

    def test_replaces_previous_idf_table_contents(self):
        Indekser.save_idf_to_sqlite([("stare", 1.0)], self.db_path)
        Indekser.save_idf_to_sqlite([("nowe", 2.0)], self.db_path)

        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            rows = conn.execute(
                "SELECT word, idf FROM word_idf"
            ).fetchall()

        self.assertEqual(rows, [("nowe", 2.0)])


class TestTfIdfMath(unittest.TestCase):
    def test_tf_and_idf_for_small_corpus(self):
        # Dokument 1: "kot kot pies"; dokument 2: "kot dom"
        docs = [
            ["kot", "kot", "pies"],
            ["kot", "dom"],
        ]
        n_documents = len(docs)

        document_frequency = {}
        for words in docs:
            for word in set(words):
                document_frequency[word] = document_frequency.get(word, 0) + 1

        idf = {
            word: math.log(n_documents / df)
            for word, df in document_frequency.items()
        }

        tf_doc1_kot = docs[0].count("kot") / len(docs[0])
        tfidf_doc1_kot = tf_doc1_kot * idf["kot"]

        self.assertAlmostEqual(tf_doc1_kot, 2 / 3)
        self.assertEqual(document_frequency["kot"], 2)
        self.assertAlmostEqual(idf["kot"], 0.0)
        self.assertAlmostEqual(idf["pies"], math.log(2))
        self.assertAlmostEqual(tfidf_doc1_kot, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
