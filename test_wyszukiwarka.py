import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np

# ZMIEŃ na nazwę swojego pliku z wyszukiwarką (bez .py)
import Wyszukiwarka


class TestCleanQuery(unittest.TestCase):
    def test_lowercase_and_punctuation(self):
        self.assertEqual(Wyszukiwarka.CleanQuery("Ala, MA Kota!"), "ala ma kota")

    def test_newlines_replaced_with_space(self):
        self.assertEqual(Wyszukiwarka.CleanQuery("kot\npies"), "kot pies")

    def test_empty_string(self):
        self.assertEqual(Wyszukiwarka.CleanQuery(""), "")


class DbTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "Baza.db")

    def tearDown(self):
        self.temp_dir.cleanup()


class TestCreateQueryVector(DbTestCase):
    def setUp(self):
        super().setUp()
        # Celowo nie w kolejności alfabetycznej
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("CREATE TABLE word_idf (word TEXT PRIMARY KEY, idf REAL NOT NULL)")
            conn.executemany(
                "INSERT INTO word_idf VALUES (?, ?)",
                [("pies", 4.0), ("kot", 2.0)],
            )

    def test_vector_sorted_alphabetically_with_tfidf(self):
        # kot: tf = 2/3, idf = 2.0 -> 4/3; pies: tf = 1/3, idf = 4.0 -> 4/3
        vector = Wyszukiwarka.Create_query_vector(["kot", "pies", "kot"], self.db_path)

        self.assertEqual(vector.shape, (2,))
        np.testing.assert_allclose(vector, [4 / 3, 4 / 3], rtol=1e-5)

    def test_stop_words_are_ignored(self):
        with_stop = Wyszukiwarka.Create_query_vector(["kot", "nie", "oraz"], self.db_path)
        without_stop = Wyszukiwarka.Create_query_vector(["kot"], self.db_path)

        np.testing.assert_allclose(with_stop, without_stop)

    def test_unknown_word_gives_zero_vector(self):
        vector = Wyszukiwarka.Create_query_vector(["nieznane"], self.db_path)

        np.testing.assert_allclose(vector, [0.0, 0.0])

    def test_empty_query_gives_zero_vector(self):
        vector = Wyszukiwarka.Create_query_vector([], self.db_path)

        np.testing.assert_allclose(vector, [0.0, 0.0])


class TestReadDocsVectors(DbTestCase):
    def _create_docs(self, rows):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                'CREATE TABLE "document_vectors" '
                '("ID" TEXT, "NAZWA DOKUMENTU" TEXT, "kot" REAL, "pies" REAL)'
            )
            conn.executemany(
                'INSERT INTO "document_vectors" VALUES (?, ?, ?, ?)', rows
            )

    def test_best_match_is_first(self):
        self._create_docs([
            ("id-1", "a.txt", 1.0, 0.0),
            ("id-2", "b.txt", 0.0, 1.0),
        ])

        result = Wyszukiwarka.Read_docs_vectors(np.array([1.0, 0.0], dtype=np.float32), self.db_path)

        self.assertEqual(result[0][0], "id-1")
        self.assertEqual(result[0][1], "a.txt")
        self.assertAlmostEqual(float(result[0][2]), 1.0)
        self.assertEqual(result[1][0], "id-2")
        self.assertAlmostEqual(float(result[1][2]), 0.0)

    def test_returns_at_most_five_results(self):
        self._create_docs([
            (f"id-{i}", f"{i}.txt", float(i), 1.0) for i in range(7)
        ])

        result = Wyszukiwarka.Read_docs_vectors(np.array([1.0, 0.0], dtype=np.float32), self.db_path)

        self.assertEqual(len(result), 5)
        self.assertEqual([r[0] for r in result], ["id-6", "id-5", "id-4", "id-3", "id-2"])

    def test_empty_table_returns_empty_list(self):
        self._create_docs([])

        result = Wyszukiwarka.Read_docs_vectors(np.array([1.0, 0.0], dtype=np.float32), self.db_path)

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)