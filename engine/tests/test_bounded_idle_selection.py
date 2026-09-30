import unittest

from omni_core.bounded_idle_selection import select_idle_workspace


class BoundedIdleTests(unittest.TestCase):
    def test_exact_stable_selection_visits_entire_corpus(self):
        visited = 0
        def records():
            nonlocal visited
            for index in range(100000):
                visited += 1
                yield {"id": str(index), "score": index % 37}
        actual = select_idle_workspace(records(), capacity=7,
                                       score=lambda row: row["score"], eligible=lambda _row: True)
        expected = [str(index) for index in range(100000) if index % 37 == 36][:7]
        self.assertEqual([row["id"] for row in actual], expected)
        self.assertEqual(visited, 100000)

    def test_resource_refusal_and_eligibility_are_not_learning_caps(self):
        records = [{"id": str(index), "score": index} for index in range(30)]
        actual = select_idle_workspace(records, capacity=4, score=lambda row: row["score"],
                                       eligible=lambda row: int(row["id"]) % 2 == 0)
        self.assertEqual([row["id"] for row in actual], ["28", "26", "24", "22"])
        with self.assertRaises(RuntimeError):
            select_idle_workspace(records, capacity=4, score=lambda row: row["score"],
                                  eligible=lambda _row: True, reserve=lambda _size: False)


if __name__ == "__main__":
    unittest.main()
