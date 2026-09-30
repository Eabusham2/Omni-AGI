"""Parser-only native views; no brain, model, training or huge corpus."""
import json
import unittest
from unittest.mock import patch

import pyarrow as arrow
from omni_core import datasets


class LazyNestedColumnarTests(unittest.TestCase):
    def row(self, value):
        return next(datasets._bounded_columnar_rows(arrow.RecordBatch.from_pylist([value]), arrow))

    def test_nested_children_preserve_exact_json_without_whole_struct_or_list_conversion(self):
        value = {"details": {"items": [1, None, 3], "more": {"unicode": "中文🙂"}}}
        row = self.row(value)
        self.assertIsInstance(row["details"], datasets.ColumnarMapping)
        self.assertIsInstance(row["details"]["items"], datasets.ColumnarSequence)
        self.assertEqual(json.loads("".join(datasets._bounded_json_encoding(row))), value)

    def test_nested_large_string_uses_native_view_and_complete_leased_learning_text(self):
        text = "中文🙂\n" * 12_000
        row = self.row({"details": {"items": [text, "tail"]}})
        self.assertIsInstance(row["details"]["items"][0], datasets.ColumnarTextValue)
        payload, _, rejection = datasets._training_value(row)
        try:
            self.assertIsNone(rejection)
            complete = "".join(piece for piece, _ in payload.windows())
            self.assertEqual(json.loads(complete), {"details": {"items": [text, "tail"]}})
        finally: payload.close()

    def test_typed_dialogue_keeps_all_nested_targets_in_a_lease(self):
        row = self.row({"messages": [{"role": "user", "content": "a human question"},
                                     {"role": "assistant", "content": "the observed answer"}]})
        payload, provenance, rejection = datasets._training_value(row)
        try:
            self.assertIsNone(rejection)
            complete = "".join(piece for piece, _ in payload.windows())
            self.assertIn("a human question", complete)
            self.assertIn("the observed answer", complete)
            self.assertEqual(provenance["format"], "typed-dialogue")
        finally: payload.close()

    def test_binary_literal_matches_previous_semantics_in_bounded_pieces(self):
        for value in (b"plain", b"a'b", b'a"b', b'a\'"\\\n\x00\xff'):
            row = self.row({"blob": value})
            encoded = json.loads("".join(datasets._bounded_json_encoding(row)))
            self.assertEqual(encoded["blob"], str(value))

    def test_map_dictionary_and_null_views_visit_every_value(self):
        mapping = arrow.scalar([("one", 1), ("two", None)], type=arrow.map_(arrow.string(), arrow.int64()))
        view = datasets._columnar_value(mapping, arrow)
        self.assertEqual(json.loads("".join(datasets._bounded_json_encoding(view))), [["one", 1], ["two", None]])
        dictionary = arrow.DictionaryArray.from_arrays([0], ["value"])[0]
        self.assertEqual(datasets._columnar_value(dictionary, arrow), "value")
        self.assertIsNone(datasets._columnar_value(arrow.scalar(None), arrow))


if __name__ == "__main__": unittest.main()
