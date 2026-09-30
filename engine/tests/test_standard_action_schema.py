"""Schema structures only; no cortex, forward, model or training."""
import unittest
from unittest.mock import patch

from omni_core.native_action_protocol import structural_schema, validate_structural_value


class StandardActionSchemaTests(unittest.TestCase):
    def test_recursive_ref_keeps_structure_and_validates_actual_nested_arguments(self):
        raw = {"$defs": {"node": {"type": "object", "properties": {
            "name": {"type": "string"}, "children": {"type": "array", "items": {"$ref": "#/$defs/node"}}},
            "required": ["name"]}}, "$ref": "#/$defs/node"}
        schema = structural_schema(raw)
        self.assertEqual(schema["$ref"], raw["$ref"])
        self.assertTrue(validate_structural_value({"name": "root", "children": [{"name": "leaf"}]}, schema))
        self.assertFalse(validate_structural_value({"name": "root", "children": [{"name": 7}]}, schema))

    def test_deeper_than_old_product_cap_is_not_rejected(self):
        raw, value = {"type": "integer"}, 5
        for _ in range(60):
            raw = {"type": "object", "properties": {"child": raw}, "required": ["child"]}
            value = {"child": value}
        self.assertTrue(validate_structural_value(value, structural_schema(raw)))

    def test_pattern_dependency_condition_and_unevaluated_constraints_are_retained(self):
        raw = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "patternProperties": {"^x_": {"type": "integer", "minimum": 0}},
            "properties": {"enabled": {"type": "boolean"}, "name": {"type": "string", "pattern": "^[A-Z]+$"}},
            "dependentRequired": {"enabled": ["name"]}, "unevaluatedProperties": False}
        schema = structural_schema(raw)
        self.assertTrue(validate_structural_value({"enabled": True, "name": "OK", "x_n": 2}, schema))
        self.assertFalse(validate_structural_value({"enabled": True}, schema))
        self.assertFalse(validate_structural_value({"name": "lower"}, schema))
        self.assertFalse(validate_structural_value({"x_n": -1}, schema))
        self.assertFalse(validate_structural_value({"untyped": 2}, schema))

    def test_const_literals_and_property_names_are_not_stripped_as_persona_prose(self):
        raw = {"type": "object", "description": "annotation prose", "properties": {
            "description": {"const": {"description": "a literal value", "title": "literal"}}}}
        schema = structural_schema(raw)
        self.assertNotIn("description", schema)
        self.assertIn("description", schema["properties"])
        self.assertTrue(validate_structural_value({"description": {"description": "a literal value", "title": "literal"}}, schema))
        self.assertFalse(validate_structural_value({"description": {}}, schema))

    def test_boolean_schemas_and_numeric_guards(self):
        self.assertTrue(validate_structural_value({"any": "actual"}, structural_schema(True)))
        self.assertFalse(validate_structural_value({}, structural_schema(False)))
        number = structural_schema({"type": "number"})
        self.assertFalse(validate_structural_value(float("nan"), number))
        self.assertFalse(validate_structural_value(float("inf"), number))
        self.assertFalse(validate_structural_value(True, number))

    def test_declared_old_dialect_uses_its_real_array_tuple_semantics(self):
        schema = structural_schema({"$schema": "http://json-schema.org/draft-07/schema#", "type": "array",
            "items": [{"type": "integer"}, {"type": "string"}], "additionalItems": False})
        self.assertTrue(validate_structural_value([2, "actual"], schema))
        self.assertFalse(validate_structural_value([2, 3], schema))
        self.assertFalse(validate_structural_value([2, "actual", 4], schema))

    def test_unprovided_external_schema_cannot_fetch_network_or_read_local_files(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("implicit schema network access")):
            self.assertFalse(validate_structural_value({}, structural_schema({"$ref": "https://example.invalid/schema.json"})))
            self.assertFalse(validate_structural_value({}, structural_schema({"$ref": "file:///private/user-data.json"})))


if __name__ == "__main__": unittest.main()
