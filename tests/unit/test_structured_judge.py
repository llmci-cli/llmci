"""Tests for the structured-output (JSON Schema) judge and validator."""

import pytest

from llmci.errors import ConfigError
from llmci.judges.factory import create_judge
from llmci.judges.structured import StructuredJudge, validate_schema
from llmci.models import JudgeConfig

OBJ_SCHEMA = {
    "type": "object",
    "required": ["id", "name", "price"],
    "additionalProperties": False,
    "properties": {
        "id": {"type": "integer"},
        "name": {"type": "string", "minLength": 1},
        "price": {"type": "number", "minimum": 0},
    },
}


class TestValidator:
    def test_valid_object_has_no_errors(self):
        assert validate_schema({"id": 1, "name": "x", "price": 9.5}, OBJ_SCHEMA) == []

    def test_missing_required(self):
        errors = validate_schema({"id": 1, "price": 9.5}, OBJ_SCHEMA)
        assert any("missing required property 'name'" in e for e in errors)

    def test_wrong_type(self):
        errors = validate_schema({"id": 1, "name": "x", "price": "free"}, OBJ_SCHEMA)
        assert any("price" in e and "number" in e for e in errors)

    def test_additional_properties_rejected(self):
        errors = validate_schema(
            {"id": 1, "name": "x", "price": 1, "extra": 9}, OBJ_SCHEMA
        )
        assert any("unexpected properties" in e for e in errors)

    def test_bool_is_not_integer(self):
        assert validate_schema(True, {"type": "integer"})
        assert validate_schema(True, {"type": "boolean"}) == []

    def test_number_accepts_int_and_float(self):
        assert validate_schema(3, {"type": "number"}) == []
        assert validate_schema(3.5, {"type": "number"}) == []

    def test_enum(self):
        assert validate_schema("a", {"enum": ["a", "b"]}) == []
        assert validate_schema("z", {"enum": ["a", "b"]})

    def test_string_constraints(self):
        schema = {"type": "string", "minLength": 2, "maxLength": 4, "pattern": "^a"}
        assert validate_schema("ab", schema) == []
        assert validate_schema("a", schema)  # too short
        assert validate_schema("xyz", schema)  # pattern mismatch

    def test_number_bounds(self):
        schema = {"type": "number", "minimum": 0, "maximum": 10}
        assert validate_schema(5, schema) == []
        assert validate_schema(-1, schema)
        assert validate_schema(11, schema)

    def test_array_items_and_length(self):
        schema = {"type": "array", "minItems": 1, "items": {"type": "integer"}}
        assert validate_schema([1, 2], schema) == []
        assert validate_schema([], schema)  # too few
        assert validate_schema([1, "x"], schema)  # bad item type

    def test_list_of_types(self):
        assert validate_schema(None, {"type": ["string", "null"]}) == []
        assert validate_schema("x", {"type": ["string", "null"]}) == []
        assert validate_schema(5, {"type": ["string", "null"]})


class TestJudge:
    async def test_valid_scores_one(self):
        judge = StructuredJudge(OBJ_SCHEMA)
        res = await judge.evaluate_single("", "", '{"id": 1, "name": "x", "price": 2}')
        assert res.score == 1.0

    async def test_invalid_json_scores_zero(self):
        judge = StructuredJudge(OBJ_SCHEMA)
        res = await judge.evaluate_single("", "", "not json{")
        assert res.score == 0.0
        assert "invalid JSON" in res.reason

    async def test_schema_violation_scores_zero(self):
        judge = StructuredJudge(OBJ_SCHEMA)
        res = await judge.evaluate_single("", "", '{"id": 1}')
        assert res.score == 0.0
        assert "name" in res.reason

    async def test_partial_credit(self):
        judge = StructuredJudge(OBJ_SCHEMA, partial_credit=True)
        # 2 of 3 required fields valid (price missing) -> 2/3.
        res = await judge.evaluate_single("", "", '{"id": 1, "name": "x"}')
        assert res.score == pytest.approx(2 / 3)

    async def test_partial_credit_counts_invalid_field(self):
        judge = StructuredJudge(OBJ_SCHEMA, partial_credit=True)
        # price present but wrong type -> only id, name count -> 2/3.
        res = await judge.evaluate_single(
            "", "", '{"id": 1, "name": "x", "price": "free"}'
        )
        assert res.score == pytest.approx(2 / 3)


class TestFactory:
    def test_inline_schema(self):
        judge = create_judge(JudgeConfig(type="structured", json_schema=OBJ_SCHEMA))
        assert isinstance(judge, StructuredJudge)

    def test_missing_schema_raises(self):
        with pytest.raises(ConfigError, match="requires a 'json_schema'"):
            create_judge(JudgeConfig(type="structured"))

    def test_schema_from_file(self, tmp_path):
        import json

        p = tmp_path / "schema.json"
        p.write_text(json.dumps(OBJ_SCHEMA))
        judge = create_judge(JudgeConfig(type="structured", json_schema=str(p)))
        assert isinstance(judge, StructuredJudge)

    def test_missing_file_raises(self):
        with pytest.raises(ConfigError, match="schema file not found"):
            create_judge(JudgeConfig(type="structured", json_schema="/no/such.json"))

    def test_partial_credit_flag_passthrough(self):
        judge = create_judge(
            JudgeConfig(type="structured", json_schema=OBJ_SCHEMA, partial_credit=True)
        )
        assert isinstance(judge, StructuredJudge)
        assert judge.partial_credit is True
