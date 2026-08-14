"""Tests for core/tool_exec.py."""

import asyncio
import pytest

from vulcan.core.tool_exec import normalize_schema, clean_for_json


# ── Helpers ────────────────────────────────────────────────────────────────────

def run(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)


# ── normalize_schema: type alias conversions ──────────────────────────────────

class TestNormalizeSchemaTypeConversions:
    def test_dict_becomes_object(self):
        func = {"parameters": {"type": "dict"}}
        normalize_schema(func)
        assert func["parameters"]["type"] == "object"

    def test_float_becomes_number(self):
        func = {"parameters": {"type": "float"}}
        normalize_schema(func)
        assert func["parameters"]["type"] == "number"

    def test_int_becomes_integer(self):
        func = {"parameters": {"type": "int"}}
        normalize_schema(func)
        assert func["parameters"]["type"] == "integer"

    def test_nested_dict_type_converted(self):
        func = {
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "dict"}
                }
            }
        }
        normalize_schema(func)
        assert func["parameters"]["properties"]["data"]["type"] == "object"

    def test_nested_float_type_converted(self):
        func = {
            "parameters": {
                "properties": {
                    "amount": {"type": "float"}
                }
            }
        }
        normalize_schema(func)
        assert func["parameters"]["properties"]["amount"]["type"] == "number"

    def test_string_type_unchanged(self):
        func = {"parameters": {"type": "string"}}
        normalize_schema(func)
        assert func["parameters"]["type"] == "string"

    def test_object_type_unchanged(self):
        func = {"parameters": {"type": "object"}}
        normalize_schema(func)
        assert func["parameters"]["type"] == "object"


# ── normalize_schema: array items default ────────────────────────────────────

class TestNormalizeSchemaArrayItems:
    def test_array_without_items_gets_default(self):
        func = {"parameters": {"type": "array"}}
        normalize_schema(func)
        assert func["parameters"]["items"] == {"type": "string"}

    def test_array_with_existing_items_unchanged(self):
        func = {"parameters": {"type": "array", "items": {"type": "integer"}}}
        normalize_schema(func)
        assert func["parameters"]["items"] == {"type": "integer"}

    def test_nested_array_without_items_gets_default(self):
        func = {
            "parameters": {
                "properties": {
                    "tags": {"type": "array"}
                }
            }
        }
        normalize_schema(func)
        assert func["parameters"]["properties"]["tags"]["items"] == {"type": "string"}


# ── normalize_schema: removes response and constraints fields ─────────────────

class TestNormalizeSchemaRemovesFields:
    def test_removes_response_field(self):
        func = {"name": "foo", "response": {"type": "string"}, "parameters": {}}
        normalize_schema(func)
        assert "response" not in func

    def test_removes_constraints_field(self):
        func = {"name": "foo", "constraints": "must be positive", "parameters": {}}
        normalize_schema(func)
        assert "constraints" not in func

    def test_removes_both_response_and_constraints(self):
        func = {
            "name": "foo",
            "response": {"type": "object"},
            "constraints": "some constraint",
            "parameters": {}
        }
        normalize_schema(func)
        assert "response" not in func
        assert "constraints" not in func

    def test_name_field_preserved(self):
        func = {"name": "get_balance", "response": {}, "parameters": {}}
        normalize_schema(func)
        assert func["name"] == "get_balance"

    def test_no_response_no_error(self):
        func = {"name": "foo", "parameters": {}}
        normalize_schema(func)  # should not raise
        assert "response" not in func


# ── clean_for_json: standard Python types pass through ───────────────────────

class TestCleanForJsonStandardTypes:
    def test_string_passthrough(self):
        assert run(clean_for_json("hello")) == "hello"

    def test_int_passthrough(self):
        assert run(clean_for_json(42)) == 42

    def test_float_passthrough(self):
        assert run(clean_for_json(3.14)) == 3.14

    def test_bool_passthrough(self):
        assert run(clean_for_json(True)) is True

    def test_none_passthrough(self):
        assert run(clean_for_json(None)) is None

    def test_list_passthrough(self):
        result = run(clean_for_json([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_dict_passthrough(self):
        result = run(clean_for_json({"a": 1, "b": "x"}))
        assert result == {"a": 1, "b": "x"}

    def test_tuple_preserved_as_tuple(self):
        result = run(clean_for_json((1, 2)))
        assert result == (1, 2)
        assert isinstance(result, tuple)

    def test_nested_dict_passthrough(self):
        obj = {"key": {"inner": [1, 2, 3]}}
        result = run(clean_for_json(obj))
        assert result == {"key": {"inner": [1, 2, 3]}}


# ── clean_for_json: complex numbers ──────────────────────────────────────────

class TestCleanForJsonComplexNumbers:
    def test_complex_returns_real_part(self):
        result = run(clean_for_json(3 + 4j))
        assert result == 3.0

    def test_complex_zero_imaginary(self):
        result = run(clean_for_json(5 + 0j))
        assert result == 5.0

    def test_complex_negative_real(self):
        result = run(clean_for_json(-2 + 1j))
        assert result == -2.0


# ── clean_for_json: numpy types (mock if not available) ──────────────────────

class TestCleanForJsonNumpyTypes:
    def test_numpy_bool_converted_to_bool(self):
        try:
            import numpy as np
            result = run(clean_for_json(np.bool_(True)))
            assert result is True
            assert type(result) is bool
        except ImportError:
            pytest.skip("numpy not available")

    def test_numpy_integer_converted_to_int(self):
        try:
            import numpy as np
            result = run(clean_for_json(np.int64(42)))
            assert result == 42
            assert type(result) is int
        except ImportError:
            pytest.skip("numpy not available")

    def test_numpy_floating_converted_to_float(self):
        try:
            import numpy as np
            result = run(clean_for_json(np.float64(3.14)))
            assert abs(result - 3.14) < 1e-10
            assert type(result) is float
        except ImportError:
            pytest.skip("numpy not available")

    def test_numpy_array_converted_to_list(self):
        try:
            import numpy as np
            result = run(clean_for_json(np.array([1, 2, 3])))
            assert result == [1, 2, 3]
            assert isinstance(result, list)
        except ImportError:
            pytest.skip("numpy not available")

    def test_without_numpy_plain_types_still_work(self):
        """Verify clean_for_json works fine when numpy is absent (via mocking)."""
        import unittest.mock as mock
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("numpy not available")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            # Plain types should still pass through
            result = run(clean_for_json({"x": 1, "y": "hello"}))
            assert result == {"x": 1, "y": "hello"}
