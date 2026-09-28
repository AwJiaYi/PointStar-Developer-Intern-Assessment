from src.tools import calculator


def test_basic_arithmetic():
    assert calculator("14 - 5") == "9"
    assert calculator("5 + 3 * 2") == "11"
    assert calculator("(20 + 10) / 3") == "10"


def test_division_by_zero_returns_safe_error():
    result = calculator("10 / 0")
    assert result.startswith("Calculator error:")
    assert "Division by zero" in result


def test_blocks_python_code_execution():
    result = calculator("__import__('os').system('echo unsafe')")
    assert result.startswith("Calculator error:")


def test_blocks_unsupported_power_operator():
    result = calculator("2 ** 100")
    assert result.startswith("Calculator error:")
    assert "Unsupported mathematical operator" in result
