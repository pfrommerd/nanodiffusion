from nanoconfig import config, options, MISSING

@config
class Simple:
    a: int = 1
    b: str = MISSING
    c: float = 1.0

def test_simple():
    config_opts = options.as_options(Simple)
    parsed = options.parse_cli_options(config_opts, ["--a=2", "--b=b", "--c=3.0"])
    assert parsed["a"] == "2"
    assert parsed["b"] == "b"
    assert parsed["c"] == "3.0"
    simple = options.from_parsed_options(parsed, Simple)
    assert simple.a == 2
    assert simple.b == "b"
    assert simple.c == 3.0

@config(variant="base")
class VariantBase:
    a: int = 1

@config(variant="v1")
class Variant1(VariantBase):
    a: int = 2
    b: str = "b1"
    c: str = "foo"

def test_variant():
    config_opts = options.as_options(VariantBase)
    parsed = options.parse_cli_options(config_opts, ["--v1.c=bar", "--type=v1"])
    assert parsed["v1.c"] == "bar"
    variant1 = options.from_parsed_options(parsed, VariantBase)
    assert variant1.a == 2
    assert variant1.c == "bar"