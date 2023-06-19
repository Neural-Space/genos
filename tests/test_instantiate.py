import copy
from typing import Any, Dict, List, Optional

import pytest
from genos import instantiate
from genos.instantiate import ObjectConfig, RecursiveClassInstantiationError
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError


class Level3:
    def __init__(self, a_3: str, b_3: int = 10):
        self.a_3 = a_3
        self.b_3 = b_3

    def __eq__(self, other: Any) -> Any:
        """Overrides the default implementation"""
        if isinstance(other, Level3):

            return self.a_3 == other.a_3 and self.b_3 == other.b_3
        return NotImplemented


class Level2:
    def __init__(self, a_2: Level3, b_2: int):
        self.a_2 = a_2
        self.b_2 = b_2

    def __eq__(self, other: Any) -> Any:
        """Overrides the default implementation"""
        if isinstance(other, Level2):

            return self.a_2 == other.a_2 and self.b_2 == other.b_2
        return NotImplemented


class Level1:
    def __init__(self, a_1: Level2):
        self.a_1 = a_1

    def __eq__(self, other: Any) -> Any:
        """Overrides the default implementation"""
        if isinstance(other, Level1):

            return self.a_1 == other.a_1
        return NotImplemented


def get_b():
    return 2


def dummy_test_method(a, b):
    return a + b


# The following code block was taken from Hydra
# https://github.com/facebookresearch/hydra/blob/master/tests/tests.test_utils.py
# Start Hydra block
def some_method() -> int:
    return 42


non_callable_object: List = []


class Bar:
    def __init__(self, a: Any, b: Any, c: Any, d: Any = "default_value") -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __repr__(self) -> str:
        return f"a={self.a}, b={self.b}, c={self.c}, d={self.d}"

    @staticmethod
    def static_method() -> int:
        return 43

    def __eq__(self, other: Any) -> Any:
        """Overrides the default implementation"""
        if isinstance(other, Bar):

            return (
                self.a == other.a
                and self.b == other.b
                and self.c == other.c
                and self.d == other.d
            )
        return NotImplemented

    def __ne__(self, other: Any) -> Any:
        """Overrides the default implementation (unnecessary in Python 3)"""
        x = self.__eq__(other)
        if x is not NotImplemented:
            return not x
        return NotImplemented


class Foo:
    def __init__(self, x: int) -> None:
        self.x = x

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, Foo):
            return self.x == other.x
        return False


class Baz(Foo):
    @classmethod
    def class_method(self, y: int) -> Any:
        return self(y + 1)

    @staticmethod
    def static_method(z: int) -> int:
        return z


class Fii:
    def __init__(self, a: Baz = Baz(10)):
        self.a = a

    def __repr__(self) -> str:
        return f"a={self.a}"

    def __eq__(self, other: Any) -> Any:
        """Overrides the default implementation"""
        if isinstance(other, Fii):

            return self.a == other.a
        return NotImplemented

    def __ne__(self, other: Any) -> Any:
        """Overrides the default implementation (unnecessary in Python 3)"""
        x = self.__eq__(other)
        if x is not NotImplemented:
            return not x
        return NotImplemented


fii = Fii()


def fum(k: int) -> int:
    return k + 1


@pytest.mark.parametrize(  # type: ignore
    "path, expected_type", [("tests.test_instantiate.Bar", Bar)]
)
def test_get_class(path: str, expected_type: type) -> None:
    assert instantiate.get_class(path) == expected_type


@pytest.mark.parametrize(  # type: ignore
    "path, return_value, expected_error",
    [
        ("tests.test_instantiate.some_method", 42, None),
        ("tests.test_instantiate.invalid_method", None, ValueError),
        ("cannot.locate.this.package", None, ValueError),
        ("tests.test_instantiate.non_callable_object", None, ValueError),
    ],
)
def test_get_method(path: str, return_value: Any, expected_error: Exception) -> None:
    if expected_error is not None:
        with pytest.raises(expected_error):
            assert instantiate.get_method(path)() == return_value
    else:
        assert instantiate.get_method(path)() == return_value


@pytest.mark.parametrize(  # type: ignore
    "path, return_value", [("tests.test_instantiate.Bar.static_method", 43)]
)
def test_get_static_method(path: str, return_value: Any) -> None:
    assert instantiate.get_static_method(path)() == return_value


@pytest.mark.parametrize(  # type: ignore
    "input_conf, key_to_get_config, kwargs_to_pass, expected",
    [
        (
            {
                "cls": "tests.test_instantiate.Bar",
                "params": {"a": 10, "b": 20, "c": 30, "d": 40},
            },
            None,
            {},
            Bar(10, 20, 30, 40),
        ),
        (
            {
                "all_params": {
                    "main": {
                        "cls": "tests.test_instantiate.Bar",
                        "params": {"a": 10, "b": 20, "c": "${all_params.aux.c}"},
                    },
                    "aux": {"c": 30},
                }
            },
            "all_params.main",
            {"d": 40},
            Bar(10, 20, 30, 40),
        ),
        (
            {"cls": "tests.test_instantiate.Bar", "params": {"b": 20, "c": 30}},
            None,
            {"a": 10, "d": 40},
            Bar(10, 20, 30, 40),
        ),
        (
            {
                "cls": "tests.test_instantiate.Bar",
                "params": {"b": 200, "c": "${params.b}"},
            },
            None,
            {"a": 10, "d": 40},
            Bar(10, 200, 200, 40),
        ),
        # Check class and static methods
        (
            {"cls": "tests.test_instantiate.Baz.class_method", "params": {"y": 10}},
            None,
            {},
            Baz(11),
        ),
        (
            {"cls": "tests.test_instantiate.Baz.static_method", "params": {"z": 43}},
            None,
            {},
            43,
        ),
        # Check nested types and static methods
        ({"cls": "tests.test_instantiate.Fii", "params": {}}, None, {}, Fii(Baz(10))),
        (
            {"cls": "tests.test_instantiate.fii.a.class_method", "params": {"y": 10}},
            None,
            {},
            Baz(11),
        ),
        (
            {"cls": "tests.test_instantiate.fii.a.static_method", "params": {"z": 43}},
            None,
            {},
            43,
        ),
        # Check that default value is respected
        (
            {
                "cls": "tests.test_instantiate.Bar",
                "params": {"b": 200, "c": "${params.b}"},
            },
            None,
            {"a": 10},
            Bar(10, 200, 200, "default_value"),
        ),
        (
            {"cls": "tests.test_instantiate.Bar", "params": {}},
            None,
            {"a": 10, "b": 20, "c": 30},
            Bar(10, 20, 30, "default_value"),
        ),
        # call a function from a module
        ({"cls": "tests.test_instantiate.fum", "params": {"k": 43}}, None, {}, 44),
        # Check builtins
        ({"cls": "builtins.str", "params": {"object": 43}}, None, {}, "43"),
    ],
)
def test_class_instantiate(
    input_conf: Dict[str, Any],
    key_to_get_config: Optional[str],
    kwargs_to_pass: Dict[str, Any],
    expected: Any,
) -> Any:
    conf = OmegaConf.create(input_conf)
    assert isinstance(conf, DictConfig)
    if key_to_get_config is None:
        config_to_pass = conf
    else:
        config_to_pass = OmegaConf.select(conf, key_to_get_config)
    config_to_pass_copy = copy.deepcopy(config_to_pass)
    obj = instantiate.instantiate(config_to_pass, **kwargs_to_pass)
    assert obj == expected
    # make sure config is not modified by instantiate
    assert config_to_pass == config_to_pass_copy


def test_class_instantiate_pass_omegaconf_node() -> Any:
    pc = ObjectConfig()
    # This is a bit clunky because it exposes a problem with the backport of dataclass on Python 3.6
    # see: https://github.com/ericvsmith/dataclasses/issues/155
    pc.cls = "tests.test_instantiate.Bar"
    pc.params = {"b": 200, "c": {"x": 10, "y": "${params.b}"}}
    conf = OmegaConf.structured(pc)
    obj = instantiate.instantiate(conf, **{"a": 10, "d": Foo(99)})
    assert obj == Bar(10, 200, {"x": 10, "y": 200}, Foo(99))
    assert OmegaConf.is_config(obj.c)


def test_class_warning() -> None:
    expected = Bar(10, 20, 30, 40)
    config = OmegaConf.structured(
        {
            "cls": "tests.test_instantiate.Bar",
            "params": {"a": 10, "b": 20, "c": 30, "d": 40},
        }
    )
    assert instantiate.instantiate(config) == expected

    config = OmegaConf.structured(
        {
            "cls": "tests.test_instantiate.Bar",
            "params": {"a": 10, "b": 20, "c": 30, "d": 40},
        }
    )
    assert instantiate.instantiate(config) == expected


# End Hydra block


@pytest.mark.parametrize(
    "input_conf, key_to_get_config, kwargs_to_pass, expected",
    [
        (
            {
                "cls": "tests.test_instantiate.Level1",
                "params": {
                    "a_1": {
                        "cls": "tests.test_instantiate.Level2",
                        "params": {
                            "a_2": {
                                "cls": "tests.test_instantiate.Level3",
                                "params": {"a_3": "a_3", "b_3": 11},
                            },
                            "b_2": 42,
                        },
                    }
                },
            },
            None,
            {},
            Level1(a_1=Level2(a_2=Level3(a_3="a_3", b_3=11), b_2=42)),
        ),
        (
            {
                "cls": "tests.test_instantiate.dummy_test_method",
                "params": {
                    "a": 1,
                    "b": {
                        "cls": "tests.test_instantiate.get_b",
                    },
                },
            },
            None,
            {},
            3,
        ),
    ],
)
def test_recursive_instantiate(
    input_conf: Dict[str, Any],
    key_to_get_config: Optional[str],
    kwargs_to_pass: Dict[str, Any],
    expected: Any,
) -> Any:
    conf = OmegaConf.create(input_conf)
    assert isinstance(conf, DictConfig)
    if key_to_get_config is None:
        config_to_pass = conf
    else:
        config_to_pass = OmegaConf.select(conf, key_to_get_config)
    config_to_pass_copy = copy.deepcopy(config_to_pass)
    obj = instantiate.recursive_instantiate(config_to_pass, **kwargs_to_pass)
    assert obj == expected
    # make sure config is not modified by instantiate
    assert config_to_pass == config_to_pass_copy


@pytest.mark.parametrize(
    "input_conf, args, expected_instantiate_exception, expected_recursive_instantiate_exception",
    [
        ({"cls": "some.path.that.will.fail", "params": {}}, [], Exception, Exception),
        (
            {"cls": "tests.test_instantiate.SomeLevel", "params": {}},
            [],
            ValueError,
            RecursiveClassInstantiationError,
        ),
        (
            {"cls": "tests.test_instantiate.SomeLevel", "params": {}},
            [],
            ValueError,
            RecursiveClassInstantiationError,
        ),
        (
            {"classsss": "tests.test_instantiate.SomeLevel", "params": {}},
            [],
            ValueError,
            ConfigKeyError,
        ),
        (
            None,
            [],
            AssertionError,
            AssertionError,
        ),
    ],
)
def test_invalid_instantiate(
    input_conf: Dict,
    args: List,
    expected_instantiate_exception: Exception,
    expected_recursive_instantiate_exception: Exception,
):

    if input_conf:
        conf = OmegaConf.create(input_conf)
        assert isinstance(conf, DictConfig)
    else:
        conf = input_conf
    with pytest.raises(expected_instantiate_exception):
        instantiate.instantiate(conf)

    with pytest.raises(expected_recursive_instantiate_exception):
        instantiate.recursive_instantiate(conf)
