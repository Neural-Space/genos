"""
Authors:
 - Ayushman Dash <ayushman@neuralspace.ai>
 - Kushal Jain <kushal@neuralspace.ai>
"""

import builtins
import copy
import logging
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Dict, List, Type, Union

from omegaconf import MISSING, DictConfig, OmegaConf, _utils, read_write

logger = logging.getLogger(__name__)


# The following code blocks has been taken from Hydra
# https://github.com/facebookresearch/hydra/blob/master/hydra/_internal/utils.py
# Start Hydra block

TaskFunction = Callable[[Any], Any]


@dataclass
# This extends Dict[str, Any] to allow for the deprecated "class" field.
# Once support for class field removed this can stop extending Dict.
class ObjectConfig(Dict[str, Any]):
    """A dataclass which holds config entries to instantiate any class or function
    Schema details here: `panini.schemas.instantiator.ObjectConfigSchema`
    """

    # class, class method or function name
    cls: str = MISSING

    # positional args if any
    args: List = field(default_factory=list)

    # parameters to pass to cls when calling it
    params: Any = field(default_factory=dict)


def __instantiate_class(
    clazz: Type[Any],
    class_config: Union[ObjectConfig, DictConfig],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """This function instantiates a class using a class reference
    Args:
        - `clazz`: (Type[Any])
            Reference to a class
        - `config`: (panini.types.generic.ObjectConfig)
            An ObjectConfig or DictConfig describing what to call and what params to use
        - `args`: (List)
            A list of positional arguments which are passed to the constructor.
        - `kwargs`: (Dict)
            A dictionary of keyword arguments that are passed to the class constructor
    Return:
        - `object` (:obj:`object`):
            an instance of class `clazz`
    """
    final_kwargs = __get_kwargs(class_config, **kwargs)
    return clazz(*args, **final_kwargs)


def __instantiate_class_recursive(
    clazz: Type[Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """This function instantiates a class using a class reference
    Args:
        - `clazz`: (Type[Any])
            Reference to a class
        - `config`: (panini.types.generic.ObjectConfig)
            An ObjectConfig or DictConfig describing what to call and what params to use
        - `args`: (List)
            A list of positional arguments which are passed to the constructor.
        - `kwargs`: (Dict)
            A dictionary of keyword arguments that are passed to the class constructor
    Return:
        - `object` (:obj:`object`):
            an instance of class `clazz`
    """
    return clazz(*args, **kwargs)


def __call_callable(
    fn: Callable[..., Any],
    config: Union[ObjectConfig, DictConfig],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Calls function `fn` byy passing a given set of keyword arguments and positional arguments
    Args:
        - `fn`: (Callable)
            Reference to a function
        - `config`: (panini.types.generic.ObjectConfig)
            An ObjectConfig or DictConfig describing what to call and what params to use
        - `args`: (List)
            A list of positional arguments which are passed to the constructor.
        - `kwargs`: (Dict)
            A dictionary of keyword arguments that are passed to the class constructor
    Return:
        - the return value of `fn`
    """
    final_kwargs = __get_kwargs(config, **kwargs)
    return fn(*args, **final_kwargs)


def __get_type(
    parts: List[str], n: int, obj: type, path: str
) -> Union[type, Callable[..., Any]]:
    """Returns a reference to a class or a callable type given a string reference as a list.
    E.g., ["path", "to", "some", "package"]
       Args:
           - `parts`: (List)
               List of reference paths. E.g., ["path", "to", "some", "package"]
           - `n`: (int)
               Number of parts in the `parts`
           - `obj`: (type)
               The `Class` we are looking for in `parts`
           - `path`: (str)
               The full path to the class or callable type
       Return:
           - `obj_callable` or `obj_callable`:
               instance of a class or a callable type
    """
    for part in parts[n:]:
        if not hasattr(obj, part):
            raise ValueError(
                f"Error finding attribute ({part}) in class ({obj.__name__}): {path}"
            )
        obj = getattr(obj, part)
    if isinstance(obj, type):
        obj_type: type = obj
        return obj_type
    elif callable(obj):
        obj_callable: Callable[..., Any] = obj
        return obj_callable
    else:
        # dummy case
        raise ValueError(f"Invalid type ({type(obj)}) found for {path}")


def __locate(path: str) -> Union[type, Callable[..., Any]]:
    """Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.
    Args:
        - `path`: (List)
            string reference to some class or callable type. E.g., path.to.some.Class
    Return:
        - `obj_callable` or `obj_callable`:
            instance of a class or a callable type
    """

    parts = [part for part in path.split(".") if part]
    module = None
    for n in reversed(range(len(parts))):
        try:
            module = import_module(".".join(parts[:n]))
        except Exception as e:
            if n == 0:
                logger.error(f"Error loading module {path} : {e}")
                raise e
            continue
        if module:
            break
    if module:
        obj = module
    else:
        obj = builtins

    return __get_type(parts, n, obj, path)


def __get_kwargs(class_config: Union[ObjectConfig, DictConfig], **kwargs: Any) -> Any:
    # copy config to avoid mutating it when merging with kwargs
    config_copy = copy.deepcopy(class_config)

    # Manually set parent as deepcopy does not currently handles it (https://github.com/omry/omegaconf/issues/130)
    # noinspection PyProtectedMember
    config_copy._set_parent(class_config._get_parent())  # type: ignore
    class_config = config_copy

    params = class_config.params if "params" in class_config else OmegaConf.create()
    assert isinstance(
        params, DictConfig
    ), f"Input config params are expected to be a mapping, found {type(class_config.params).__name__}"
    primitives = {}
    rest = {}
    for k, v in kwargs.items():
        if _utils.is_primitive_type_annotation(v) or isinstance(v, (dict, list)):
            primitives[k] = v
        else:
            rest[k] = v
    final_kwargs = {}
    with read_write(params):
        params.merge_with(OmegaConf.create(primitives))

    for k, v in params.items():
        final_kwargs[k] = v

    for k, v in rest.items():
        final_kwargs[k] = v
    return final_kwargs


def __get_cls_name(config: Union[ObjectConfig, DictConfig]) -> str:
    """A helper function to get the `cls` variable in the config
    Args:
        - `config`: (panini.types.generic.ObjectConfig)
             An ObjectConfig or DictConfig describing what to call and what params to use
    """
    if "cls" in config:
        return config.cls
    else:
        raise ValueError("Input config does not have a cls field")


def call(config: Union[ObjectConfig, DictConfig], *args: Any, **kwargs: Any) -> Any:
    """This function instantiates a class or a callable type
    Args:
       - `config`: (panini.types.generic.ObjectConfig)
            An ObjectConfig or DictConfig describing what to call and what params to use
       - `args`: (panini.types.generic.ObjectConfig)
            optional positional parameters pass-through
       - `kwargs`: (panini.types.generic.ObjectConfig)
            optional named parameters pass-through
    Return:
       - the return value from the specified class or method
    """
    assert config is not None, "Input config is None"
    try:
        cls = __get_cls_name(config)
        type_or_callable = __locate(cls)
        if isinstance(type_or_callable, type):
            return __instantiate_class(type_or_callable, config, *args, **kwargs)
        else:
            assert callable(type_or_callable)
            return __call_callable(type_or_callable, config, *args, **kwargs)
    except Exception as e:
        logger.error(f"Error instantiating '{config}' : {e}")
        raise e


# Alias for call
instantiate = call


def get_class(path: str) -> type:
    """This function returns a reference to a class given a path. E.g., path.to.some.Class
    Args:
       - `path`: (str)
            The full path to the class or callable type
    Return:
       - a reference to the class
    """
    try:
        cls = __locate(path)
        if not isinstance(cls, type):
            raise ValueError(f"Located non-class in {path} : {type(cls).__name__}")
        return cls
    except Exception as e:
        logger.error(f"Error initializing class at {path} : {e}")
        raise e


def get_method(path: str) -> Callable[..., Any]:
    """This function returns a reference to a function given a path. E.g., path.to.some.function
    Args:
       - `path`: (str)
            The full path to the function
    Return:
       - a reference to the function
    """
    if not path:
        raise ValueError("path cannot be empty. make sure to pass a correct reference.")
    try:
        cl = __locate(path)
        if not callable(cl):
            raise ValueError(f"Non callable object located : {type(cl).__name__}")
        return cl
    except Exception as e:
        logger.error(f"Error getting callable at {path} : {e}")
        raise e


# Alias for get_method
get_static_method = get_method


# End Hydra block


class RecursiveClassInstantiationError(Exception):
    pass


def instantiate_argument(object_config_argument: Any):
    """This function instantiates an argument if it is a class config.
    Otherwise it just returns the argument value.
    This is a recursive function and instantiates arguments recursively.
    Args:
       - `object_config_argument`: (Any)
            Arguments of a class as in ObjectConfig
    Return:
       - object_config_argument:
            all initialized arguments
    """
    if (
        isinstance(object_config_argument, dict)
        or isinstance(object_config_argument, DictConfig)
    ) and "cls" in object_config_argument.keys():
        return recursive_instantiate(OmegaConf.create(object_config_argument))

    if isinstance(object_config_argument, list):
        for i, value in enumerate(object_config_argument):
            object_config_argument[i] = instantiate_argument(value)
    return object_config_argument


def recursive_instantiate(
    config: Union[ObjectConfig, DictConfig, dict], *args: Any, **kwargs: Any
) -> Any:
    """This function instantiates a class recursively using an ObjectConfig.
    Args:
       - `config`: (panini.types.generic.ObjectConfig)
            An ObjectConfig or DictConfig describing what to call and what params to use
       - `args`: (panini.types.generic.ObjectConfig)
            optional positional parameters pass-through
       - `kwargs`: (panini.types.generic.ObjectConfig)
            optional named parameters pass-through
    Return:
       - instance of the class
    """
    if isinstance(config, Dict):
        config = OmegaConf.create(config)

    assert config is not None, "Input config is None"
    # copy config to avoid mutating it when merging with kwargs
    config_copy = copy.deepcopy(config)
    config_copy._set_parent(config._get_parent())
    config = config_copy

    try:
        # clazz = get_class(__get_cls_name(config))
        cls = __get_cls_name(config)
        clazz = __locate(cls)
        params = config.params if "params" in config else OmegaConf.create()
        if "args" in config:
            args = config.args
        assert isinstance(
            params, DictConfig
        ), "Input config params are expected to be a mapping, found {}".format(
            type(config.params)
        )
        params.merge_with(OmegaConf.create(kwargs))

        # Needed to avoid omegaconf.errors.UnsupportedValueType triggered by
        # params[name] = instantiate_argument(arg)
        params = OmegaConf.to_container(params)

        args = list(args)
        for i, arg in enumerate(args):
            args[i] = instantiate_argument(arg)

        for name, arg in params.items():
            params[name] = instantiate_argument(arg)

        if isinstance(clazz, type):
            return __instantiate_class_recursive(clazz, *args, **params)
        else:
            assert callable(clazz)
            return __call_callable(clazz, config, *args, **params)

    except Exception as e:
        raise RecursiveClassInstantiationError(
            f"Error instantiating {config['cls']} : {e}"
        )
