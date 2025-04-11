from . import Config, Missing, MISSING

from dataclasses import dataclass, fields, MISSING as DC_MISSING
import typing as ty
import types
import abc
import sys
import logging
logger = logging.getLogger(__name__)

_type = type
T = ty.TypeVar("T")

class OptionParseError(ValueError):
    pass

@dataclass
class Option:
    name: str
    # The following have no practical impact
    # on the parsing part, but are used for help documentation
    type: type # May be int, float, str, bool,
               # dict[str, str], list[T], tuple[T]
               # ty.Any yields a generic type of one of the above
    # A default value. Used for help documentation,
    # but not actually returned as a default value.
    # If MISSING, then the option is required.
    default: ty.Any | Missing

def as_options(type : ty.Type[T], default : T | Missing = MISSING, *, 
                # For variants, the type of the base class for the variant.
                # We will not include options contained in the base type.
                relative_to_base : ty.Type | None = None,
                prefix : str = "",
            ) -> ty.Iterator[Option] | T:
    if isinstance(type, ty.Type) and issubclass(type, Config):
        # If we have variants, add a "type" option
        variants = type.__variants__
        variant_lookup = {type: alias for alias, type in variants.items()}
        if variants:
            # If the type is an abstract class,
            # we require the user to specify
            type_default = MISSING if isinstance(type, abc.ABCMeta) else type
            if default is not MISSING:
                type_default = _type(default)
            if type_default not in variant_lookup:
                raise ValueError(f"No variant registration for type {type_default} at \"{prefix}\".")
            type_default = variant_lookup[type_default] if type_default is not MISSING else MISSING

            yield Option(_join(prefix, "type"), str, type_default)

        # Do not output fields for the base class
        # if relative_to_base is not None
        base_fields = (
            set() if relative_to_base is None else 
            set(f.name for f in fields(relative_to_base))
        )
        for f in fields(type):
            if f.name in base_fields: # We should not include these fields
                continue
            field_default = (
                getattr(default, f.name) if default is not MISSING else
                (f.default if f.default is not DC_MISSING else MISSING)
            )
            flat = f.metadata.get("flat", False)
            yield from as_options(f.type, default=field_default,
                               prefix=prefix if flat else _join(prefix, f.name))
        # If we have variants, output each of the variant options
        if variants:
            for alias, variant_type in variants.items():
                # If the type is the same as the base type,
                # we do not need to output the base type
                if variant_type == type:
                    continue
                # Only specify the default if the type is the same.
                subvariant_default = (
                    default if variant_type is _type(default) else MISSING
                )
                yield from as_options(variant_type,
                    default=subvariant_default,
                    prefix=_join(prefix, alias),
                    relative_to_base=type
                )
    elif isinstance(type, ty.Type) and issubclass(type, (str, int, float, bool)):
        yield Option(prefix, type, default)
    elif _is_optional(type)[0]:
        type = _is_optional(type)[1]
        yield from as_options(type, default=None if default is MISSING else default, prefix=prefix)
    elif isinstance(type, (ty.GenericAlias, ty._GenericAlias)):
        logger.warning(f"unknown option generic type {type}")
    else:
        raise ValueError(f"Unable to convert type {type} to option.")

def _is_optional(t: object) -> tuple[bool, ty.Type]:
    origin = ty.get_origin(t)
    if not (origin is ty.Union or origin is types.UnionType):
        return False, None
    a, b = ty.get_args(t)
    if a is type(None) or b is type(None):
        return True, a if a is not None else b


def _join(prefix, name):
    if prefix:
        return f"{prefix}.{name}"
    return name

# Will parse the options, removing any parsed
# options from the dictionary
def from_parsed_options(options: dict[str, str], 
                  type : ty.Type[T], default: T | Missing = MISSING, *,
                  prefix="") -> T:
    if isinstance(type, ty.Type) and issubclass(type, Config):
        if type.__variants__:
            default_type = _type(default) if default is not MISSING else (
                type if not isinstance(type, abc.ABCMeta) else MISSING
            )
            variant = options.pop(_join(prefix, "type"), MISSING)
            if variant is not MISSING and variant not in type.__variants__:
                raise OptionParseError(f"Invalid variant {variant} for {type}")
            config_type = type.__variants__[variant] if variant is not MISSING else default_type
            config_fields = default_type if config_type is MISSING else config_type
            if config_type is MISSING:
                raise OptionParseError(f"Missing variant specifier {prefix}type for {type}")
            # If we specified a variant different than the default,
            # remove the default value.
            if config_type != default_type:
                default = MISSING
        else:
            config_type = type
        config_fields = {}
        # First go through the base fields
        for f in fields(type):
            flat = f.metadata.get("flat", False)
            config_fields[f.name] = (
                prefix if flat else _join(prefix, f.name), f
            )
        # Go through the variant-specific fields
        if config_type is not type:
            for f in fields(config_type):
                # If we are overriding a base field,
                # override the field so we get the updated default
                flat = f.metadata.get("flat", False)
                if f.name in config_fields:
                    config_fields[f.name] = (
                        prefix if flat else _join(prefix, f.name), f
                    )
                else:
                    config_fields[f.name] = (
                        _join(prefix, variant) if flat else
                        _join(prefix, f"{variant}.{f.name}"), f
                    )
        # Now we can parse the options,
        config_args = {}
        for field_prefix, f in config_fields.values():
            default_value = (
                getattr(default, f.name) if default is not MISSING else
                (f.default if f.default is not DC_MISSING else MISSING)
            )
            value = from_parsed_options(options, f.type, default_value, 
                                        prefix=field_prefix)
            if value is not MISSING:
                config_args[f.name] = value
        return config_type(**config_args)
    elif isinstance(type, ty.Type) and issubclass(type, (str, int, float, bool)):
        value = options.pop(prefix, default)
        if value is MISSING:
            raise OptionParseError(f"Missing option {prefix} for {type}")
        if issubclass(type, bool) and type(value) is str:
            value = value.lower() == "true"
        return type(value)
    elif _is_optional(type)[0]:
        type = _is_optional(type)[1]
        return from_parsed_options(options, type, default=None if default is MISSING else default, prefix=prefix)
    elif isinstance(type, (ty.GenericAlias,ty._GenericAlias)):
        logger.warning(f"unknown parsing of generic type {type}")
        return MISSING
    else:
        raise OptionParseError(f"Unable to parse options for type {type}.")

# Will parse all known options from the list
def parse_cli_options(options: ty.Iterable[Option],
                  args: list[str] | None = None,
                  parse_all: bool = True, parse_help = True) -> dict[str, str]:
    options = list(options)
    options.append(Option("help", bool, False))
    valid_keys = set(o.name for o in options)
    if args is None:
        args = list(sys.argv[1:])
    parsed_options = {}
    parsed_args = []
    last_key = None
    for i, arg in enumerate(args):
        if arg.startswith("--"):
            last_key = None
            arg = arg[2:]
            if "=" in arg:
                key, value = arg.split("=", 1)
                if key in valid_keys:
                    parsed_args.append(i)
                    parsed_options[key] = value
            else:
                last_key = arg
                # For --flag, we set the value to "true"
                if last_key in valid_keys:
                    parsed_args.append(i)
                    parsed_options[last_key] = "true"
        elif last_key is not None:
            if last_key in valid_keys:
                parsed_args.append(i)
                parsed_options[last_key] = arg
            last_key = None
    # Remove from the list
    for i in reversed(parsed_args):
        del args[i]
    if parse_all and len(args) > 0:
        raise OptionParseError(f"Unknown options {args}. Valid options are {valid_keys}.")
    for opt in options:
        if opt.default is MISSING and opt.name not in parsed_options:
            raise OptionParseError(f"Missing option {opt.name} for {opt.type}")

    if parsed_options.get("help", False):
        print("Options:")
        for opt in options:
            type_name = opt.type.__name__
            if opt.type == bool:
                print(f"  --{opt.name}")
            else:
                if opt.default is not MISSING:
                    print(f"  --{opt.name}={opt.default} ({type_name})")
                else:
                    print(f"  --{opt.name} ({type_name})")
        sys.exit(0)

    return parsed_options
