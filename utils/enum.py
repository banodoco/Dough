from enum import Enum

class ExtendedEnum(Enum):

    @classmethod
    def value_list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_