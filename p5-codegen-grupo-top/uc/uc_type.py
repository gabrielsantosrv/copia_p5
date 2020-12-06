class uCType:
    """
    Class that represents a type in the uC language.  Basic
    Types are declared as singleton instances of this type.
    """

    def __init__(
        self, name, binary_ops=set(), unary_ops=set(), rel_ops=set(), assign_ops=set()
    ):
        """
        You must implement yourself and figure out what to store.
        """
        self.typename = name
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.rel_ops = rel_ops
        self.assign_ops = assign_ops


# Create specific instances of basic types. You will need to add
# appropriate arguments depending on your definition of uCType
IntType = uCType(
    "int",
    unary_ops={"-", "+", "--", "++", "p--", "p++", "*", "&"},
    binary_ops={"+", "-", "*", "/", "%"},
    rel_ops={"==", "!=", "<", ">", "<=", ">="},
    assign_ops={"=", "+=", "-=", "*=", "/=", "%="},
)

FloatType = uCType(
    "float",
    unary_ops={"-", "+", "*", "&"},
    binary_ops={"+", "-", "*", "/", "%"},
    rel_ops={"==", "!=", "<", ">", "<=", ">="},
    assign_ops={"=", "+=", "-=", "*=", "/=", "%="},
)

CharType = uCType(
    "char",
    rel_ops={"==", "!=", "&&", "||"},
    assign_ops={"="},
)

BoolType = uCType(
    "bool",
    unary_ops={"!"},
    rel_ops={"==", "!=", "&&", "||"},
    assign_ops={"="},
)

StringType = uCType(
    "string",
    rel_ops={"==", "!="},
    assign_ops={"="}
)

VoidType = uCType("void")

class ArrayType(uCType):
    def __init__(self, element_type, size=None):
        self.type = element_type
        self.size = size
        super().__init__(name="array", unary_ops={"*", "&"}, rel_ops={"==", "!="})

class FunctionType(uCType):
    def __init__(self, return_type=IntType, parameter_types=[]):
        self.return_type = return_type
        self.parameter_types = parameter_types

        if return_type == IntType:
            super().__init__(None,
                unary_ops={"-", "+", "--", "++", "p--", "p++", "*", "&"},
                binary_ops={"+", "-", "*", "/", "%"},
                rel_ops={"==", "!=", "<", ">", "<=", ">="},
                assign_ops={"=", "+=", "-=", "*=", "/=", "%="}
            )
        elif return_type == FloatType:
            super().__init__(None,
                unary_ops={"-", "+", "*", "&"},
                binary_ops={"+", "-", "*", "/", "%"},
                rel_ops={"==", "!=", "<", ">", "<=", ">="},
                assign_ops={"=", "+=", "-=", "*=", "/=", "%="}
            )
        elif return_type == CharType:
            super().__init__(None,
                rel_ops={"==", "!=", "&&", "||"},
                assign_ops={"="}
            )
        elif return_type == BoolType:
            super().__init__(None,
                unary_ops={"!"},
                rel_ops={"==", "!=", "&&", "||"},
                assign_ops={"="}
            )