import sys


def represent_node(obj, indent):
    def _repr(obj, indent, printed_set):
        """
        Get the representation of an object, with dedicated pprint-like format for lists.
        """
        if isinstance(obj, list):
            indent += 1
            sep = ",\n" + (" " * indent)
            final_sep = ",\n" + (" " * (indent - 1))
            return (
                "["
                + (sep.join((_repr(e, indent, printed_set) for e in obj)))
                + final_sep
                + "]"
            )
        elif isinstance(obj, Node):
            if obj in printed_set:
                return ""
            else:
                printed_set.add(obj)
            result = obj.__class__.__name__ + "("
            indent += len(obj.__class__.__name__) + 1
            attrs = []
            for name in obj.__slots__[:-1]:
                if name == "bind":
                    continue
                value = getattr(obj, name)
                value_str = _repr(value, indent + len(name) + 1, printed_set)
                attrs.append(name + "=" + value_str)
            sep = ",\n" + (" " * indent)
            final_sep = ",\n" + (" " * (indent - 1))
            result += sep.join(attrs)
            result += ")"
            return result
        elif isinstance(obj, str):
            return obj
        else:
            return ""

    # avoid infinite recursion with printed_set
    printed_set = set()
    return _repr(obj, indent, printed_set)


class Node:
    """Abstract base class for AST nodes."""

    __slots__ = "coord"
    attr_names = ()

    def __init__(self, coord=None):
        self.coord = coord

    def __repr__(self):
        """Generates a python representation of the current node"""
        return represent_node(self, 0)

    def children(self):
        """A sequence of all children that are Nodes"""
        pass

    def show(
        self,
        buf=sys.stdout,
        offset=0,
        attrnames=False,
        nodenames=False,
        showcoord=False,
        _my_node_name=None,
    ):
        """Pretty print the Node and all its attributes and children (recursively) to a buffer.
        buf:
            Open IO buffer into which the Node is printed.
        offset:
            Initial offset (amount of leading spaces)
        attrnames:
            True if you want to see the attribute names in name=value pairs. False to only see the values.
        nodenames:
            True if you want to see the actual node names within their parents.
        showcoord:
            Do you want the coordinates of each Node to be displayed.
        """
        lead = " " * offset
        if nodenames and _my_node_name is not None:
            buf.write(lead + self.__class__.__name__ + " <" + _my_node_name + ">: ")
            inner_offset = len(self.__class__.__name__ + " <" + _my_node_name + ">: ")
        else:
            buf.write(lead + self.__class__.__name__ + ":")
            inner_offset = len(self.__class__.__name__ + ":")

        if self.attr_names:
            if attrnames:
                nvlist = [
                    (n, represent_node(getattr(self, n), offset+inner_offset+1+len(n)+1))
                    for n in self.attr_names
                    if getattr(self, n) is not None
                ]
                attrstr = ", ".join("%s=%s" % nv for nv in nvlist)
            else:
                vlist = [getattr(self, n) for n in self.attr_names]
                attrstr = ", ".join(
                    represent_node(v, offset + inner_offset + 1) for v in vlist
                )
            buf.write(" " + attrstr)

        if showcoord:
            if self.coord and self.coord.line != 0:
                buf.write(" %s" % self.coord)
        buf.write("\n")

        for (child_name, child) in self.children():
            child.show(buf, offset + 4, attrnames, nodenames, showcoord, child_name)


class BinaryOp(Node):
    __slots__ = ("op", "lvalue", "rvalue", "gen_location", "uc_type", "coord")

    def __init__(self, op, lvalue, rvalue, coord=None):
        self.op = op
        self.lvalue = lvalue
        self.rvalue = rvalue
        self.coord = coord
        self.gen_location = None
        self.uc_type = None

    def children(self):
        nodelist = []
        if self.lvalue is not None:
            nodelist.append(("lvalue", self.lvalue))
        if self.rvalue is not None:
            nodelist.append(("rvalue", self.rvalue))
        return tuple(nodelist)

    attr_names = ("op",)

class Constant(Node):
    __slots__ = ("type", "value", "gen_location", "uc_type", "coord")

    def __init__(self, type, value, coord=None):
        self.type = type
        self.value = value
        self.coord = coord
        self.gen_location = None
        self.uc_type = None

    def children(self):
        return tuple()

    attr_names = (
        "type",
        "value",
    )

class Program(Node):
    __slots__ = ("gdecls", "text", "coord")

    def __init__(self, gdecls, coord=None):
        self.gdecls = gdecls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.gdecls or []):
            nodelist.append(("gdecls[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()

class ID(Node):
    __slots__ = ("name", "scope", "type", "uc_type", "gen_location", "kind", "coord")

    def __init__(self, name, coord=None):
        self.coord = coord
        self.name = name
        self.scope = None
        self.uc_type = None
        self.kind = None
        self.gen_location = None
        self.type = None

    def children(self):
        return ()

    attr_names = ("name",)

class ArrayDecl(Node):
    __slots__ = ("type", "vardecl", "dim", "uc_type", "coord")

    def __init__(self, type, vardecl, dim, coord=None):
        self.dim = dim
        self.type = type
        self.coord = coord
        self.vardecl = vardecl
        self.uc_type = None

    def children(self):
        children = []
        if self.type is not None:
            children.append(("type", self.type))
        if self.dim is not None:
            children.append(("dim", self.dim))
        return tuple(children)

    attr_names = ()
    
class ArrayRef(Node):
    __slots__ = ("lid", "rid", "uc_type", "gen_location", "coord")
    
    def __init__(self, lid, rid, coord=None):
        self.lid = lid
        self.rid = rid
        self.coord = coord
        self.uc_type = None
        self.gen_location = None

    def children(self):
        children = []
        if self.lid is not None:
            children.append(("lid", self.lid))
        if self.rid is not None:
            children.append(("rid", self.rid))
        return tuple(children)

    attr_names = ()

class Assert(Node):
    __slots__ = ("expression", "coord", "cfg")
    
    def __init__(self, expression, coord=None):
        self.expression = expression
        self.coord = coord
        self.cfg = None

    def children(self):
        children = []
        if self.expression is not None:
            children.append(("expression", self.expression))
        return tuple(children)

    attr_names = ()

class Assignment(Node):
    __slots__ = ("op", "lvalue", "rvalue", "uc_type", "coord")

    def __init__(self, op, lvalue, rvalue, coord=None):
        self.op = op
        self.lvalue = lvalue
        self.rvalue = rvalue
        self.coord = coord
        self.uc_type = None

    def children(self):
        children = []
        if self.identifier is not None:
            children.append(("lvalue", self.lvalue))
        if self.value is not None:
            children.append(("rvalue", self.rvalue))
        return tuple(children)

    attr_names = ("op",)

class Break(Node):
    __slots__ = ("coord")    

    def __init__(self, coord=None):
        self.coord = coord

    def children(self):
        return ()

    attr_names = ()

class Cast(Node):
    __slots__ = ("type", "expression", "uc_type", "gen_location", "coord")

    def __init__(self, type, expression, coord=None):
        self.type = type
        self.expression = expression
        self.coord = coord
        self.uc_type = None
        self.gen_location = None

    def children(self):
        children = []
        if self.type is not None:
            children.append(("type", self.type))
        if self.expression is not None:
            children.append(("expression", self.expression))
        return tuple(children)

    attr_names = ()

class Compound(Node):
    __slots__ = ("declaration_opt_list", "statement_opt_list", "coord")    

    def __init__(self, declaration_opt_list, statement_opt_list, coord=None):
        self.declaration_opt_list = declaration_opt_list
        self.statement_opt_list = statement_opt_list
        self.coord = coord

    def children(self):
        children = []
        for i, child in enumerate(self.declaration_opt_list or []):
            children.append(("declaration_opt_list[%s]" % i, child))
        for i, child in enumerate(self.statement_opt_list or []):
            children.append(("statement_opt_list[%s]" % i, child))
        return tuple(children)

    attr_names = ()

class Decl(Node):
    __slots__ = ("name", "type", "init_declarator_list_opt", "uc_type", "cfg", "coord")

    def __init__(self, name, type, init_declarator_list_opt, coord=None):
        self.name = name
        self.type = type
        self.init_declarator_list_opt = init_declarator_list_opt
        self.coord = coord
        self.uc_type = None
        self.cfg = None

    def children(self):
        children = []
        if self.type is not None:
            children.append(("type", self.type))
        if self.init_declarator_list_opt is not None:
            children.append(("init_declarator_list_opt", self.init_declarator_list_opt))
        return tuple(children)

    attr_names = ("name",)

class DeclList(Node):
    __slots__ = ("declaration_list", "coord")    

    def __init__(self, declaration_list, coord=None):
        self.declaration_list = declaration_list
        self.coord = coord

    def children(self):
        children = []
        for i, child in enumerate(self.declaration_list or []):
            children.append(("declaration_list[%s]" % i, child))
        return tuple(children)

    attr_names = ()

class EmptyStatement(Node):
    __slots__ = ("coord")    

    def __init__(self, coord=None):
        self.coord = coord

    def children(self):
        return ()

    attr_names = ()

class ExprList(Node):
    __slots__ = ("expression_list", "coord")

    def __init__(self, expression_list, coord=None):
        self.expression_list = expression_list
        self.coord = coord

    def children(self):
        children = []
        for i, child in enumerate(self.expression_list or []):
            children.append(("expression_list[%s]" % i, child))
        return tuple(children)

    attr_names = ()

class For(Node):
    __slots__ = ("declaration", "expression_opt_cond", "expression_opt_iter", "statement", "coord")    

    def __init__(self, declaration, expression_opt_cond, expression_opt_iter, statement, coord=None):
        self.declaration = declaration
        self.expression_opt_cond = expression_opt_cond
        self.expression_opt_iter = expression_opt_iter
        self.statement = statement
        self.coord = coord

    def children(self):
        children = []
        if self.declaration is not None:
            children.append(("declaration", self.declaration))
        if self.expression_opt_cond is not None:
            children.append(("expression_opt_cond", self.expression_opt_cond))
        if self.expression_opt_iter is not None:
            children.append(("expression_opt_iter", self.expression_opt_iter))
        if self.statement is not None:
            children.append(("statement", self.statement))
        return tuple(children)

    attr_names = ()

class FuncCall(Node):
    __slots__ = ("identifier", "arg_list", "type", "uc_type", "gen_location", "coord")

    def __init__(self, identifier, arg_list, coord=None):
        self.identifier = identifier
        self.arg_list = arg_list
        self.coord = coord
        self.gen_location = None
        self.uc_type = None
        self.type = None

    def children(self):
        children = []
        if self.identifier is not None:
            children.append(("identifier", self.identifier))
        if self.arg_list is not None:
            children.append(("arg_list", self.arg_list))
        return tuple(children)

    attr_names = ()

class FuncDecl(Node):
    __slots__ = ("agr_list", "type", "vardecl", "uc_type", "coord")

    def __init__(self, agr_list, type, vardecl, coord=None):
        self.type = type
        self.agr_list = agr_list
        self.coord = coord
        self.vardecl = vardecl
        self.uc_type = None

    def children(self):
        children = []
        if self.agr_list is not None:
            children.append(("agr_list", self.agr_list))
        if self.type is not None:
            children.append(("type", self.type))
        return tuple(children)

    attr_names = ()

class FuncDef(Node):
    __slots__ = ("type_specifier", "declarator", "declaration_opt_list", "compound_statement", "cfg", "uc_type", "coord")

    def __init__(self, type_specifier, declarator, declaration_opt_list, compound_statement, coord=None):
        self.type_specifier = type_specifier
        self.declarator = declarator
        self.declaration_opt_list = declaration_opt_list
        self.compound_statement = compound_statement
        self.coord = coord
        self.cfg = None
        self.uc_type = None

    def children(self):
        children = []
        if self.type_specifier is not None:
            children.append(("type_specifier", self.type_specifier))
        if self.declarator is not None:
            children.append(("declarator", self.declarator))
        if self.compound_statement is not None:
            children.append(("compound_statement", self.compound_statement))
        for i, child in enumerate(self.declaration_opt_list or []):
            children.append(("declaration_opt_list[%s]" % i, child))
        return tuple(children)

    attr_names = ()

class GlobalDecl(Node):
    __slots__ = ("declaration_list", "coord")

    def __init__(self, declaration_list, coord=None):
        self.declaration_list = declaration_list
        self.coord = coord

    def children(self):
        children = []
        for i, child in enumerate(self.declaration_list or []):
            children.append(("declaration_list[%s]" % i, child))
        return tuple(children)

    attr_names = ()

class If(Node):
    __slots__ = ("expression", "statement_if", "statement_else", "cfg", "coord")

    def __init__(self, expression, statement_if, statement_else, coord=None):
        self.expression = expression
        self.statement_if = statement_if
        self.statement_else = statement_else
        self.coord = coord
        self.cfg = None

    def children(self):
        children = []
        if self.expression is not None:
            children.append(("expression", self.expression))
        if self.statement_if is not None:
            children.append(("statement_if", self.statement_if))
        if self.statement_else is not None:
            children.append(("statement_else", self.statement_else))
        return tuple(children)

    attr_names = ()

class InitList(Node):
    __slots__ = ("declarator_list", "coord")    

    def __init__(self, declarator_list, coord=None):
        self.declarator_list = declarator_list
        self.coord = coord

    def children(self):
        children = []
        for i, child in enumerate(self.declarator_list or []):
            children.append(("declarator_list[%s]" % i, child))
        return tuple(children)

    attr_names = ()

class ParamList(Node):
    __slots__ = ("parameter_list", "coord")  

    def __init__(self, parameter_list, coord=None):
        self.parameter_list = parameter_list
        self.coord = coord

    def children(self):
        children = []
        for i, child in enumerate(self.parameter_list or []):
            children.append(("parameter_list[%s]" % i, child))
        return tuple(children)

    attr_names = ()

class Print(Node):
    __slots__ = ("expression", "coord")  

    def __init__(self, expression, coord=None):
        self.expression = expression
        self.coord = coord

    def children(self):
        children = []
        if self.expression is not None:
            children.append(("expression", self.expression))
        return tuple(children)

    attr_names = ()

class Read(Node):
    __slots__ = ("expression", "coord")

    def __init__(self, expression, coord=None):
        self.expression = expression
        self.coord = coord

    def children(self):
        children = []
        if self.expression is not None:
            children.append(("expression", self.expression))
        return tuple(children)

    attr_names = ()

class Return(Node):
    __slots__ = ("expression", "coord")  

    def __init__(self, expression, coord=None):
        self.expression = expression
        self.coord = coord

    def children(self):
        children = []
        if self.expression is not None:
            children.append(("expression", self.expression))
        return tuple(children)

    attr_names = ()

class Type(Node):
    __slots__ = ("name", "coord")    

    def __init__(self, name, coord=None):
        self.name = name
        self.coord = coord

    def children(self):
        return ()

    attr_names = ("name",)

class UnaryOp(Node):
    __slots__ = ("op", "expression", "uc_type", "gen_location", "coord")

    def __init__(self, op, expression, coord=None):
        self.op = op
        self.expression = expression
        self.coord = coord
        self.uc_type = None
        self.gen_location = None

    def children(self):
        children = []
        if self.expression is not None:
            children.append(("expression", self.expression))
        return tuple(children)

    attr_names = ("op",)

class VarDecl(Node):
    __slots__ = ("declname", "decl", "type", "coord")

    def __init__(self, declname, decl, coord=None):
        self.declname = declname
        self.decl = decl
        self.type = type
        self.coord = coord

    def children(self):
        children = []
        if self.type is not None:
            children.append(("type", self.type))
        return tuple(children)

    attr_names = ()

class While(Node):
    __slots__ = ("expression", "statement", "coord")    

    def __init__(self, expression, statement, coord=None):
        self.expression = expression
        self.statement = statement
        self.coord = coord

    def children(self):
        children = []
        if self.expression is not None:
            children.append(("expression", self.expression))
        if self.statement is not None:
            children.append(("statement", self.statement))
        return tuple(children)

    attr_names = ()