import argparse
import pathlib
import sys
from uc.uc_ast import ID, InitList, Constant, ExprList, Return, ArrayDecl, ArrayRef, Compound, If, BinaryOp, FuncCall
from uc.uc_parser import UCParser
from uc.uc_type import CharType, FloatType, IntType, BoolType, StringType, VoidType, ArrayType, FunctionType

current_func_def = None
loop_def = 0
func_def = 0
current_scope = 1
current_params = []
array_curr_dim = None
class SymbolTable(dict):
    """Class representing a symbol table. It should provide functionality
    for adding and looking up nodes associated with identifiers.
    """

    def __init__(self):
        super().__init__()

    def add(self, name, value, is_global=False):
        global current_func_def
        global current_scope

        if current_func_def is not None and not is_global:
            key = str((name, current_func_def["name"], current_scope))
            self[key] = value
        else:
            self[name] = value

    def get_value_scope(self, name, scope):
        if scope is None or scope == 0:
            return None, None

        key = str((name, current_func_def["name"], scope))
        current_func_value = self.get(key, None)
        if current_func_value is not None:
            return current_func_value, scope

        return self.get_value_scope(name, scope - 1)

    def lookup(self, name, is_global=False):
        global current_func_def
        global current_scope

        value, scope = None, None
        if current_func_def is not None and not is_global:
            value, scope = self.get_value_scope(name, current_scope)

        if value is not None:
            return value, scope

        return self.get(name, None), current_scope

class NodeVisitor:
    """A base NodeVisitor class for visiting uc_ast nodes.
    Subclass it and define your own visit_XXX methods, where
    XXX is the class name you want to visit with these
    methods.
    """

    _method_cache = None

    def visit(self, node):
        """Visit a node."""

        if self._method_cache is None:
            self._method_cache = {}

        visitor = self._method_cache.get(node.__class__.__name__, None)
        if visitor is None:
            method = "visit_" + node.__class__.__name__            
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[node.__class__.__name__] = visitor

        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a
        node. Implements preorder visiting of the node.
        """
        for _, child in node.children():
            self.visit(child)


class Visitor(NodeVisitor):
    """
    Program visitor class. This class uses the visitor pattern. You need to define methods
    of the form visit_NodeName() for each kind of AST node that you want to process.
    """

    def __init__(self):
        # Initialize the symbol table
        self.symtab = SymbolTable()
        self.typemap = {
            "int": IntType,
            "float": FloatType,
            "char": CharType,
            "bool": BoolType,
            "string": StringType,
            "void": VoidType,
        }

    def _assert_semantic(self, condition, msg_code, coord, name="", ltype="", rtype=""):
        """Check condition, if false print selected error message and exit"""
        error_msgs = {
            1: f"{name} is not defined",
            2: f"{ltype} must be of type(int)",
            3: "Expression must be of type(bool)",
            4: f"Cannot assign {rtype} to {ltype}",
            5: f"Assignment operator {name} is not supported by {ltype}",
            6: f"Binary operator {name} does not have matching LHS/RHS types",
            7: f"Binary operator {name} is not supported by {ltype}",
            8: "Break statement must be inside a loop",
            9: "Array dimension mismatch",
            10: f"Size mismatch on {name} initialization",
            11: f"{name} initialization type mismatch",
            12: f"{name} initialization must be a single element",
            13: "Lists have different sizes",
            14: "List & variable have different sizes",
            15: f"conditional expression is {ltype}, not type(bool)",
            16: f"{name} is not a function",
            17: f"no. arguments to call {name} function mismatch",
            18: f"Type mismatch with parameter {name}",
            19: "The condition expression must be of type(bool)",
            20: "Expression must be a constant",
            21: "Expression is not of basic type",
            22: f"{name} does not reference a variable of basic type",
            23: f"\n{name}\nIs not a variable",
            24: f"Return of {ltype} is incompatible with {rtype} function definition",
            25: f"Name {name} is already defined in this scope",
            26: f"Unary operator {name} is not supported",
            27: "Undefined error",
        }
        if not condition:
            msg = error_msgs.get(msg_code)
            print("SemanticError: %s %s" % (msg, coord), file=sys.stdout)
            sys.exit(1)

    def visit_Program(self, node):
        # Visit all of the global declarations
        for _decl in node.gdecls:
            self.visit(_decl)
        # TODO: Manage the symbol table

    def visit_GlobalDecl(self, node):
        for declaration in node.declaration_list:
            self.visit(declaration)

    def visit_VarDecl(self, node):
        global current_scope
        global current_params
        global func_def
        uc_type = self.visit(node.type)
        name = node.declname.name
        value, scope = self.symtab.lookup(name)

        self._assert_semantic(value is None or scope != current_scope or name in current_params or func_def == 1, 25, node.coord, name=name)
        if value is None or scope != current_scope:
            self.symtab.add(name, uc_type)

        node.declname.uc_type = uc_type        
        return uc_type

    def visit_Type(self, node):
        return self.typemap.get(node.name, None)

    def visit_ID(self, node):
        global current_scope
        _type, scope = self.symtab.lookup(node.name)        
        node.uc_type = _type
        node.scope = current_scope
        node.kind = None
        
        return _type

    def visit_Constant(self, node):
        node.uc_type = self.typemap.get(node.type, None)
        return node.uc_type

    def visit_Cast(self, node):
        self.visit(node.expression)
        node.uc_type = self.visit(node.type)

    def visit_BinaryOp(self, node):
        self.visit(node.lvalue)
        ltype = node.lvalue.uc_type
        self.visit(node.rvalue)
        rtype = node.rvalue.uc_type

        self._assert_semantic(
            ltype == rtype,
            6,
            node.coord,
            ltype=ltype,
            rtype=rtype,
            name=node.op
        )

        is_binary_op = hasattr(ltype, "binary_ops") and node.op in ltype.binary_ops
        is_rel_op = hasattr(ltype, "rel_ops") and node.op in ltype.rel_ops
        
        self._assert_semantic(
            is_binary_op or is_rel_op,
            7,
            node.coord,
            ltype="type({})".format(ltype.typename),
            name=node.op
        )

        node.uc_type = ltype if is_binary_op else (BoolType if is_rel_op else None)
        return node.uc_type

    def visit_Assignment(self, node):
        # visit right side
        self.visit(node.rvalue)
        rtype = node.rvalue.uc_type
        # visit left side (must be a location)
        _var = node.lvalue
        self.visit(_var)
        if isinstance(_var, ID):
            value, scope = self.symtab.lookup(_var.name)          
            self._assert_semantic(value is not None, 1, node.coord, name=_var.name)
        ltype = node.lvalue.uc_type
        # Check that assignment is allowed
        self._assert_semantic(ltype == rtype, 4, node.coord, ltype="type({})".format(ltype.typename), rtype="type({})".format(rtype.typename))
        # Check that assign_ops is supported by the type
        self._assert_semantic(
            node.op in ltype.assign_ops, 5, node.coord, name=node.op, ltype="type({})".format(ltype.typename)
        )
        node.uc_type = VoidType

    def visit_ArrayRef(self, node):
        subscript = node.rid
        self.visit(subscript)

        if isinstance(subscript, ID):
            value, scope = self.symtab.lookup(subscript.name)
            self._assert_semantic(value is not None, 1, node.coord, name=subscript.name)

        sub_type = subscript.uc_type
        self._assert_semantic(sub_type == IntType, 2, subscript.coord, ltype="type({})".format(sub_type.typename))

        self.visit(node.lid)
        if isinstance(node.lid, ArrayRef):
            node.uc_type = node.lid.lid.uc_type.type
        else:
            node.uc_type = node.lid.uc_type.type

        return node.uc_type

    def visit_ArrayDecl(self, node):
        self.visit(node.type)
        
        global array_curr_dim
        decl = None
        if(isinstance(node.vardecl, ArrayDecl)):                         
            decl = node.vardecl.vardecl.declname                 
                        
            dim1 = int(node.vardecl.dim.value) if node.vardecl.dim is not None else node.vardecl.dim                        
            dim2 = int(node.dim.value) if node.dim is not None else node.dim             
            self._assert_semantic((dim1 is not None and dim2 is not None) or (dim1 is None and dim2 is None),
                                    9, 
                                    node.vardecl.vardecl.coord)

            node.uc_type = ArrayType(decl.uc_type, (dim1, dim2))
            array_curr_dim = ArrayType(decl.uc_type, (dim1, dim2))
        else:            
            decl = node.vardecl.declname            
            dim = int(node.dim.value) if node.dim is not None else node.dim            
            
            if array_curr_dim is None:
                node.uc_type = ArrayType(decl.uc_type, dim)
            else:
                node.uc_type = array_curr_dim
        
        if node.dim is not None:
            self.visit(node.dim)            

        self.symtab.add(decl.name, node.uc_type)
        

    def visit_Compound(self, node):
        global current_scope
        current_scope += 1

        if node.declaration_opt_list is not None:
            for declaration in node.declaration_opt_list:
                self.visit(declaration)

        if node.statement_opt_list is not None:
            for statement in node.statement_opt_list:
                self.visit(statement)

    def visit_Decl(self, node):
        node.uc_type = self.visit(node.type)
        self.visit(node.name)                        

        if node.init_declarator_list_opt is not None:            
            if isinstance(node.init_declarator_list_opt, InitList):                             
                self._assert_semantic(
                    isinstance(node.name.uc_type, ArrayType),
                    12,
                    node.name.coord,
                    name=node.name.name
                )
               
                prev_len = None                
                len_list = []

                if node.type.dim is None:
                    node.type.dim = len(node.init_declarator_list_opt.declarator_list)

                for init_decl in node.init_declarator_list_opt.declarator_list:                                    
                    self.visit(init_decl)                    
                    if isinstance(init_decl, InitList):
                        if isinstance(node.type.dim, int):
                            node.type.dim = [node.type.dim, len(init_decl.declarator_list)]
                        for init_decl_1 in init_decl.declarator_list:
                            self.visit(init_decl_1)
                        
                        if prev_len is not None:                            
                            self._assert_semantic(
                                prev_len == len(init_decl.declarator_list),
                                13,
                                node.name.coord
                            )   

                        prev_len = len(init_decl.declarator_list)
                        len_list.append(len(init_decl.declarator_list))
                
                for length in len_list:
                    self._assert_semantic(
                        node.name.uc_type.size[1] is None or node.name.uc_type.size[1] == length,
                        14,
                        node.name.coord
                    )             
                        
                dim1 = node.name.uc_type.size[0] if type(node.name.uc_type.size) == tuple else node.name.uc_type.size
                self._assert_semantic(
                    node.name.uc_type.size is None or dim1 is None or dim1 == len(node.init_declarator_list_opt.declarator_list),
                    14,
                    node.name.coord
                )
                
                for init_decl in node.init_declarator_list_opt.declarator_list:                                      
                    self._assert_semantic(
                        isinstance(init_decl, Constant) or isinstance(init_decl, InitList),
                        20,
                        init_decl.coord,
                    )
                
            else:                
                self.visit(node.init_declarator_list_opt)   
               
                if isinstance(node.name.uc_type, ArrayType) and isinstance(node.init_declarator_list_opt, Constant):
                    self._assert_semantic(
                        node.name.uc_type.size is None or node.name.uc_type.size == len(node.init_declarator_list_opt.value),
                        10,
                        node.name.coord,
                        name=node.name.name)
                                
                self._assert_semantic(
                    (isinstance(node.name.uc_type, ArrayType) and node.name.uc_type.type.typename == 'char' and
                    node.init_declarator_list_opt.uc_type.typename == 'string') or \
                        node.name.uc_type == node.init_declarator_list_opt.uc_type,
                        11,
                        node.name.coord,
                        name=node.name.name
                    )        
        
        return node.uc_type

    def has_return(self, compound_statement):
        has_return = False
        if compound_statement is not None and isinstance(compound_statement, Compound) and compound_statement.statement_opt_list is not None:
            for statement in compound_statement.statement_opt_list:
                if isinstance(statement, Return):
                    return True
                if isinstance(statement, Compound):
                    has_return = self.has_return(statement)
                if isinstance(statement, If):
                    has_return = self.has_return(statement.statement_if) or self.has_return(statement.statement_else)
        return has_return

    def visit_FuncDef(self, node):
        global current_func_def
        global current_scope
        global current_params
        global func_def
        current_scope = 1
        func_def = 1
        uc_type = self.visit(node.type_specifier)
        node.uc_type = uc_type

        current_func_def = {"name": node.declarator.name.name}
        self.visit(node.declarator)        

        decl_list = []
        if node.declaration_opt_list is not None:
            decl_list = node.declaration_opt_list
            for declaration in node.declaration_opt_list:
                self.visit(declaration.type)

        current_func_def["type"] = node.declarator.uc_type
        current_func_def["decl_list"] = decl_list
        self.visit(node.compound_statement)
                    
        uc_type = uc_type if uc_type is not None else VoidType
        # TODO corrigir para caso o tipo do retorno seja Void, pois essa condição sempre falhará
        self._assert_semantic(
            not (uc_type != VoidType and not self.has_return(node.compound_statement)),
            24,
            node.compound_statement.coord,
            ltype="type({})".format(VoidType.typename),
            rtype="type({})".format(uc_type.typename)
        )
        current_params = []
        return uc_type

    @staticmethod
    def get_expr_name(expr):
        if isinstance(expr, BinaryOp):
            return expr.lvalue.name

        if isinstance(expr, Constant):
            return expr.value

        if isinstance(expr, ArrayRef):
            if isinstance(expr.lid, ArrayRef):
                return expr.lid.lid.name
            return expr.lid.name

        if isinstance(expr, FuncCall):
            return expr.identifier.name

        return expr.name

    def visit_FuncCall(self, node):
        global current_func_def

        identifier = node.identifier
        self.visit(identifier)

        def_type = self.symtab.get(identifier.name)
        self._assert_semantic(
            isinstance(def_type, FunctionType),
            16,
            node.coord,
            name=identifier.name
        )

        expression_list = []
        if node.arg_list is not None:
            if isinstance(node.arg_list, ExprList):
                expression_list = node.arg_list.expression_list
            else:
                expression_list = [node.arg_list]

        self._assert_semantic(
            len(expression_list) == len(def_type.parameter_types),
            17,
            node.coord,
            name=identifier.name
        )

        for expr, decl_type in zip(expression_list, def_type.parameter_types):
            self.visit(expr)
            self._assert_semantic(
                expr.uc_type == decl_type,
                18,
                expr.coord,
                name=Visitor.get_expr_name(expr)
            )

        node.uc_type = def_type.return_type
        return node.uc_type

    def visit_Return(self, node):
        global current_func_def
        def_type = current_func_def["type"].return_type

        if node.expression is not None:
            self.visit(node.expression)
            expr_type = node.expression.uc_type            
            self._assert_semantic(
                expr_type == def_type,
                24,
                node.coord,                
                ltype="type({})".format(expr_type.typename),
                rtype="type({})".format(def_type.typename)
            )

    def visit_If(self, node):
        self.visit(node.expression)
        
        self._assert_semantic(
            node.expression.uc_type == BoolType,
            19,
            node.expression.coord
        )
        
        if node.statement_if is not None:
            self.visit(node.statement_if)
        if node.statement_else is not None:
            self.visit(node.statement_else)

    
    def visit_Break(self, node):        
        global loop_def
        self._assert_semantic(
            loop_def > 0,
            8,
            node.coord
        )

    def visit_For(self, node):
        #First, append the current loop node to the dedicated list attribute used to bind the node to nested break statement
        global current_scope
        global loop_def
        current_scope += 1
        loop_def += 1

        self.visit(node.declaration)

        if node.expression_opt_cond is not None:
            self.visit(node.expression_opt_cond)
            self._assert_semantic(
                node.expression_opt_cond.uc_type == BoolType,
                19,
                node.coord)

        if node.expression_opt_iter is None:
            self.visit(node.expression_opt_iter)

        self.visit(node.statement)
        loop_def -= 1

    def visit_While(self, node):
        #First, append the current loop node to the dedicated list attribute used to bind the node to nested break statement. Then, visit the condition and check if the conditional expression is of boolean type or return a type error. Finally, visit the body of the while (stmt).
        global current_scope
        global loop_def
        
        current_scope += 1        
        loop_def += 1

        self.visit(node.expression)
        self._assert_semantic(
            node.expression.uc_type == BoolType,
            15,
            node.coord,
            ltype="type({})".format(node.expression.uc_type.typename)
        )
        self.visit(node.statement)
        
        loop_def -= 1

    def visit_UnaryOp(self, node):
        self.visit(node.expression)       

        if node.expression.uc_type is not None:
            self._assert_semantic(
                node.op in node.expression.uc_type.unary_ops,
                26,
                node.coord,
                name=node.op,
            )

            node.uc_type = node.expression.uc_type

        if not isinstance(node.expression, Constant):
            expr_name = Visitor.get_expr_name(node.expression)
            value, scope = self.symtab.lookup(expr_name)            
            self._assert_semantic(value is not None, 1, node.coord, name=expr_name)

    def visit_Assert(self, node): 
        self.visit(node.expression)

        self._assert_semantic(
            node.expression.uc_type == BoolType,
            3,
            node.expression.coord            
        )

    def visit_EmptyStatement(self, node):
        pass

    def visit_Print(self, node):        
        if node.expression is not None:
            exprs = None
            if isinstance(node.expression, ExprList):
                exprs = node.expression.expression_list
            else:
                exprs = [node.expression]
                
            for expr in exprs:
                self.visit(expr)

                if not isinstance(expr, ArrayRef) and not isinstance(expr, FuncCall):
                    self._assert_semantic(
                        expr.uc_type in {IntType, FloatType, CharType, BoolType, StringType},
                        22,
                        expr.coord,
                        name=Visitor.get_expr_name(expr)
                    ) 

                if isinstance(expr, ArrayRef) or isinstance(expr, FuncCall):
                    self._assert_semantic(
                        expr.uc_type in {IntType, FloatType, CharType, BoolType, StringType},
                        21,
                        expr.coord
                    )

    def visit_Read(self, node):
        if isinstance(node.expression, ExprList):
            for expr in node.expression.expression_list:
                self.visit(expr)
                self._assert_semantic(
                    not isinstance(expr, Constant),
                    23,
                    expr.coord,
                    name=expr                   
                )
        else:
            self.visit(node.expression)            
            self._assert_semantic(
                not isinstance(node.expression, Constant),
                23,
                node.expression.coord,
                name=node.expression                    
            )            

    def visit_ExprList(self, node):
        for exp in node.expression_list:
            self.visit(exp)
            
            name = exp.value if isinstance(exp, Constant) else exp.name 
            self._assert_semantic(
                self.symtab.lookup(name)[0] is not None,
                1,
                node.coord,
                name=name
            )

    def visit_ParamList(self, node):
        for param in node.parameter_list:
            self.visit(param)

    def visit_InitList(self, node):
        for element in node.declarator_list:
            self.visit(element)

            if not isinstance(element, InitList):
                self._assert_semantic(
                    isinstance(element, Constant),
                    20,
                    node.coord,
                )

    def visit_FuncDecl(self, node):
        global current_func_def
        global current_scope
        global current_params
        global func_def
        return_type = self.visit(node.type)

        parameter_types = []
        if node.agr_list is not None:
            for arg in node.agr_list.parameter_list:
                self.visit(arg)
                parameter_types.append(arg.uc_type)
                current_params.append(arg.name.name)

        func_type = FunctionType(return_type, parameter_types)
        self.symtab.add(node.vardecl.declname.name, func_type, is_global=True)
        node.uc_type = func_type
        func_def = 0
        return func_type

    def visit_DeclList(self, node):
        global current_func_def
        for decl in node.declaration_list:
            self.visit(decl)
            current_func_def["decl_list"].append(decl)
            
if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", help="Path to file to be semantically checked", type=str
    )
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    # set error function
    p = UCParser()
    # open file and parse it
    with open(input_path) as f:
        ast = p.parse(f.read())
        sema = Visitor()
        sema.visit(ast)
