import argparse
import pathlib
import sys

from uc import uc_sema
from uc.uc_ast import *
from uc.uc_block import CFG, BasicBlock, ConditionBlock, EmitBlocks, format_instruction
from uc.uc_interpreter import Interpreter
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor
from uc.uc_type import IntType, FloatType
from uc.uc_ast import ExprList

class CodeGenerator(NodeVisitor):
    """
    Node visitor class that creates 3-address encoded instruction sequences
    with Basic Blocks & Control Flow Graph.
    """

    def __init__(self, viewcfg):
        self.viewcfg = viewcfg
        self.current_block = []
        self.gen_location_map = {}
        self.global_declaration_map = {}
        self.local_declaration_map = {}
        self.current_func_args_map = {}
        self.current_func_name = None
        self.loops_stack = []
        self.is_assigment = False
        # version dictionary for temporaries. We use the name as a Key
        self.fname = "_glob_"
        self.versions = {self.fname: 0}

        # The generated code (list of tuples)
        # At the end of visit_program, we call each function definition to emit
        # the instructions inside basic blocks. The global instructions that
        # are stored in self.text are appended at beginning of the code
        self.code = []

        self.text = []  # Used for global declarations & constants (list, strings)

        self.binary_ops = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
            "%": "mod",
            "==": "eq",
            "!=": "ne",
            ">=": "ge",
            "<=": "le",
            ">": "gt",
            "<": "lt",
            "!": "not",
            "||": "or",
            "&&": "and",
        }

        # TODO: Complete if needed.

    def show(self, buf=sys.stdout):
        _str = ""
        for _code in self.code:
            _str += format_instruction(_code) + "\n"
        buf.write(_str)

    def new_temp(self):
        """
        Create a new temporary variable of a given scope (function name).
        """
        if self.fname not in self.versions:
            self.versions[self.fname] = 1

        if self.versions[self.fname] == 0:
            self.versions[self.fname] = 1

        name = "%" + "%d" % (self.versions[self.fname])
        self.versions[self.fname] += 1
        return name

    def new_text(self, typename):
        """
        Create a new literal constant on global section (text).
        """
        name = "@." + typename + "." + "%d" % (self.versions["_glob_"])
        self.versions["_glob_"] += 1
        return name

    def get_gen_location_and_type(self, expr):
        # expr_name = expr.name if not isinstance(expr, ArrayRef) else expr.lid.name
        expr_name = Visitor.get_expr_name(expr)

        expr_gen_location = None
        if isinstance(expr, (FuncCall, ArrayRef, Constant, BinaryOp)):
            expr_gen_location = expr.gen_location
        else:
            expr_gen_location = self.gen_location_map[expr_name][0]

        expr_type = self.gen_location_map[expr_name][1] if not isinstance(expr, Constant) else expr.type
        return expr_gen_location, expr_type

    # You must implement visit_Nodename methods for all of the other
    # AST nodes.  In your code, you will need to make instructions
    # and append them to the current block code list.
    #
    # A few sample methods follow. Do not hesitate to complete or change
    # them if needed.

    def visit_Constant(self, node):
        if node.type == "string":
            _target = self.new_text("str")
            inst = ("global_string", _target, node.value)
            self.text.append(inst)
        else:
            # Create a new temporary variable name
            _target = self.new_temp()
            # Make the SSA opcode and append to list of generated instructions
            inst = ("literal_" + node.type, node.value, _target)
            self.current_block.append(inst)
        # Save the name of the temporary variable where the value was placed
        node.gen_location = _target

    def visit_BinaryOp(self, node):
        # Visit the left and right expressions
        self.visit(node.lvalue)
        self.visit(node.rvalue)

        if isinstance(node.lvalue, FuncCall):
            target_left_value = self.new_temp()
            inst = ("load_{}".format(node.lvalue.uc_type.typename), node.lvalue.gen_location, target_left_value)
            self.current_block.append(inst)
        else:
            if self.local_declaration_map.get(node.lvalue.gen_location[1:], None) is not None or \
                    self.gen_location_map.get(node.lvalue.gen_location[1:], None) is not None:
                target_left_value = self.new_temp()
                if isinstance(node.lvalue, ID):
                    inst = ("load_{}".format(node.lvalue.type), node.lvalue.gen_location, target_left_value)
                else:
                    inst = ("load_{}".format(node.lvalue.uc_type.typename), node.lvalue.gen_location, target_left_value)
                self.current_block.append(inst)
            else:
                target_left_value = node.lvalue.gen_location

        if isinstance(node.rvalue, (Constant, ArrayRef)):
            target_right_value = node.rvalue.gen_location
        else:
            target_right_value = self.new_temp()
            inst = ("load_{}".format(node.rvalue.uc_type.typename), node.rvalue.gen_location, target_right_value)
            self.current_block.append(inst)

        # Make a new temporary for storing the result
        target = self.new_temp()

        # Create the opcode and append to list
        if isinstance(node.lvalue, ID):
            opcode = self.binary_ops[node.op] + "_" + node.lvalue.type
        else:
            opcode = self.binary_ops[node.op] + "_" + node.lvalue.uc_type.typename
        inst = (opcode, target_left_value, target_right_value, target)
        self.current_block.append(inst)

        # Store location of the result on the node
        node.gen_location = target

    def visit_Print(self, node):
        # Visit the expression
        if node.expression is None:
            inst = ("print_void",)
            self.current_block.append(inst)
        else:
            expr_list = node.expression.expression_list if isinstance(node.expression, ExprList) else [node.expression]

            for expr in expr_list:
                self.visit(expr)

                expr_gen_location, expr_type = self.get_gen_location_and_type(expr)
                if isinstance(expr, ArrayRef):
                    if isinstance(expr.lid, ArrayRef):
                        expr.lid.lid.gen_location = expr_gen_location
                        expr.lid.lid.type = expr_type
                    else:
                        expr.lid.gen_location = expr_gen_location
                        expr.lid.type = expr_type
                else:
                    expr.gen_location = expr_gen_location
                    if not isinstance(expr, BinaryOp):
                        expr.type = expr_type

                inst = ("print_{}".format(expr_type), expr_gen_location)
                self.current_block.append(inst)

    def visit_VarDecl(self, node):
        # Allocate on stack memory
        name = node.declname.name
        var_type = node.type.name
        _varname = "%" + name

        if name in self.global_declaration_map:
            _varname = "@" + name

            type = self.global_declaration_map[name].type.type
            if isinstance(type, VarDecl):
                type = type.type
            global_type = type.name
            init = self.global_declaration_map[name].init_declarator_list_opt
            if isinstance(init, InitList):
                global_value = [item.value for item in init.declarator_list]
                global_type += '_{}'.format(len(global_value))
            else:
                global_value = init.value
            inst = ("global_{}".format(global_type), "@{}".format(name), global_value)
            self.current_block.append(inst)
        else:
            if self.local_declaration_map.get(name, None) is None:
                inst = ("alloc_" + node.type.name, _varname)
                self.current_block.append(inst)


            if node.decl is not None and node.decl.init_declarator_list_opt is not None:
                self.visit(node.decl.init_declarator_list_opt)
                literal = node.decl.init_declarator_list_opt
                location = literal.gen_location if not isinstance(literal, UnaryOp) else literal.expression.gen_location
                inst = ("store_" + node.type.name, location, _varname)
                self.current_block.append(inst)

            # if node.decl is not None and node.decl.init_declarator_list_opt is not None:
            #     target = self.new_temp()
            #     inst = (
            #         "load_{}".format(node.type.name),
            #         node.decl.init_declarator_list_opt.gen_location,
            #         target
            #     )
            #     self.current_block.append(inst)
            #     inst = (
            #         "store_{}".format(node.type.name),
            #         target,
            #         _varname,
            #     )
            #     self.current_block.append(inst)

            node.declname.type = var_type

        self.gen_location_map[name] = (_varname, var_type)
        node.declname.gen_location = _varname

    def remove_block(self, block_def):
        if block_def not in self.code:
            return None

        index = self.code.index(block_def)
        self.code.pop(index)
        while index+1 < len(self.code) and 'define_' not in self.code[index][0]:
            self.code.pop(index)

        return index

    def visit_Program(self, node):
        # Visit all of the global declarations
        for _decl in node.gdecls:
            self.visit(_decl)
        # At the end of codegen, first init the self.code with
        # the list of global instructions allocated in self.text
        self.code = self.text.copy()
        # Also, copy the global instructions into the Program node
        node.text = self.text.copy()
        # After, visit all the function definitions and emit the
        # code stored inside basic blocks.
        for _decl in node.gdecls:
            if isinstance(_decl, FuncDef):
                # _decl.cfg contains the Control Flow Graph for the function
                # cfg points to start basic block
                bb = EmitBlocks()
                bb.visit(_decl.cfg)
                index = None
                i = 0
                for _code in bb.code:
                    if 'define_' in _code[0]:
                        index = self.remove_block(_code)
                        i = 0

                    if index is None:
                        self.code.append(_code)
                    else:
                        self.code.insert(index+i, _code)

                    i += 1

        if self.viewcfg:  # evaluate to True if -cfg flag is present in command line
            for _decl in node.gdecls:
                if isinstance(_decl, FuncDef):
                    dot = CFG(_decl.decl.name.name)
                    dot.view(_decl.cfg)  # _decl.cfg contains the CFG for the function

        # for code in self.code:
        #     print(code)

    # TODO: Complete.

    def send_func_return_instruction(self, body, return_type):
        stmt_list = body.statement_opt_list if isinstance(body, Compound) else [body]

        for stmt in stmt_list:
            if isinstance(stmt, Return):
                inst = ("return_{}".format(return_type), stmt.expression.gen_location)
                self.current_block.append(inst)
                break

    def visit_FuncDef(self, node):
        func_block = BasicBlock(node.declarator.name.name)
        self.current_func_name = node.declarator.name.name

        self.visit(node.declarator)

        args = node.declarator.type.agr_list
        if args is not None:
            for declaration in args.parameter_list:
                self.visit(declaration)

        if node.declarator.init_declarator_list_opt is not None:
            self.visit(node.declarator.init_declarator_list_opt)

        self.visit(node.compound_statement)

        return_type = node.type_specifier.name
        if return_type == "void":
            inst = ("return_void",)
            self.current_block.append(inst)
        # else:
        #     for stmt in node.compound_statement.statement_opt_list:
        #         if isinstance(stmt, Return):
        #             inst = ("return_{}".format(return_type), stmt.expression.gen_location)
        #             self.current_block.append(inst)
        #         elif isinstance(stmt, If):
        #             self.send_func_return_instruction(stmt.statement_if, return_type)
        #             self.send_func_return_instruction(stmt.statement_else, return_type)
        #         elif isinstance(stmt, For) or isinstance(stmt, While):
        #             self.send_func_return_instruction(stmt.statement, return_type)

        for inst in self.current_block:
            func_block.append(inst)
        node.cfg = func_block

        self.versions = {self.fname: 0}
        self.current_func_args_map = {}
        self.current_block = []

    def visit_ParamList(self, node):
        for param in node.parameter_list:
            self.visit(param)

    def visit_GlobalDecl(self, node):
        for declaration in node.declaration_list:
            if isinstance(declaration, Decl):
                self.global_declaration_map[declaration.name.name] = declaration
                self.visit(declaration)

    def visit_Decl(self, node):
        if isinstance(node.type, ArrayDecl):
            self.local_declaration_map[node.name.name] = node
        self.visit(node.type)
        self.visit(node.name)

        if isinstance(node.type, ArrayDecl):
            #handle arrays
            if node.init_declarator_list_opt is None:
                array_type = node.type.vardecl.type.name
                array_dim = node.type.dim.value
                if isinstance(node.type.type, ArrayDecl): #handle matrix declaration
                    cols = node.type.type.dim.value
                    array_dim = "{}_{}".format(array_dim, cols)
                array_name = node.name.name
                inst = ("alloc_{}_{}".format(array_type, array_dim), "%" + array_name)
                self.current_block.append(inst)
            else:
                if isinstance(node.init_declarator_list_opt, InitList):
                    declaration = node.type
                    if isinstance(declaration.type, ArrayDecl): #handle matrix
                        init_list_row = node.init_declarator_list_opt.declarator_list
                        initial_values = [[col.value for col in row.declarator_list] for row in init_list_row]
                    else:
                        init_list = node.init_declarator_list_opt
                        initial_values = [item.value for item in init_list.declarator_list]

                    initial_values_target = self.new_text("const")
                    array_type = declaration.vardecl.type.name
                    if isinstance(declaration.dim, list):
                        array_dim = '_'.join([str(dim) for dim in declaration.dim])
                    else:
                        array_dim = declaration.dim

                    inst = ("global_{}_{}".format(array_type, array_dim), initial_values_target, initial_values)
                    self.current_block.insert(0, inst)
                    array_name = node.name.name
                    inst = ("alloc_{}_{}".format(array_type, array_dim), "%"+array_name)
                    self.current_block.append(inst)
                    inst = ("store_{}_{}".format(array_type, array_dim), initial_values_target, "%"+array_name)
                    self.current_block.append(inst)
                elif isinstance(node.init_declarator_list_opt, Constant): #handle strings
                    string_name = node.name.name
                    initial_value = node.init_declarator_list_opt.value
                    string_len = len(initial_value)
                    initial_value_target = self.new_text("str")
                    inst = ("global_string", initial_value_target, initial_value)
                    self.current_block.insert(0, inst)
                    inst = ("alloc_char_{}".format(string_len), "%"+string_name)
                    self.current_block.append(inst)
                    inst = ("store_char_{}".format(string_len), initial_value_target, "%"+string_name)
                    self.current_block.append(inst)

    def visit_ArrayDecl(self, node):
        self.visit(node.type)

        if isinstance(node.type, ArrayDecl):
            array_decl = node.type.type.declname
        else:
            array_decl = node.type.declname
        array_name = array_decl.name
        if self.global_declaration_map.get(array_name, None) is None:
            self.local_declaration_map[array_name] = node


    def visit_FuncDecl(self, node):
        self.local_declaration_map = {}
        args_define = []
        args_full_info = []
        if node.agr_list is not None:
            for arg in node.agr_list.parameter_list:
                #self.visit(arg)
                target = self.new_temp()
                args_define.append((arg.type.type.name, target))
                args_full_info.append((arg.type.type.name, target, arg.name.name))

        return_type = node.type.type.name
        func_name = node.type.declname.name

        inst = ("define_{}".format(return_type), '@{}'.format(func_name), args_define)
        self.current_block.append(inst)

        inst = ("entry:",)
        self.current_block.append(inst)

        if return_type != "void":
            return_target = self.new_temp()
            inst = ("alloc_{}".format(return_type), return_target)
            self.current_block.append(inst)

            self.gen_location_map[func_name] = (return_target, return_type)

        for arg in args_full_info:
            inst = ("alloc_{}".format(arg[0]), "%{}".format(arg[2]))
            self.current_block.append(inst)

        for arg in args_full_info:
            inst = ("store_{}".format(arg[0]), arg[1], "%{}".format(arg[2]))
            self.current_block.append(inst)
            #store a map from name to register
            self.local_declaration_map[arg[2]] = arg[1]

        # for arg in args_full_info:
        #     new_target = self.new_temp()
        #     arg_type = arg[0]
        #     arg_name = arg[2]
        #
        #     inst = ("load_{}".format(arg_type), "%{}".format(arg_name), new_target)
        #     self.current_block.append(inst)
        #     self.gen_location_map[arg_name] = (new_target, arg_type)
        #     self.current_func_args_map[arg_name] = arg_name

    def visit_DeclList(self, node):
        if node.declaration_list is not None:
            for declaration in node.declaration_list:
                self.visit(declaration)

    def visit_Type(self, node):
        pass

    def visit_If(self, node):
        self.visit(node.expression)
        if_target = self.new_temp()
        else_target = self.new_temp()

        inst = ("cbranch", node.expression.gen_location, if_target, else_target)
        self.current_block.append(inst)

        self.current_block.append((if_target[1:]+":", ))
        self.visit(node.statement_if)
        end = else_target
        if node.statement_else is not None:
            end = self.new_temp()
            self.current_block.append(("jump", end))
            self.current_block.append((else_target[1:]+":", ))
            self.visit(node.statement_else)

        self.current_block.append((end[1:]+":", ))

        cb = ConditionBlock("if")
        cb.next_block = BasicBlock(if_target)
        cb.fall_through = BasicBlock(else_target)
        node.cfg = cb

    def visit_For(self, node):
        if node.declaration is not None:
            self.visit(node.declaration)

        begin_label = self.new_temp()
        body_label = self.new_temp()
        end_label = self.new_temp()
        self.current_block.append((begin_label[1:]+":", ))

        if node.expression_opt_cond is not None:
            self.visit(node.expression_opt_cond)
            inst = ("cbranch", node.expression_opt_cond.gen_location, body_label, end_label)
            self.current_block.append(inst)
            self.loops_stack.append(end_label)

        self.current_block.append((body_label[1:] + ":",))
        self.visit(node.statement)

        if node.expression_opt_iter is not None:
            self.visit(node.expression_opt_iter)

        self.current_block.append(("jump", begin_label))
        self.current_block.append((end_label[1:] + ":",))
        if len(self.loops_stack) > 0:
            self.loops_stack.pop()

    def visit_While(self, node):
        begin_label = self.new_temp()
        body_label = self.new_temp()
        end_label = self.new_temp()
        self.current_block.append((begin_label[1:]+":", ))

        if node.expression is not None:
            self.visit(node.expression)
            inst = ("cbranch", node.expression.gen_location, body_label, end_label)
            self.current_block.append(inst)
            self.loops_stack.append(end_label)

        self.current_block.append((body_label[1:] + ":",))
        self.visit(node.statement)

        self.current_block.append(("jump", begin_label))
        self.current_block.append((end_label[1:] + ":",))
        if len(self.loops_stack) > 0:
            self.loops_stack.pop()

    def visit_Compound(self, node):
        if node.declaration_opt_list is not None:
            for decl in node.declaration_opt_list:
                self.visit(decl)

        if node.statement_opt_list is not None:
            for stmt in node.statement_opt_list:
                self.visit(stmt)

    def visit_Assignment(self, node):
        self.is_assigment = True
        self.visit(node.lvalue)
        self.is_assigment = False
        if isinstance(node.rvalue, UnaryOp) and len(node.rvalue.op) == 3:
            self.visit(node.rvalue.expression)
        else:
            self.visit(node.rvalue)

        if isinstance(node.rvalue, BinaryOp):
            lvalue = node.rvalue.lvalue
            while isinstance(lvalue, BinaryOp):
                lvalue = lvalue.lvalue            

            assignment_type = lvalue.type if not isinstance(lvalue, ArrayRef) else lvalue.uc_type.typename
        else:
            assignment_type = node.rvalue.uc_type.typename

        if isinstance(node.rvalue, UnaryOp) and len(node.rvalue.op) == 3:
            target_rvalue = self.new_temp()
            rvalue_gen_location = node.rvalue.gen_location if isinstance(node.rvalue, Constant) else node.rvalue.expression.gen_location

            inst = ("load_{}".format(assignment_type), rvalue_gen_location, target_rvalue)
            self.current_block.append(inst)

            inst = ("store_{}".format(assignment_type), target_rvalue, node.lvalue.gen_location)
            self.current_block.append(inst)
            self.visit(node.rvalue)

        if isinstance(node.lvalue, ArrayRef):
            inst = ("store_{}_*".format(assignment_type), node.rvalue.gen_location, node.lvalue.gen_location)
            self.current_block.append(inst)
        elif not isinstance(node.rvalue, UnaryOp) or len(node.rvalue.op) < 3:
            if isinstance(node.rvalue, (Constant, BinaryOp, ArrayRef)):
                target_rvalue = node.rvalue.gen_location
            else:
                target_rvalue = self.new_temp()
                if isinstance(node.rvalue, ArrayRef):
                    inst = ("load_{}_*".format(assignment_type), node.rvalue.gen_location, target_rvalue)
                else:
                    inst = ("load_{}".format(assignment_type), node.rvalue.gen_location, target_rvalue)
                self.current_block.append(inst)

            inst = ("store_{}".format(assignment_type), target_rvalue, node.lvalue.gen_location)
            self.current_block.append(inst)

    def visit_Break(self, node):
        end_label = self.loops_stack.pop()
        self.current_block.append(("jump", end_label))


    def visit_FuncCall(self, node):
        if node.arg_list is not None:
            expr_list = node.arg_list.expression_list if isinstance(node.arg_list, ExprList) else [node.arg_list]
            target_list = []
            for expr in expr_list:
                self.visit(expr)
                expr_gen_location, expr_type = self.get_gen_location_and_type(expr)

                target = self.new_temp()
                inst = ("load_{}".format(expr_type), expr_gen_location, target)
                self.current_block.append(inst)
                target_list.append(target)

            for (target, expr) in zip(target_list, expr_list):
                inst = ("param_{}".format(expr_type), target)
                self.current_block.append(inst)

        func_name = node.identifier.name
        return_target = self.gen_location_map[func_name][0]
        return_type = self.gen_location_map[func_name][1]

        inst = None
        if return_type == "void":
            inst = ("call_{}".format(return_type), "@{}".format(func_name),)
        else:
            target = self.new_temp()
            inst = ("call_{}".format(return_type), "@{}".format(func_name), target)
            
            node.gen_location = target
            node.type = return_type 
        
        self.current_block.append(inst)

    def visit_Assert(self, node):
        self.visit(node.expression)
        true_target = self.new_temp()
        false_target = self.new_temp()
        end = self.new_temp()
        
        inst = ("cbranch", node.expression.gen_location, true_target, false_target)
        self.current_block.append(inst)
        self.current_block.append((true_target[1:] + ":",))
        self.current_block.append(("jump", end))

        self.current_block.append((false_target[1:] + ":",))
        message = f"assertion_fail on {node.expression.coord.line}:{node.expression.coord.column}"
        assert_message = f"@assert_message_{node.coord.line}_{node.coord.column}"
        self.current_block.insert(0, ("global_string", assert_message, message))
        self.current_block.append(("print_string", assert_message))
        self.current_block.append(("jump", "%exit"))

        self.current_block.append((end[1:] + ":",))

        cb = ConditionBlock("assert")
        cb.next_block = BasicBlock(true_target)
        cb.fall_through = BasicBlock(false_target)
        node.cfg = cb

    def visit_EmptyStatement(self, node):
        pass

    def visit_Read(self, node):
        raise NotImplementedError()

    def visit_Return(self, node):
        if node.expression is not None:
            self.visit(node.expression)
            '''
            load it if necessary and store its value to the return location.
            Then generate a jump to the return block if needed.
            Do not forget to update the predecessor of the return block.
            '''
            return_reserved_target = self.gen_location_map[self.current_func_name][0]
            expr_type = node.expression.type if not isinstance(node.expression, BinaryOp) else node.expression.lvalue.type

            inst = ("store_{}".format(expr_type), node.expression.gen_location, return_reserved_target)
            self.current_block.append(inst)
            inst = ("jump", "%exit")
            self.current_block.append(inst)
            inst = ("exit:",)
            self.current_block.append(inst)
            target = self.new_temp()
            inst = ("load_{}".format(expr_type), return_reserved_target, target)
            self.current_block.append(inst)
            inst = ("return_{}".format(expr_type), target)
            self.current_block.append(inst)

            node.expression.gen_location = target
        else:
            inst = ("jump", "%exit")
            self.current_block.append(inst)
            inst = ("exit:",)
            self.current_block.append(inst)

    def visit_ID(self, node):
        if node.name in self.gen_location_map:
            node.gen_location = self.gen_location_map[node.name][0]
            node.type = self.gen_location_map[node.name][1]

    def visit_Cast(self, node):
        self.visit(node.expression)
        value = Visitor.get_expr_name(node.expression)
        target = self.new_temp()
        if node.type.name == "int":
            inst = ('fptosi', "%{}".format(value), target)
        else:
            inst = ('sitofp', "%{}".format(value), target)

        self.current_block.append(inst)
        node.gen_location = target

    def visit_UnaryOp(self, node):
        self.visit(node.expression)
        expr_reg = node.expression.gen_location
        expr_type = node.expression.type if not isinstance(node.expression, BinaryOp) else node.expression.uc_type.typename
        unary_target = None
        if node.op[-2:] in ['++', '--'] or node.op[-3:] in ['++', '--']:
            literal_reg = self.new_temp()
            unary_target = self.new_temp()
            inst = ("literal_"+expr_type, 1, literal_reg)
            self.current_block.append(inst)
            opcode = self.binary_ops[node.op[-1]] + "_" + expr_type
            temp = self.new_temp()
            inst = ("load_"+expr_type, expr_reg, temp)
            self.current_block.append(inst)
            result = self.new_temp()
            inst = ("alloc_"+expr_type, result)
            self.current_block.append(inst)
            inst = ("store_"+expr_type, expr_reg, result)
            self.current_block.append(inst)

            inst = (opcode, temp, literal_reg, unary_target)
            self.current_block.append(inst)

            inst = ("store_" + expr_type, unary_target, expr_reg)
            self.current_block.append(inst)
            node.gen_location = unary_target
        elif node.op == '!':
            unary_target = self.new_temp()
            self.current_block.append(("not_bool", expr_reg, unary_target))
            node.gen_location = unary_target
        elif node.op == '-':
            zero_register = self.new_temp()
            target = self.new_temp()
            self.current_block.append(("literal_"+expr_type, 0, zero_register))
            opcode = self.binary_ops['-'] + "_" + expr_type
            temp = self.new_temp()
            inst = ("load_"+expr_type, expr_reg, temp)
            self.current_block.append(inst)
            inst = (opcode, zero_register, temp, target)
            self.current_block.append(inst)
            node.expression.gen_location = target

    def visit_ExprList(self, node):
        pass

    
    def visit_ArrayRef(self, node):
        self.visit(node.lid)
        self.visit(node.rid)

        if isinstance(node.lid, ArrayRef): #handle matrix
            curr_node = node.lid
            self.visit(curr_node.lid)
            self.visit(curr_node.rid)

            if isinstance(curr_node.rid, BinaryOp):
                right_type = curr_node.rid.uc_type.typename
            else:
                right_type = curr_node.rid.type

            row_offset = self.new_temp()
            inst = ("load_{}".format(right_type), curr_node.rid.gen_location, row_offset)
            self.current_block.append(inst)
            size_reg = self.new_temp()
            matrix_name = curr_node.lid.name
            if self.global_declaration_map.get(matrix_name, None):
                matrix_decl = self.global_declaration_map[matrix_name]
            else:
                matrix_decl = self.local_declaration_map[matrix_name]

            if isinstance(matrix_decl.dim, Constant):
                col_count = matrix_decl.dim.value
            else:
                col_count = matrix_decl.dim[1]
            inst = ("literal_int", col_count, size_reg)
            self.current_block.append(inst)
            mult_result = self.new_temp()
            inst = ("mul_int", size_reg, row_offset, mult_result)
            self.current_block.append(inst)
            col_offset = self.new_temp()
            inst = ("load_{}".format(right_type), node.rid.gen_location, col_offset)
            self.current_block.append(inst)
            index_reg =  self.new_temp()
            inst = ("add_int", mult_result, col_offset, index_reg)
            self.current_block.append(inst)
            new_target = self.new_temp()
            inst = ("elem_{}".format(right_type), curr_node.lid.gen_location, index_reg, new_target)
            self.current_block.append(inst)
            if not self.is_assigment:
                ref_target = self.new_temp()
                inst = ("load_{}_*".format(curr_node.lid.type), new_target, ref_target)
                self.current_block.append(inst)
                node.gen_location = ref_target
            else:
                node.gen_location = new_target
        else:
            if isinstance(node.rid, BinaryOp):
                right_type = node.rid.uc_type.typename
            else:
                right_type = node.rid.type

            if isinstance(node.rid, (Constant, BinaryOp)):
                offset = node.rid.gen_location
            else:
                offset = self.new_temp()
                inst = ("load_{}".format(right_type), node.rid.gen_location, offset)
                self.current_block.append(inst)
            new_target = self.new_temp()
            inst = ("elem_{}".format(right_type), node.lid.gen_location, offset, new_target)
            self.current_block.append(inst)

            if not self.is_assigment:
                ref_target = self.new_temp()
                inst = ("load_{}_*".format(node.lid.type), new_target, ref_target)
                self.current_block.append(inst)
                node.gen_location = ref_target
            else:
                node.gen_location = new_target



    def visit_InitList(self, node):
        raise NotImplementedError()
    
if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate uCIR. By default, this script only runs the interpreter on the uCIR. \
              Use the other options for printing the uCIR, generating the CFG or for the debug mode.",
        type=str,
    )
    parser.add_argument(
        "--ir",
        help="Print uCIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--cfg", help="Show the cfg of the input_file.", action="store_true"
    )
    parser.add_argument(
        "--debug", help="Run interpreter in debug mode.", action="store_true"
    )
    args = parser.parse_args()

    print_ir = args.ir
    create_cfg = args.cfg
    interpreter_debug = args.debug

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

    gen = CodeGenerator(create_cfg)
    gen.visit(ast)
    gencode = gen.code

    if print_ir:
        print("Generated uCIR: --------")
        gen.show()
        print("------------------------\n")

    vm = Interpreter(interpreter_debug)
    vm.run(gencode)
