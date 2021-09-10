from __future__ import print_function
import jedi
import ast
from jedi.api.environment import get_default_environment
from pathlib import Path
import time
import argparse

# Find aliases for tensorflow packages
class FindTfImport(ast.NodeVisitor):
    def __init__(self, package="tensorflow"):
        self.package = package
        self.alias = [package]

    def visit_Import(self, node):
        for name in node.names:
            if name.name == self.package:
                self.alias.append(name.asname)

# convert python2 code to python3
def futurize_file(file_name):
    import os
    os.system("futurize --stage1 -w %s" % file_name)


class FileData:
    def __init__(self, path, project):
        try:
            with open(path) as f:
                data = f.read()
            self.tree = ast.parse(data, path)
            self.script = jedi.Script(data, path=path, project=project)
        except Exception as e:
            print(e)
            print('[Futurizing] %s' % path)
            try:
                futurize_file(path)
                with open(path) as f:
                    data = f.read()
                self.tree = ast.parse(data, path)
                self.script = jedi.Script(data, path=path, project=project)
            except:
                self.tree = None
                self.script = None
                print(e)

# Get file data with cache
def get_file_data(path, project, file_data_cache):
    if path not in file_data_cache:
        file_data_cache[path] = FileData(path, project)
    return file_data_cache[path]


class FuncDef:
    def __init__(self, path, func_name, project, file_data_cache):
        self.path = path
        self.__name__ = func_name
        self.def_node = None
        file_data = get_file_data(path, project, file_data_cache)
        self.tree = file_data.tree

    def get_func_def_node(self):
        if not self.def_node:
            class FuncDefExtractor(ast.NodeVisitor):
                def __init__(self, func_name):
                    self.def_node = None
                    self.__name__ = func_name

                def visit_FunctionDef(self, node):
                    if node.name == self.__name__:
                        self.def_node = node
                    else:
                        self.generic_visit(node)

            def_extractor = FuncDefExtractor(self.__name__)
            def_extractor.generic_visit(self.tree)
            self.def_node = def_extractor.def_node
        return self.def_node


class RandomCallExtractor(ast.NodeVisitor):
    def __init__(self):
        self.random_calls = []

    def visit_Call(self, node):
        if 'random' in get_func_prefix(node)[0]:
            self.random_calls.append(node)
        self.generic_visit(node)

# Extract variables in the loop that should be excluded
class AssignVarExtractor(ast.NodeVisitor):
    def __init__(self, script, in_loop_lineno):
        self.assign_vars = []
        self.in_loop_lineno = in_loop_lineno
        self.script = script

    def contain_random_call(self, node):
        ex = RandomCallExtractor()
        ex.generic_visit(node)
        return len(ex.random_calls) > 0

    def visit_Assign(self, node):
        contain_random_call = self.contain_random_call(node)

        def fn_if_defined_outloop(node):
            if self.script.get_references(node.lineno, node.col_offset)[0].line < self.in_loop_lineno or contain_random_call:
                self.assign_vars.append(node)
        for target in node.targets:
            tuple_fn(fn_if_defined_outloop, target)

        self.generic_visit(node)

    def visit_AugAssign(self, node):
        contain_random_call = self.contain_random_call(node)

        def fn_if_defined_outloop(node):
            if self.script.get_references(node.lineno, node.col_offset)[0].line < self.in_loop_lineno or contain_random_call:
                self.assign_vars.append(node)

        tuple_fn(fn_if_defined_outloop, node.target)
        self.generic_visit(node)

    # Extract variables that defined outside the loop, and it calls class methods like "append"
    def visit_Call(self, node):
        if hasattr(node, "func") and hasattr(node.func, "value"):
            dec = self.script.goto(
                node.func.value.lineno, node.func.value.col_offset)
            if dec and dec[0].type == 'statement' and dec[0].line < self.in_loop_lineno:
                self.assign_vars.append(node.func.value)

        self.generic_visit(node)


def if_expand_func_def(func_def):
    return func_def.type == 'function' and func_def.module_name != 'builtins'


def get_func_prefix(func_call):
    full_prefix = ""
    last_prefix = ""
    func_call = func_call.func

    if hasattr(func_call, 'value'):
        while hasattr(func_call, 'value'):
            if hasattr(func_call, 'attr'):
                full_prefix = func_call.attr + '.' + full_prefix
                func_call = func_call.value
            else:
                break
        if hasattr(func_call, 'id'):
            last_prefix = func_call.id
    return full_prefix[:-1], last_prefix


def if_track_func_call(func_call, ops_set, tf_alias):

    full_prefix, last_prefix = get_func_prefix(func_call)
    if last_prefix in tf_alias and full_prefix in ops_set:
        return True

# Apply fn to basic ast nodes except for Tuple
def tuple_fn(fn, node):
    if isinstance(node, ast.Tuple):
        for name in node.elts:
            tuple_fn(fn, name)
    else:
        fn(node)

# Apply fn to basic ast nodes except for Tuple and List
def tuple_list_fn(fn, node):
    if isinstance(node, ast.Tuple) or isinstance(node, ast.List):
        for name in node.elts:
            tuple_list_fn(fn, name)
    else:
        fn(node)


class ExpandFuncExtractor(ast.NodeVisitor):

    # count: expand level of the current function
    # out_loop: set to true if current node is in the function that is called inside a loop
    # in_loop: set to true if current node is inside a loop in current function
    def __init__(self, root, script, ops_set, tf_alias, project, st_time, count=3, out_loop=False, call_nodes=[], call_files=[], file_data_cache={}, traced_function=set(), exclude_check_lines=set(), out_exclude_check_lines=set()):
        self.func_defs = []
        self.func_call_in_loop = []
        self.count = count
        self.root = root
        self.script = script
        self.out_loop = out_loop
        self.in_loop = False
        self.in_loop_lineno = -1
        self.in_loop_end_lineno = -1
        self.ops_set = ops_set
        self.tf_alias = tf_alias
        self.call_nodes = call_nodes
        self.call_files = call_files
        self.project = project
        self.file_data_cache = file_data_cache
        self.traced_function = traced_function
        self.exclude_check_lines = exclude_check_lines
        self.out_exclude_check_lines = out_exclude_check_lines
        self.st_time = st_time

    def visit_Call(self, node):

        call_defs = self.script.infer(node.lineno, node.col_offset)

        # Track function definitions for tracing later
        if call_defs and if_expand_func_def(call_defs[0]) and (not self.is_in_exclude_lines(node.lineno, node.end_lineno)):
            self.func_defs.append((call_defs[0], self.in_loop, node))

        # Detect repeated creating computation nodes in in_loop or out_loop
        if self.in_loop and (not self.is_in_exclude_lines(node.lineno, node.end_lineno)) and if_track_func_call(node, self.ops_set, self.tf_alias):
            self.func_call_in_loop.append(
                [node, str(self.script.path), self.call_nodes, self.call_files])
        elif self.out_loop and (not self.is_in_out_exclude_lines(node.lineno, node.end_lineno)) and if_track_func_call(node, self.ops_set, self.tf_alias):
            self.func_call_in_loop.append(
                [node, str(self.script.path), self.call_nodes, self.call_files])

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if self.count < 3 and hasattr(node, "args"):
            for arg in node.args.args:
                for ref in self.script.get_references(arg.lineno, arg.col_offset)[1:]:
                    self.out_exclude_check_lines.add(
                        self.get_line_sig(ref.line))

        back_in_loop = self.in_loop

        # Filter out function defined in the for loop
        if node.lineno > self.in_loop_lineno and node.lineno <= self.in_loop_end_lineno:
            self.in_loop = False
        self.generic_visit(node)
        self.in_loop = back_in_loop

    def visit_For(self, node):

        # Filter for loop with hard coded list/set/tuple
        if isinstance(node.iter, ast.List) or isinstance(node.iter, ast.Set) or isinstance(node.iter, ast.Tuple):
            return
        # Filter for loop with small iterations
        if isinstance(node.iter, ast.Call) and hasattr(node.iter.func, 'id') and node.iter.func.id == 'range' and len(node.iter.args) == 1 and hasattr(node.iter.args[0], 'value') and isinstance(node.iter.args[0].value, int) and node.iter.args[0].value <= 10:
            return

        if isinstance(node.target, ast.AST):
            self.visit(node.target)
        if isinstance(node.iter, ast.AST):
            self.visit(node.iter)

        self.in_loop = True
        self.in_loop_lineno = node.lineno
        self.in_loop_end_lineno = node.end_lineno

        def fn(node):
            for ref in self.script.get_references(node.lineno, node.col_offset)[1:]:
                if ref.line >= self.in_loop_lineno and ref.line <= self.in_loop_end_lineno:
                    self.exclude_check_lines.add(self.get_line_sig(ref.line))
                if is_in_out_exclude:
                    self.out_exclude_check_lines.add(self.get_line_sig(ref.line))

        is_in_out_exclude = True if self.is_in_out_exclude_lines(
            node.lineno, node.end_lineno) else False
        if hasattr(node, 'target'):
            tuple_list_fn(fn, node.target)

        # Extract variables in the loop that should be excluded, and add them into exclude_check_lines
        extractor = AssignVarExtractor(self.script, self.in_loop_lineno)
        for item in node.body:
            if isinstance(item, ast.AST):
                extractor.visit(item)
        for var in extractor.assign_vars:
            for ref in self.script.get_references(var.lineno, var.col_offset)[1:]:
                self.exclude_check_lines.add(self.get_line_sig(ref.line))
        del extractor

        # Start scanning the for loop body to find possible target APIs
        for item in node.body:
            if isinstance(item, ast.AST):
                self.visit(item)
        self.in_loop = False
        self.in_loop_lineno = -1
        self.in_loop_end_lineno = -1
        for item in node.orelse:
            if isinstance(item, ast.AST):
                self.visit(item)

    def visit_If(self, node):
        back_in_loop = self.in_loop
        back_out_loop = self.out_loop

        # filter out if i == 0 in loop
        if hasattr(node.test, 'ops') and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq) and isinstance(node.test.comparators[0], ast.Constant):
            self.in_loop = False
            self.out_loop = False
        self.generic_visit(node)
        self.in_loop = back_in_loop
        self.out_loop = back_out_loop

    def visit_Assign(self, node):

        def fn(node):
            for ref in self.script.get_references(node.lineno, node.col_offset):
                if ref.line >= self.in_loop_lineno and ref.line <= self.in_loop_end_lineno:
                    self.exclude_check_lines.add(self.get_line_sig(ref.line))

        def fn2(node):
            for ref in self.script.get_references(node.lineno, node.col_offset):
                self.out_exclude_check_lines.add(self.get_line_sig(ref.line))

        # Add variables that depend on excluded variables
        if self.is_in_exclude_lines(node.lineno, node.end_lineno):
            for target in node.targets:
                tuple_fn(fn, target)
        if self.is_in_out_exclude_lines(node.lineno, node.end_lineno):
            for target in node.targets:
                tuple_fn(fn2, target)
        self.generic_visit(node)

    def visit_AugAssign(self, node):

        def fn(node):
            for ref in self.script.get_references(node.lineno, node.col_offset):
                if ref.line >= self.in_loop_lineno and ref.line <= self.in_loop_end_lineno:
                    self.exclude_check_lines.add(self.get_line_sig(ref.line))

        def fn2(node):
            for ref in self.script.get_references(node.lineno, node.col_offset):
                self.out_exclude_check_lines.add(self.get_line_sig(ref.line))

        # Add variables that depend on excluded variables
        if self.is_in_exclude_lines(node.lineno, node.end_lineno):
            tuple_fn(fn, node.target)
        if self.is_in_out_exclude_lines(node.lineno, node.end_lineno):
            tuple_fn(fn2, node.target)
        self.generic_visit(node)

    def visit_With(self, node):
        # filter out "with tf.GradientTape() as tape"
        for item in node.items:
            if hasattr(item.context_expr, 'func') and hasattr(item.context_expr.func, 'attr') and item.context_expr.func.attr == 'GradientTape':
                return

        back_in_loop = self.in_loop
        back_out_loop = self.out_loop

        # if "with" uses loop variables, remove loop tag
        if self.is_in_exclude_lines(node.lineno, node.end_lineno):
            self.in_loop = False
        if self.is_in_out_exclude_lines(node.lineno, node.end_lineno):
            self.out_loop = False
        self.generic_visit(node)
        self.in_loop = back_in_loop
        self.out_loop = back_out_loop

    # drop comprehension nodes to avoid false positives
    def visit_ListComp(self, node):
        pass
    def visit_SetComp(self, node):
        pass
    def visit_DictComp(self, node):
        pass
    def visit_GeneratorExp(self, node):
        pass
    def visit_Lambda(self, node):
        pass

    def get_line_sig(self, lineno):
        return '%s/%d' % (str(self.script.path), lineno)

    def is_in_exclude_lines(self, st_lineno, end_lineno):
        for i in range(st_lineno, end_lineno+1):
            if self.get_line_sig(i) in self.exclude_check_lines:
                return True
        return False

    def is_in_out_exclude_lines(self, st_lineno, end_lineno):
        for i in range(st_lineno, end_lineno+1):
            if self.get_line_sig(i) in self.out_exclude_check_lines:
                return True
        return False

    def trace(self):
        if not self.root:
            return
        # process one file should not exceed 5 minutes
        if time.time() - self.st_time > 600:
            return
        if self.count == 3:
            self.generic_visit(self.root)
        else:
            # visit function defs if it is in the recursive procedure
            self.visit_FunctionDef(self.root)
        if self.count > 0:
            for func_def, cur_in_loop, call_node in self.func_defs:
                # not traced function
                if func_def.module_path and str(func_def.module_path) + ' ' + func_def.name not in self.traced_function and (cur_in_loop or self.out_loop):
                    d = func_def.module_path and FuncDef(
                        func_def.module_path, func_def.name, self.project, self.file_data_cache)
                    file_data = get_file_data(
                        func_def.module_path, self.project, self.file_data_cache)
                    func_extractor = ExpandFuncExtractor(d.get_func_def_node(), file_data.script, self.ops_set, self.tf_alias, self.project, self.st_time, self.count-1, cur_in_loop or self.out_loop, [call_node] + self.call_nodes, [self.script.path] + self.call_files, self.file_data_cache, self.traced_function, self.exclude_check_lines, self.out_exclude_check_lines)
                    func_extractor.trace()
                    self.func_defs += func_extractor.func_defs
                    self.func_call_in_loop += func_extractor.func_call_in_loop
                    self.traced_function.add(
                        str(func_def.module_path) + ' ' + func_def.name)

def scan_file(f, project, ops_set, file_data_cache, traced_function, exclude_check_lines, out_exclude_check_lines, cur_found_set):
    print('[Scaning] %s' % str(f))

    file_data = get_file_data(f, project, file_data_cache)
    tree = file_data.tree
    script = file_data.script
    if not tree or not script:
        print("[Error] Failed to read the file %s" % f)
        return
    alias_finder = FindTfImport()
    alias_finder.generic_visit(tree)
    st_time = time.time()

    extractor = ExpandFuncExtractor(tree, script, ops_set, alias_finder.alias, project, st_time, file_data_cache=file_data_cache, traced_function=traced_function, exclude_check_lines=exclude_check_lines, out_exclude_check_lines=out_exclude_check_lines)
    extractor.trace()
    found_calls = []
    if extractor.func_call_in_loop:
        for call, path, call_nodes, call_files in extractor.func_call_in_loop:
            full_prefix, last_prefix = get_func_prefix(call)
            cur_str = str("[Checker1 Find] %s in %s, line %d, column %d\n" % (
                last_prefix + '.' + full_prefix, path, call.lineno, call.col_offset))
            if cur_str not in cur_found_set:
                cur_found_set.add(cur_str)
                for i in range(len(call_nodes)):
                    cur_str += str(((i+1)*2) * ' ' + 'Called in %s, line %d, column %d\n' %
                                   (call_files[i], call_nodes[i].lineno, call_nodes[i].col_offset))
                found_calls.append(cur_str)
    return found_calls


def scan_single_repo(project_path):
    ops_file = "./tf_ops_detection"
    ops_set = set()
    with open(ops_file) as f:
        for line in f.readlines():
            ops_set.add(line.strip())

    project = jedi.Project(path=project_path)
    file_data_cache = {}
    traced_function = set()
    exclude_check_lines = set()
    out_exclude_check_lines = set()
    for f in Path(project_path).glob('./**/*.py'):
        l = scan_file(f, project, ops_set, file_data_cache, traced_function,
                      exclude_check_lines, out_exclude_check_lines, set())
        if l:
            print(l)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path")
    args = parser.parse_args()
    scan_single_repo(args.project_path)
