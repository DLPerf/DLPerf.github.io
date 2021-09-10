from __future__ import print_function
import jedi
import ast
from jedi.api.environment import get_default_environment
from pathlib import Path
import time
import argparse
import copy

# Find aliases for tensorflow packages
class FindTfImport(ast.NodeVisitor):
    def __init__(self, package = "tensorflow"):
        self.package = package
        self.alias = [package]
    def visit_Import(self, node):
        for name in node.names:   
            if name.name == self.package:
                self.alias.append(name.asname)
    def visit_ImportFrom(self, node):
        pass

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

def get_file_data(path, project):
    return FileData(path, project)

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

# Extract chain calls of dataset APIs
class ChainCallExtractor(ast.NodeVisitor):
    def __init__(self):
        self.chain_calls = []
    def visit_Call(self, node):
        self.chain_calls.append(get_func_prefix(node))
        if not hasattr(node.func, 'value') or not isinstance(node.func.value, ast.Call):
            self.chain_calls.reverse()
        else:
            self.generic_visit(node)

# Extract dataset definitions
class DefinedDataSetExtractor(ast.NodeVisitor):
    def __init__(self, p_extractor):
            self.p_extractor = p_extractor
    def visit_Call(self, node):
        extractor = ChainCallExtractor()
        extractor.visit_Call(node)
        chain_calls = extractor.chain_calls
        name, prefix = chain_calls[0]
        
        # Add new defined_dataset_node
        cur_line_sig = self.p_extractor.get_line_sig(node.lineno)
        if self.p_extractor.is_dataset_call(name, prefix):
            self.p_extractor.defined_usage_lines_map[cur_line_sig] = set()
            self.p_extractor.defined_usage_calls_map[cur_line_sig] = []
      
# Apply fn to basic ast nodes except for Tuple
def tuple_fn(fn, node):
    if isinstance(node, ast.Tuple):
        for name in node.elts:
            tuple_fn(fn, name)
    else:
        fn(node)

def reverse_chain_calls(defined_usage_calls_map):
    for key, calls in defined_usage_calls_map.items():
        cur_map = {}
        lines_order = []
        for op, line in calls:
            if line not in lines_order:
                lines_order.append(line)
                cur_map[line] = []
            cur_map[line].append((op, line))
        res_calls = []
        for line in lines_order:
            cur_map[line].reverse()
            res_calls += cur_map[line]
        defined_usage_calls_map[key] = res_calls

class DataChainExtractor(ast.NodeVisitor):
  
    def __init__(self, root, script, alias, project, st_time, detect_parel_ops, ops_arg_limit):

        self.root = root
        self.script = script
        self.tf_alias = alias
        self.project = project
        self.st_time = st_time
        self.detec_paral_ops = detect_parel_ops
        self.ops_arg_limit = ops_arg_limit # the number args of functions to detect num_parallel_calls 

        self.defined_usage_calls_map = {} # {defined_dataset_line: [ref_calls]}
        self.defined_usage_lines_map = {} # {defined_dataset_line: set(refe_lines)}
        self.ref_line_defined_data_line_map = {} # {ref_line: defined_dataset_line}
        self.ops_without_parallel = []
    
    def visit_Assign(self, node):
        
        def fn(node):
            for ref in self.script.get_references(node.lineno, node.col_offset):
                self.defined_usage_lines_map[cur_line_sig].add(self.get_line_sig(ref.line))
                self.ref_line_defined_data_line_map[self.get_line_sig(ref.line)] = cur_line_sig
        
        def fn2(node):
             for ref in self.script.get_references(node.lineno, node.col_offset):
                self.defined_usage_lines_map[parent_line_sig].add(self.get_line_sig(ref.line))
                self.ref_line_defined_data_line_map[self.get_line_sig(ref.line)] = parent_line_sig


        cur_line_sig =  self.is_in_dataset_lines(node.lineno, node.end_lineno)
        if cur_line_sig:
            for target in node.targets:
                tuple_fn(fn, target)
        else:
            cur_line_sig =  self.is_in_defined_lines(node.lineno, node.end_lineno)
            if cur_line_sig:
                parent_line_sig = self.ref_line_defined_data_line_map[cur_line_sig]
                for target in node.targets:
                    tuple_fn(fn2, target)
          
        self.generic_visit(node)
    

    def visit_Call(self, node):

        def map_paral_detect():
            if hasattr(node.func, "attr") and node.func.attr in self.detec_paral_ops:
                num_arg_limit = self.ops_arg_limit[self.detec_paral_ops.index(node.func.attr)]
                is_paralel = False
                for arg in node.args:
                    if hasattr(arg, 'id') and arg.id == 'num_parallel_calls':
                        is_paralel = True
                        break
                for keyword in node.keywords:
                    if hasattr(keyword, 'arg') and keyword.arg == 'num_parallel_calls':
                        is_paralel = True
                        break
                if not is_paralel and len(node.args) < num_arg_limit:
                    self.ops_without_parallel.append("[Find] (%s) in %s should be called with num_parallel_calls" % (node.func.attr, self.get_line_sig(node.lineno)))

        cur_line_sig = self.is_in_defined_lines(node.lineno, node.end_lineno)
        if cur_line_sig:
            defined_dataset_sig = self.ref_line_defined_data_line_map[cur_line_sig]
            self.defined_usage_calls_map[defined_dataset_sig].append((get_func_prefix(node), self.get_line_sig(node.lineno)))
            map_paral_detect()
        else:
            cur_line_sig = self.is_in_dataset_lines(node.lineno, node.end_lineno)
            if cur_line_sig:
                self.defined_usage_calls_map[cur_line_sig].append((get_func_prefix(node), self.get_line_sig(node.lineno)))
                self.ref_line_defined_data_line_map[cur_line_sig] = cur_line_sig
                map_paral_detect()

        self.generic_visit(node)

    def get_line_sig(self, lineno):
        return '%s/%d' % (str(self.script.path), lineno)
    
    def is_in_defined_lines(self, st_lineno, end_lineno):
        for i in range(st_lineno, end_lineno+1):
            if self.get_line_sig(i) in self.ref_line_defined_data_line_map:
                return self.get_line_sig(i)
        return None

    def is_in_dataset_lines(self, st_lineno, end_lineno):
        for i in range(st_lineno, end_lineno+1):
            if self.get_line_sig(i) in self.defined_usage_lines_map:
                return self.get_line_sig(i)
        return None

    # Decide whether this function call is tf.data.**Dataset
    def is_dataset_call(self, name, prefix):
        return prefix in self.tf_alias and "Dataset" in name
    
    def trace(self):
        if not self.root:
            return
        # process one file should not exceed 5 minutes
        if time.time() - self.st_time > 600:
            return
        extractor = DefinedDataSetExtractor(self)
        extractor.generic_visit(self.root)

        self.generic_visit(self.root)
        reverse_chain_calls(self.defined_usage_calls_map)

def detect_data_calls_violations(defined_usage_calls_map, ops_list):
    violate_calls = []
    for _, calls in defined_usage_calls_map.items():
        in_ops_calls = []
        for call, call_line in calls:
            if call[0] in ops_list:
                in_ops_calls.append((ops_list.index(call[0]), call[0], call_line))
    
        for i in range(len(in_ops_calls)):
            for j in range(i+1, len(in_ops_calls)):
                index1, call1, line1 =  in_ops_calls[i]
                index2, call2, line2 =  in_ops_calls[j]
                if index1 > index2:
                    violate_call_str = '[Find] (%s) in %s should be called before (%s) in %s' % (call2, line2, call1, line1)
                    violate_calls.append(violate_call_str)

    return violate_calls


def scan_file(f, project):
    print('[Scaning] %s' % str(f))

    file_data = get_file_data(f, project)
    tree = file_data.tree
    script = file_data.script

    if not tree or not script:
        print("[Error] Failed to read the file %s" % f)
        return
    alias_finder = FindTfImport()
    alias_finder.generic_visit(tree)
    st_time = time.time()

    extractor = DataChainExtractor(tree, script, alias_finder.alias, project, st_time,['map', 'interleave', 'map_with_legacy_function'], [2,4,2])
    extractor.trace()
    
    order_found_calls = detect_data_calls_violations(extractor.defined_usage_calls_map, ['batch','map'])
    paral_found_calls = copy.copy(extractor.ops_without_parallel)

    del extractor
    return (order_found_calls, paral_found_calls)


def scan_single_repo(project_path):
    project = jedi.Project(path=project_path)
    for f in Path(project_path).glob('./**/*.py'):
        order_found_calls, paral_found_calls = scan_file(f, project)
        if order_found_calls:
            print("[Checker2 Find]: %s" % str(order_found_calls))
        if paral_found_calls:
            print("[Checker3 Find]: %s" % str(paral_found_calls))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path")
    args = parser.parse_args()
    scan_single_repo(args.project_path)
