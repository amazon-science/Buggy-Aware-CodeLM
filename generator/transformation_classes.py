# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import ast 
import random 

class BaseNodeTransformer(ast.NodeTransformer):
     def __init__(self, exclude_lineno=[], target_node=None) -> None:
        super().__init__()
        self.flag = False 
        self.exclude_lineno = exclude_lineno
        self.linenumber = None 
        self.target_node = target_node

class BaseNodeVisitor(ast.NodeVisitor):
     def __init__(self, exclude_lineno=[], target_node=None) -> None:
        super().__init__()
        self.flag = False 
        self.exclude_lineno = exclude_lineno
        self.linenumber = None 
        self.target_node = target_node

class IfBlockRemover(BaseNodeTransformer):
    """ 
        Define a customized function for removing one if block 
    """
   
    def __init__(self, exclude_lineno=[], target_node=None) -> None:
        super().__init__(exclude_lineno, target_node)

    def visit_If(self, node):
        if self.flag is False:
            if self.target_node is None and node.lineno not in self.exclude_lineno:
                self.flag = True 
                self.linenumber = node.lineno
                return None  # Return None to remove the target If node
            else:
                if node is self.target_node:
                    self.linenumber = node.lineno
                    self.flag = True 
        return self.generic_visit(node)
    
class WhileBlockRemover(BaseNodeTransformer):
    """
        Define a customized function for removing one while block
    """
    def __init__(self, exclude_lineno=[], target_node=None) -> None:
        super().__init__(exclude_lineno, target_node)

    def visit_While(self, node):
        if self.flag is False:
            if self.target_node is None and node.lineno not in self.exclude_lineno:
                self.flag = True 
                self.linenumber = node.lineno
                return None  # Return None to remove the target If node
            else:
                if node is self.target_node:
                    self.linenumber = node.lineno
                    self.flag = True 
        return self.generic_visit(node)

class ForBlockRemover(BaseNodeTransformer):
    """
        Define a customized function for removing one for block
    """
    def __init__(self, exclude_lineno=[], target_node=None) -> None:
        super().__init__(exclude_lineno, target_node)

    def visit_For(self, node):
        if self.flag is False:
            if self.target_node is None and node.lineno not in self.exclude_lineno:
                self.flag = True 
                self.linenumber = node.lineno
                return None  # Return None to remove the target If node
            else:
                if node is self.target_node:
                    self.flag = True 
        return self.generic_visit(node)



class WhileToIfTransformer(BaseNodeTransformer):
    """
         Define a customized function for transforming one while block to if block
    """
    def __init__(self, exclude_lineno=[], target_node=None) -> None:
        super().__init__(exclude_lineno, target_node)

    def visit_While(self, node):
        if self.flag is False:
            if self.target_node is None and node.lineno not in self.exclude_lineno:
                self.flag = True 
                self.linenumber = node.lineno
                return ast.If(
                    test=node.test,
                    body=node.body,
                    orelse=[]
                )
            else:
                if node is self.target_node:
                    self.flag = True
                    self.linenumber = node.lineno
                    return ast.If(
                        test=node.test,
                        body=node.body,
                        orelse=[]
                    )
        return self.generic_visit(node)


class NumericValueChangeNodeVisitor(BaseNodeVisitor):
    """
        Define a customized function for changing the numerical values
    """
    def __init__(self, exclude_lineno=[], target_node=None) -> None:
        super().__init__(exclude_lineno, target_node)
        self.random_value = random.sample(range(1, 100), 10)

    def visit(self, node):
        # Default behavior when visiting a node (prints the node type)
        node_type = type(node).__name__
        if self.flag is False:
            if node_type == "Constant" and  (isinstance(node.value, int) or isinstance(node.value, float)):
                if self.target_node is None and node.lineno not in self.exclude_lineno:
                    node.value = node.value + random.choice(self.random_value)
                    self.flag = True 
                    self.linenumber = node.lineno
                    return 
                else:
                    if node is self.target_node:
                        self.flag = True 
                        self.linenumber = node.lineno
                        node.value = node.value + random.choice(self.random_value)
                        return
        self.generic_visit(node)

class VariableRenamingNodeVisitor(BaseNodeVisitor):
    """
        # Define a customized function for renaming variables
    """
    def __init__(self, exclude_lineno=[], target_node=None) -> None:
        super().__init__(exclude_lineno, target_node)

    def visit(self, node):
        # Default behavior when visiting a node (prints the node type)
        # This function may throw some errors, in some situations, e.g., a[a[i]] = a[i] + 1, Be Attention. But we can skip such situations.
        node_type = type(node).__name__
        if random.choice([0, 1]) == 1:
            # operate over the left part of an assign function 
            # a = a + 2 = > a1 = a + 2
            # b, c= c, b => c, b = c, b 
            if node_type == "Assign":
                targets = node.targets 
                if self.flag is False:
                    if self.target_node is None and node.lineno not in self.exclude_lineno:
                        if type(targets[0]) is not ast.Tuple: # only one variable in the left
                            if type(targets[0]) is ast.Subscript:
                                node.targets[0].value.id = node.targets[0].value.id + random.choice(["_"] + [str(i) for i in range(1,5)] + [chr(ord('a') + i) for i in range(26)])
                            else:
                                node.targets[0].id = node.targets[0].id + random.choice(["_"] + [str(i) for i in range(1,5)] + [chr(ord('a') + i) for i in range(26)])
                        else:
                            variable_names = [targets[0].elts[i].id if type(targets[0].elts[i]) is not ast.Subscript else  targets[0].elts[i].name.id  for i in range(len(targets[0].elts))]
                            for i in range(len(targets[0].elts)):
                                if type(targets[0].elts[i]) is ast.Subscript:
                                    node.targets[0].elts[i].value.id = random.choice(variable_names)
                                else:
                                    node.targets[0].elts[i].id = random.choice(variable_names)
                        self.flag = True 
                        self.linenumber = node.lineno
                        return 
                    else:
                        if node is self.target_node:
                             if type(targets[0]) is not ast.Tuple: # only one variable in the left
                                 node.targets[0].id = node.targets[0].id + random.choice(["_"] + [str(i) for i in range(1,5)] + [chr(ord('a') + i) for i in range(26)])
                             else:
                                 variable_names = [targets[0].elts[i].id for i in range(len(targets[0].elts))]
                                 for i in range(len(targets[0].elts)):
                                    node.targets[0].elts[i].id = random.choice(variable_names)
                             self.flag = True 
                             self.linenumber = node.lineno
                             return 
        else:
            # operate over the right part of an assign function 
            if node_type == "BinOp" and type(node.left) is ast.Name:
                if self.flag is False:
                    if self.target_node is None and node.lineno not in self.exclude_lineno:
                        node.left.id = random.choice(["_"] + [str(i) for i in range(1,5)] + [chr(ord('a') + i) for i in range(26)])
                        self.flag = True 
                        self.linenumber = node.lineno
                        return 
                    else:
                        if node is self.target_node:
                            node.left.id = random.choice(["_"] + [str(i) for i in range(1,5)] + [chr(ord('a') + i) for i in range(26)])
                            self.flag = True 
                            self.linenumber = node.lineno
                            return 

            elif node_type == "BinOp" and type(node.right) is ast.Name:
                if self.flag is False:
                    if self.target_node is not None and node.lineno not in self.exclude_lineno:
                        node.right.id = random.choice(["_"] + [str(i) for i in range(1,5)] + [chr(ord('a') + i) for i in range(26)])
                        self.flag = True 
                        self.linenumber = node.lineno
                        return 
                    else:
                        if node is self.target_node:
                            node.right.id = random.choice(["_"] + [str(i) for i in range(1,5)] + [chr(ord('a') + i) for i in range(26)])
                            self.flag = True 
                            self.linenumber = node.lineno
                            return
        self.generic_visit(node)


class OperatorChangeNodeVisitor(BaseNodeVisitor):
    """
    Define a customized function for changing the operators
    """
    def __init__(self, exclude_lineno=[], target_node=None) -> None:
        super().__init__(exclude_lineno, target_node)
        self.binoperators = [ast.Add(), ast.Sub(), ast.Div(), ast.Mult(), ast.Mod()]
        self.compareoperators = [ast.NotEq(), ast.Eq(), ast.Lt(), ast.LtE(), ast.Gt(), ast.GtE()]
        self.assignoperators = [ast.Add(), ast.Sub(), ast.Div(), ast.Mult(), ast.Mod()]

    def visit(self, node):
        # Default behavior when visiting a node (prints the node type)
        node_type = type(node).__name__

        if self.flag is False and node_type == "BinOp":
            if self.target_node is None and node.lineno not in self.exclude_lineno:
                self.flag = True 
                self.linenumber = node.lineno
                new_op = random.choice([item for item in self.binoperators if item != node.op])
                node.op = new_op
                return 
            else:
                if node is self.target_node:
                    self.flag = True 
                    self.linenumber = node.lineno
                    new_op = random.choice([item for item in self.binoperators if item != node.op])
                    node.op = new_op
                    return 
        elif self.flag is False and node_type == "AugAssign":
            if self.target_node is None and node.lineno not in self.exclude_lineno:
                self.flag = True 
                self.linenumber = node.lineno
                new_op = random.choice([item for item in self.assignoperators if item != node.op])
                node.op = new_op
                return 
            else:
                if node is self.target_node:
                    self.flag = True 
                    self.linenumber = node.lineno
                    new_op = random.choice([item for item in self.assignoperators if item != node.op])
                    node.op = new_op
                    return 
        elif self.flag is False and node_type == "Compare":
            if self.target_node is None and node.lineno not in self.exclude_lineno:
                self.flag = True 
                self.linenumber = node.lineno
                new_op = random.choice([item for item in self.compareoperators if item != node.ops[0]])
                node.ops[0] = new_op
                return 
            else:
                if node is self.target_node:
                    self.flag = True 
                    self.linenumber = node.lineno
                    new_op = random.choice([item for item in self.compareoperators if item != node.ops[0]])
                    node.ops[0] = new_op
                    return 
        
        self.generic_visit(node)



class ConditionRemovalNodeVisitor(BaseNodeVisitor):
    """
    Define a customized function for delete an else or else if branch
    """
    
    def __init__(self, exclude_lineno=[], target_node=None) -> None:
        super().__init__(exclude_lineno, target_node)

    def visit(self, node):
        # Default behavior when visiting a node (prints the node type)
        node_type = type(node).__name__
        if self.flag is False and (node_type == "If" or node_type == "While"):
            if type(node.test) == ast.BoolOp and node.lineno not in self.exclude_lineno:
                random_index = random.randint(0, len(node.test.values) - 1)
                # Remove the element at the random index
                node.test.values.pop(random_index)
                self.flag = True 
                self.linenumber = node.lineno
                return 
        return self.generic_visit(node)
    
class BranchRemovalNodeVisitor(BaseNodeVisitor):
    """
    Define a customized function for delete an else or else if branch
    """
    
    def __init__(self, exclude_lineno=[], target_node=None) -> None:
        super().__init__(exclude_lineno, target_node)

    def visit(self, node):
        # Default behavior when visiting a node (prints the node type)
        node_type = type(node).__name__
        if self.flag is False and node_type == "If":
            if len(node.orelse) == 1 and node.lineno not in self.exclude_lineno:
                  if  not hasattr(node.orelse[0], "orelse"):
                       node.orelse = []
                  else:
                       node.orelse = node.orelse[0].orelse
                  self.flag = True 
                  self.linenumber = node.lineno
                  return 
        return self.generic_visit(node)
    
class KeywordRemovalTransformer(BaseNodeTransformer):
    """
    Define a customized function for delete keywords, e.g., break, continue and return 
    """
    
    def __init__(self, exclude_lineno=[], target_node=None) -> None:
        super().__init__(exclude_lineno, target_node)

    def visit(self, node):
        # Default behavior when visiting a node (prints the node type)
        node_type = type(node).__name__
        if self.flag is False and node_type in ["Break", "Continue", "Return"]:
            if self.target_node is None and node.lineno not in self.exclude_lineno:
                self.linenumber = node.lineno
                self.flag = True 
                return None
            else:
                if node is self.target_node:
                    self.linenumber = node.lineno
                    self.flag = True 
                    return None
        return self.generic_visit(node)
      

# Define a customized function for find the parent node of a given node. 
class NodeFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.flag = False # only change one time
        self.target_node = None 

    def visit(self, node):
        # Default behavior when visiting a node (prints the node type)
        node_type = type(node).__name__
        if  self.flag is False and node_type == "If" and random.choice([1,2]) == 1: #random.choice([1,2]) == 1 is used to add some randomness
            if len(node.body) == 1 and (type(node.body[0]) is ast.Return() or type(node.body[0]) is ast.Continue() or type(node.body[0]) is ast.Break()):
                 self.flag = True 
                 self.target_node = node 
                 return 
        self.generic_visit(node)