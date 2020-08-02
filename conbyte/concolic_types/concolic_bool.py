# Copyright: see copyright.txt

import logging
from conbyte.expression import Expression
# import conbyte.global_utils

log = logging.getLogger("ct.con.bool")

class ConcolicBool:
    def __init__(self, value: bool, expr_engine: Expression=None):
        assert type(value) is bool
        self.value = value
        self.engine = None
        if expr_engine:
            self.expr = expr_engine.expr
            self.engine = expr_engine.engine
        else:
            self.expr = str(self.value).lower()
        # if isinstance(self.expr, list):
        #     self.expr = conbyte.global_utils.add_extended_vars_and_queries('Bool', self.expr)
        log.debug("  ConBool, value %s, expr: %s" % (self.value, self.expr))

    def __bool__(self):
        if self.engine:
            self.engine.path.which_branch(self)
        return self.value

    def __index__(self):
        from conbyte.concolic_types.concolic_int import ConcolicInt
        value = int.__int__(self.value)
        if self.engine:
            return ConcolicInt(value, Expression(["ite", ["=", self.expr, "true"], 1, 0], self.engine))
        else:
            return ConcolicInt(value)

    # def __str__(self):
    #     return "{ConType, value: %s, expr: %s)" % (self.value, self.expr)

    # custom method to get the primitive value
    def parent(self):
        return self.value

    # def to_formula(self):
    #     expr = self.expr
    #     formula = self._to_formula(expr)
    #     return formula

    # def _to_formula(self, expr):
    #     if isinstance(expr, list):
    #         formula = "( "
    #         for exp in expr:
    #             formula += self._to_formula(exp) + " "
    #         return formula + " )"
    #     else:
    #         if isinstance(expr, int):
    #             if expr < 0:
    #                 ret = "(- %s)" % -expr
    #             else:
    #                 ret = str(expr)
    #             return ret
    #         else:
    #             return str(expr)

    # def get_concrete(self):
    #     return self.value
    
    # def compare_op(self, operator, other):
    #     val_l = self.value
    #     val_r = other.value
    #     if operator == "==":
    #         value = val_l == val_r
    #         expr = ["=", self.expr, other.expr]
    #     elif operator == "!=":
    #         value = val_l != val_r
    #         expr = ['not', ["=", self.expr, other.expr]]
    #     elif operator == ">":
    #         value = val_l > val_r
    #         expr = [operator, self.expr, other.expr]
    #     elif operator == "<":
    #         value = val_l < val_r
    #         expr = [operator, self.expr, other.expr]
    #     elif operator == ">=":
    #         value = val_l >= val_r
    #         expr = [operator, self.expr, other.expr]
    #     elif operator == "<=":
    #         value = val_l <= val_r
    #         expr = [operator, self.expr, other.expr]
    #     else:
    #         return None

    #     return ConcolicBool(value, expr)

    # def __eq__(self, other):
    #     if self.value != other.value:
    #         return False
    #     else:
    #         return self._eq_worker(self.expr, other.expr)

    # # TODO
    # def __or__(self, other):
    #     value = self.value | other.value
    #     expr = ["and", self.expr, other.expr]
    #     return ConcolicBool(value, expr)

    # TODO
    def __xor__(self, other):
        value = self.value ^ other.value
        expr = ["xor", self.expr, other.expr]
        return ConcolicBool(value, Expression(expr, self.engine))

    # TODO
    # def __and__(self, other):
    #     value = self.value & other.value
    #     expr = ["and", self.expr, other.expr]
    #     return ConcolicBool(value, expr)

    # For bool type
    # def negate(self):
    #     raise NotImplementedError
    #     self.value = not self.value
    #     if True: # :
    #         self.expr = ['not', self.expr]
    #     else:
    #         self.expr = self.value
