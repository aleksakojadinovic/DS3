#!/usr/bin/env python
# coding: utf-8

import numpy as np
import copy


def num_and_idx_to_str(pair):
    index, num = pair
    if num == 0:
        return ''
    # takes one pair of form (index, coeff) and returns  coeff*x_index
    if num < 0:
        leading_sign = ' - '
    else:
        if index == 0:
            leading_sign = ''
        else:
            leading_sign = ' + '

    num_str = str(abs(num)) if not np.isclose(abs(num), 1) else ''
    return f'{leading_sign}{num_str}x_{index}'

# Creates a linear expression: num
def constant(num):
    return LinearExpression([], num)

# Creates a linear expression: c*x_idx
def monomial(c, idx):
    if idx < 0:
        raise ValueError(f'Negative index: {idx}')
    return LinearExpression(
        np.hstack(
            (np.zeros(idx), np.array([c]))
        ))
    
# Creates a linear expression: x_idx
def variable(idx):
    return monomial(1.0, idx)
    
# Creates a list of variables: x_0, x_1, ..., x_n
# Useful for:
# x, y, z, u, v = variables(5)
# x + y <= z - u ... etc
def variables(n):
    return [variable(i) for i in range(n)]
    
# A linear express of form:
# c_0x_0 + c_1x_1 + ... + c_nx_n + b
class LinearExpression:
    def __init__(self,
                 coeffs=None,
                 const=0):
        if coeffs is None:
            coeffs = []
        self.coeffs   = np.array(coeffs, dtype='float32')
        self.constant = float(const)
        self.trim()
            
    # Removes trailing zeros
    # Does not touch leading zeros
    # since that would change the expression
    def trim(self):
        self.coeffs = np.trim_zeros(self.coeffs, 'b')
    
    # Length of COEFFS, does not include the constant
    def __len__(self):
        return len(self.coeffs)
    
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        indices  = list(range(len(self)))
        coeffs   = list(self.coeffs)
        together = list(filter(lambda pair: not np.isclose(pair[1], 0.0), zip(indices, coeffs)))
        
        strings  = list(map(num_and_idx_to_str, together))
        
        var_part = ''.join(strings)
        if var_part == '':
            return str(self.constant)
        
        if self.constant == 0.0:
            constant_str = ''
        else:
            constant_sign = '+' if self.constant > 0 else '-'
            constant_abs  = abs(self.constant)
            constant_str  = f'{constant_sign} {str(constant_abs)}'
            
        
        return var_part + ' ' + constant_str
    
    # Returns whether this expression is a constant
    def is_constant(self):
        return np.isclose(self.coeffs, 0.0).all()
    
    # Returns whether this expression is of form
    # c*x_i for some i
    # If which is set to true then it returns the coefficient and the index
    def is_monomial(self, which=False):
        if not np.isclose(self.constant, 0.0):
            if not which:
                return False
            return False, None, None
        zero_markers = np.isclose(self.coeffs, 0.0)
        number_of_zeros = np.sum(zero_markers)
        if not which:
            # If all but one are zeros
            return number_of_zeros == (len(self) - 1)
        
        if number_of_zeros != len(self) - 1:
            return False, None, None
        
        non_zero_idx = np.argwhere(zero_markers == False)[0][0]
        return True, self.coeffs[non_zero_idx], non_zero_idx
        
    # Returns whether this expression is of form
    # x_i for some i
    # If which is true then it returns the index
    def is_variable(self, which=False):
        is_mon, coeff, index = self.is_monomial(which=True)
        if not which:
            return is_mon and np.isclose(coeff, 1)
        
        is_it = is_mon and np.isclose(coeff, 1)
        
        if not is_it:
            return False, None
        return True, index
    
    # Takes either
    #     (1) an integer
    #            - returns the coefficient/monomial at that index
    #     (2) linear expression
    #            - if it is not a variable then it throws
    #            - otherwise returns coefficient/monomial at that variable
    def at(self, key, out='coeff'):
        if isinstance(key, LinearExpression):
            isvar, index = key.is_variable(which=True)
            if not isvar:
                raise ValueError(f'Cannot index an expression with a non-variable expression.')
            coeff = self.coeffs[index] if index <= len(self) - 1 else 0.0
            idx   = index
        else:
            coeff = self.coeffs[key] if key <= len(self) - 1 else 0.0
            idx   = key
        
        if out == 'coeff':
            return coeff
        else:
            return monomial(coeff, idx)
        
    def __getitem__(self, key):
        return self.at(key, out='coeff')
    
    # Returns constant if expression is constant, throws otherwise
    def __float__(self):
        if not self.is_constant():
            raise ValueError(f'Cannot get float value of a non-constant expression')
        return self.constant
        
    
    # =============== ARITHMETIC =======================================================
    
    # Performs a binary operation on self and other
    def do_op(self, oth, binary_op):
        if isinstance(oth, LinearExpression):
            other = copy.deepcopy(oth)
        else:
            # Assume float (or convertible to float) otherwise it will crash
            other = constant(float(oth))
            
        if len(self) < len(other):
            # Other has more variables, pad myself with zeros to the right
            self.coeffs = np.hstack((self.coeffs, np.zeros(len(other) - len(self))))
        elif len(self) > len(other):
            # I have more variables, pad other with zeros to the right
            other.coeffs = np.hstack((other.coeffs, np.zeros(len(self) - len(other))))
        self.coeffs   = binary_op(self.coeffs, other.coeffs)
        self.constant = binary_op(self.constant, other.constant)
        self.trim()
        
    # ---- NEGATION ---
    def __neg__(self):
        new = copy.deepcopy(self)
        new *= -1
        return new

    
    # ---- ADDITION ---- 
    def __iadd__(self, other):
        self.do_op(other, lambda a, b: a + b)
        return self
    
    def __add__(self, other):
        new = copy.deepcopy(self)
        new += other
        return new
    
    def __radd__(self, other):
        return self.__add__(other)
    
    # ---- SUBTRACTION ---- 
    def __isub__(self, other):
        self.do_op(other, lambda a, b: a - b)
        return self
    
    def __sub__(self, other):
        new = copy.deepcopy(self)
        new -= other
        return new
    
    def __rsub__(self, other):
        return (-self).__add__(other)
                
    # ---- Multiplication ---- 
    def __imul__(self, other):
        f = float(other)
        self.coeffs *= f
        self.constant *= f
        return self
    
    def __mul__(self, other):
        new = copy.deepcopy(self)
        new *= other
        return new
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    # ---- Float division ---- 
    def __itruediv__(self, other):
        f = float(other)
        self.coeffs /= f
        self.constant /= f
        return self
    
    def __truediv__(self, other):
        new = copy.deepcopy(self)
        new /= other
        return new

    def create_inequality(self, other, op):
        oth = other
        if not isinstance(other, LinearExpression):
            oth = constant(float(other))

        return LinearInequality(self, oth, op)

    def __gt__(self, other):
        return self.create_inequality(other, GT)

    def __lt__(self, other):
        return self.create_inequality(other, LT)

    def __ge__(self, other):
        return self.create_inequality(other, GTE)

    def __le__(self, other):
        return self.create_inequality(other, LTE)





GT = '>'
LT = '<'
GTE = '>='
LTE = '<='

flipped_op = {
    GT: LT,
    LT: GT,
    GTE: LTE,
    LTE: GTE
}

# A linear inequality of the form:
# le1 <= le2
# where le1 and le2 are linear expressions
class LinearInequality:
    def __init__(self,
                 left=constant(0),
                 right=constant(0),
                 op=GT
                 ):

        self.op = op
        self.assert_op()
        if not isinstance(left, LinearExpression):
            left = constant(float(left))
        else:
            left = copy.deepcopy(left)
        if not isinstance(right, LinearExpression):
            right = constant(float(right))
        else:
            right = copy.deepcopy(right)

        self.left  = left
        self.right = right

    def __str__(self):
        return f'{self.left} {self.op} {self.right}'

    def __repr__(self):
        return str(self)

    def assert_op(self):
        if self.op not in [GT, LT, GTE, LTE]:
            raise ValueError(f'Unknown op: {self.op}')

    def flip_op(self):
        self.op = flipped_op[self.op]

    # ================== ARITHMETIC ===================
    # Negation
    def __neg__(self):
        new = copy.deepcopy(self)
        new.left *= -1
        new.right *= -1
        new.flip_op()
        return new

    # Addition
    def __iadd__(self, other):
        self.left += other
        self.right += other
        return self

    def __add__(self, other):
        new = copy.deepcopy(self)
        new += other
        return new

    def __radd__(self, other):
        return self.__add__(other)

    # Subtraction
    def __isub__(self, other):
        self.left -= other
        self.right -= other
        return self

    def __sub__(self, other):
        new = copy.deepcopy(self)
        new -= other
        return new

    def __rsub__(self, other):
        new = copy.deepcopy(self)
        new.left.__rsub__(other)
        new.right.__rsub__(other)
        self.flip_op()
        return new

    # Multiplication

    def __imul__(self, other):
        f = float(other)
        self.left *= f
        self.right *= f
        if f < 0:
            self.flip_op()
        return self

    def __mul__(self, other):
        new = copy.deepcopy(self)
        new *= other
        return new

    def __rmul__(self, other):
        return self.__mul__(other)






