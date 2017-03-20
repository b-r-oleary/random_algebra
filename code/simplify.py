from sympy import Symbol, lambdify
from sympy import expand as expand_expr
from sympy import simplify as simplify_expr
from .monkeypatch_stats import rv_frozen,\
							scale, offset, multiply, add, inverse

def get_expression(self, seed="a"):
	"""
	get a sympy expression for the random variable algebra

	*inputs*
	seed (str) sets the prefix for variable names for the sympy expressions

	*outputs*
	(sympy expression), (list of tuples of format (sympy symbol, scipy.stats.rv_frozen))
	"""

	args  = self.args

	if any([isinstance(arg, rv_frozen)
		    for arg in args]):

		exprs, var_lists = zip(*[
				 arg.get_expression(seed=seed + str(i))
			 	 if isinstance(arg, rv_frozen) else 
			 	 (arg, [])
			 	 for i, arg in enumerate(args)
			 	 ])

		var_list = []
		for vs in var_lists:
			for k, v in vs:
				var_list.append((k, v))

		name = self.get_name()

		if name in ['scale', 'multiply']:
			return exprs[0] * exprs[1], var_list
		if name in ['offset', 'add']:
			return exprs[0] + exprs[1], var_list
		if name == 'abs':
			return abs(exprs[0]), var_list
		if name == 'inverse':
			return abs(exprs[0]), var_list
		if name in ['power', 'exp']:
			return exprs[0] ** exprs[1], var_list
		if name == 'log':
			return np.log(exprs[0]) / np.log(exprs[1]), var_list

	symbol_name = (seed + "0")
	symbol = Symbol(symbol_name)
	return symbol, [(symbol, self)]

def simplify(self, expand=True, inplace=False):
	"""
	simplifies the graph of operations used to evaluate the
	variable distribution by using an intermediate sympy
	expression

	*inputs*
	expand (bool) whether or not to expand the expression 
		before simplifying.
	inplace (bool) whether or not to set the current object to the 
		simplified object or not.
	"""
	expr, var = self.get_expression()
	if expand:
		expr = expand_expr(expr)
	expr = simplify_expr(expr)
	syms, vals = zip(*var)
	func = lambdify(syms, expr)
	output = func(*vals)
	if inplace:
		self.__dict__.update(output.__dict__)
	else:
		return output

setattr(rv_frozen, 'get_expression', get_expression)
setattr(rv_frozen, 'simplify', simplify)