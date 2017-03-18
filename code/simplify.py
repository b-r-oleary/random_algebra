from sympy import simplify, Symbol, lambdify
from sympy import expand as expand_expr
from .monkeypatch_stats import rv_frozen,\
							scale, offset, multiply, add, inverse

def get_expression(self, seed="a"):
	"""
	get a sympy expression for the random variable algebra
	"""

	args  = self.args

	if any([isinstance(arg, rv_frozen)
		    for arg in args]):

		exprs, var_lists = zip(*[
				 arg.get_expression(seed=seed + str(i))
			 	 if isinstance(arg, rv_frozen) else (arg, [])
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

def simplify(self, expand=True):
	expr, var = self.get_expression()
	if expand:
		expr = expand_expr(expr)
	expr = simplify(expr)
	syms, vals = zip(*var)
	func = lambdify(syms, expr)
	return func(*vals)

setattr(rv_frozen, 'get_expression', get_expression)
setattr(rv_frozen, 'simplify', simplify)