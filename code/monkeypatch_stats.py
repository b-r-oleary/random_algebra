from .algebra import rv_continuous, rv_frozen,\
					 rv_scale, rv_offset, rv_sum
from scipy.stats import norm

def get_normal_approx(self, *args, **kwargs):
    """
    a method to be appended to scipy.stats distributions
    to quickly extract a normal distribution approximation from
    a potentially complex distribution that may be slow to compute
    """
    return norm(self.mean(*args, **kwargs), 
                self.std( *args, **kwargs))

# append this method to rv_continuous and rv_frozen
setattr(rv_continuous, "get_normal_approx",
    get_normal_approx
)

setattr(rv_frozen, "get_normal_approx",
    get_normal_approx
)

# add the addition and comparison operators
def __add__(self, other):
    if isinstance(other, (rv_frozen, rv_continuous)):
        return rv_sum(self, other)
    elif isinstance(other, (int, float)):
        return rv_offset(self, other)
    else:
        raise NotImplementedError()
    
def __mul__(self, other):
    if isinstance(other, (int, float)):
        return rv_scale(self, other)
    else:
        raise NotImplementedError()
        
def __radd__(self, other):
    return self.__add__(other)

def __rmul__(self, other):
    return self.__mul__(other)

def __sub__(self, other):
    return self + (-1) * other

def __lt__(self, other):
	if isinstance(other, (rv_frozen, rv_continuous, int, float)):
		return (self - other).cdf(0)
	else:
		raise NotImplementedError()

def __le__(self, other):
	return self.__lt__(other)

def __gt__(self, other):
	return 1 - self.__lt__(other)

def __ge__(self, other):
	return self.__gt__(other)



        
objects = [rv_frozen, rv_continuous]
methods = dict(__add__=__add__,
               __mul__=__mul__,
               __radd__=__radd__,
               __rmul__=__rmul__,
               __sub__=__sub__,
               __le__=__le__,
               __lt__=__lt__,
               __gt__=__gt__,
               __ge__=__ge__)
    
for obj in objects:
    for name, method in methods.items():
        setattr(obj, name, method)
