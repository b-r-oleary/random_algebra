

def infinite_sum(function, lower, epsrel=1.49e-04, min_n=10):
    "this is a very simple method used to evaluate infinite sums"
    total = 0
    i = lower
    previous = 0
    eps = 2 * epsrel
    
    while eps > epsrel or i < min_n:
        new = function(i)
        total += new
        i += 1
        
        if new != 0:
            new = abs(new)
            eps = abs(new - previous)
            
    return total