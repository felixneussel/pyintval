# This module contains the classes Interval and Intervalvector.
# In the class Interval, the magic methods for standard arithmetic operations
# have been overwritten. Arithmetic operations on intervals can thus be coded
# in the same way as for numeric types.
# In the class IntervalVector, the __getitem__() magic method is overwritten,
# enabling us to treat IntervalVectors like numpy arrays.
import numpy as np
    
class Interval:
    def __init__(self,lb,ub):
        if lb > ub:
            raise ValueError('Lower interval bound must be smaller than or equal to the upper bound')
        self.lb = lb
        self.ub = ub
        self.box_mean = 0.5*lb + 0.5*ub
        self.width = ub - lb

    def __str__(self):
        return f'[{self.lb} , {self.ub}]'
    
    def __repr__(self):
        return self.__str__()

    def __add__(self,other):
        if isinstance(other,int) or isinstance(other,float):
            other = Interval(other,other)
        if isinstance(other,Interval):
            return Interval(self.lb + other.lb, self.ub + other.ub)
        return NotImplemented
    
    def __neg__(self):
        return Interval(-self.ub,-self.lb)
    
    def __sub__(self,other):
        if isinstance(other,int) or isinstance(other,float):
            other = Interval(other,other)
        return self.__add__(-other)
    
    def __rsub__(self,other):
            return Interval(other - self.ub,other-self.lb)
    
    def __mul__(self,other):
        if isinstance(other,Interval):
            interval_hull = [self.lb * other.lb,self.lb*other.ub,self.ub*other.lb,self.ub*other.ub]
            return Interval(min(interval_hull),max(interval_hull))
        elif isinstance(other,int) or isinstance(other,float):
            if other >= 0:
                return Interval(other*self.lb,other*self.ub)
            else:
                return Interval(other*self.ub,other*self.lb)
        return NotImplemented
            
    def __truediv__(self,other):
        if isinstance(other,int) or isinstance(other,float):
            return self.__mul__(1/other)
        if other.lb <= 0 and other.ub >= 0:
            raise ValueError('Division with interval containing 0')
        return self.__mul__(Interval(1/other.ub,1/other.lb))
    
    def __rtruediv__(self,other):
        if isinstance(other,int) or isinstance(other,float):
            return Interval(other,other) / self

    
    def __eq__(self,other):
        if isinstance(other,float) or isinstance(other,int):
            return self.lb == other and self.ub == other
        else:
            return self.lb == other.lb and self.ub == other.ub
        
    def almost_equal(self,other,tol=1e-12):
        if isinstance(other,Interval):
            return np.allclose([self.lb,self.ub],[other.lb,other.ub],atol=tol)
    
    def __ge__(self,other):
        if isinstance(other,float) or isinstance(other,int):
            return self.lb >= other
        return self.lb >= other.ub
    
    def __le__(self,other):
        return self.ub <= other
    
    def __gt__(self,other):
        if isinstance(other,float) or isinstance(other,int):
            return self.lb > other
        return self.lb > other.ub
    
    def __lt__(self,other):
        return self.ub < other
    
    def copy(self):
        return Interval(self.lb,self.ub)
    
    def __pow__(self,other):
        if isinstance(other,int):
            if other >= 0:
                if other%2==1 or (other%2==0 and (self > 0 or self < 0)):
                    return Interval(min(self.lb**other,self.ub**other),max(self.lb**other,self.ub**other))
                else:
                    return Interval(0,max(self.lb**other,self.ub**other))
            else:
                return 1/self**(-other)
        return NotImplemented
        
    def sqrt(self):
        if not self >= 0:
            raise ValueError('Interval must be nonnegative for sqrt.')
        return Interval(np.sqrt(self.lb),np.sqrt(self.ub))
    
    def contains(self,other):
        return other <= self.ub and other >= self.lb
    
    def abs(self):
        if self.contains(0):
            return Interval(0,max(abs(self.ub),abs(self.lb)))
        else:
            return Interval(min(abs(self.lb),abs(self.ub)),max(abs(self.lb),abs(self.ub)))
        
    def log(self):
        return Interval(np.log(self.lb),np.log(self.ub))
    
    def exp(self):
        return Interval(np.exp(self.lb),np.exp(self.ub))
    
    def log(self):
        assert self > 0
        return Interval(np.log(self.lb),np.log(self.ub))
        
    __rmul__ = __mul__
    __radd__ = __add__


    def bisect(self):
        return Interval(self.lb, self.box_mean), Interval(self.box_mean, self.ub)
    
    def __and__(self, other):
        if isinstance(other,Interval):
            return Interval(max(self.lb,other.lb),min(self.ub,other.ub))
        return NotImplemented
    
    def sin(self):
        if self.width >= 2*np.pi:
            return Interval(-1,1)
        shift = 2*np.pi*np.floor(self.lb / (2*np.pi))
        shift = Interval(self.lb - shift, self.ub - shift)
        # shift is subset of [0,4pi]
        if shift.contains(3*np.pi/2) or shift.contains(7*np.pi/2):
            lb = -1
        else:
            lb = min(np.sin(self.lb),np.sin(self.ub))
        if shift.contains(np.pi/2) or shift.contains(5*np.pi/2):
            ub = 1
        else:
            ub = max(np.sin(self.lb),np.sin(self.ub))
        return Interval(lb,ub)
    
    def cos(self):
        return (self + np.pi/2).sin()




class IntervalVector:
    __array_ufunc__ = None
    def __init__(self,box):
        if isinstance(box[0],Interval):
            self.intervals = [i.copy() for i in box]
        else:
            self.intervals = [Interval(row[0],row[1]).copy() for row in box]
        self.box_mean = np.array([i.box_mean for i in self.intervals])
        self.len = len(self.intervals)
        self.lb = np.array([i.lb for i in self.intervals])
        self.ub = np.array([i.ub for i in self.intervals])
        self.width = np.linalg.norm(self.ub - self.lb)

    def __str__(self):
        output = '(' + self.intervals[0].__str__() + '\n'
        for i in self.intervals[1:]:
            output += f' {i.__str__()}\n'
        return output[:-1] + ')'
    
    def __repr__(self):
        return self.__str__()
    
    def __getitem__(self,key):
        return self.intervals[key]
    
    def __setitem__(self, key, value):
        assert isinstance(value,Interval)
        self.intervals[key] = value
    
    def __eq__(self, other):
        return all(i == j for i,j in zip(self,other))
    
    def __add__(self,other):
        if isinstance(other,IntervalVector):
            assert self.len == other.len
            return IntervalVector([a+b for a,b in zip(self,other)])
        if isinstance(other,np.ndarray):
            assert len(other.shape) == 1 and other.shape[0] == self.len
            return IntervalVector([a+b for a,b in zip(self,other)])
        if isinstance(other,Interval) or isinstance(other,float) or isinstance(other,int):
            return IntervalVector([a + other for a in self])
        
    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other,IntervalVector):
            assert self.len == other.len
            return IntervalVector([a-b for a,b in zip(self,other)])
        if isinstance(other,np.ndarray):
            assert len(other.shape) == 1 and other.shape[0] == self.len
            return IntervalVector([a-b for a,b in zip(self,other)])
        if isinstance(other,Interval) or isinstance(other,float) or isinstance(other,int):
            return IntervalVector([a - other for a in self])
        
    def __rsub__(self, left):
        return -self + left
    
    def __neg__(self):
        return IntervalVector([-i for i in self])
        
    
    def __mul__(self,other):
        if isinstance(other,Interval) or isinstance(other,float) or isinstance(other,int):
            return IntervalVector([other * i for i in self])
        else:
            return IntervalVector([i1*i2 for i1,i2 in zip(self,other)])
        
    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other,IntervalVector):
            assert self.len == other.len
            return IntervalVector([a/b for a,b in zip(self,other)])
        if isinstance(other,np.ndarray):
            assert len(other.shape) == 1 and other.shape[0] == self.len
            return IntervalVector([a/b for a,b in zip(self,other)])
        if isinstance(other,Interval) or isinstance(other,float) or isinstance(other,int):
            return IntervalVector([a/ other for a in self])
        return NotImplemented
        
    def __rtruediv__(self, other):
        if isinstance(other,Interval) or isinstance(other,float) or isinstance(other,int):
            return IntervalVector([other/a for a in self])
        return NotImplemented

    def dot(self,other):
        if isinstance(other,IntervalVector):
            assert other.len == self.len
            return sum(i1*i2 for i1,i2 in zip(self,other))
        if isinstance(other,np.ndarray):
            assert other.shape[0] == self.len
            return sum(i1*i2 for i1,i2 in zip(self,other)) 
        return NotImplemented
        
    def outer(self,other=None):
        if other is None:
            res = self.outer(self)
            for i in range(self.len):
                res[i,i] = self[i]**2
            return res
        if isinstance(other,IntervalVector):
            assert other.len == self.len
            return IntervalMatrix([[i1*i2 for i1 in other] for i2 in self])
        return NotImplemented
    
    
    def __rmatmul__(self, left):
        if isinstance(left,np.ndarray):
            assert left.shape[1] == self.len
            return IntervalVector([self.dot(r) for r in left])

    def copy(self):
        return IntervalVector([i.copy() for i in self.intervals])
        
    def bisect(self):
        longest_axis = max(range(self.len),key=lambda i: self[i].width)#max(self,key=lambda i: i.width)
        a, b = self.copy(), self.copy()
        lower, upper = self[longest_axis].bisect()
        a[longest_axis] = lower
        b[longest_axis] = upper
        return a,b

    def norm(self):
        return sum([i**2 for i in self]).sqrt()
    
    def exp(self):
        return IntervalVector([i.exp() for i in self])
    
    def log(self):
        return IntervalVector([i.log() for i in self])
    
    def __and__(self, other):
        if isinstance(other,IntervalVector):
            return IntervalVector([a & b for a,b in zip(self,other)])
        if isinstance(other,Interval):
            return IntervalVector([a & other for a in self])
        return NotImplemented
    
    __rand__ = __and__

    def __pow__(self, other):
        if isinstance(other,int) and other > 0:
            return IntervalVector([i**other for i in self])
    
class IntervalMatrix():
    __array_ufunc__ = None
    def __init__(self,value):
        if isinstance(value[0],IntervalVector):
            self.rows = [v.copy() for v in value]
        else:
            self.rows = [IntervalVector(v) for v in value]
        self.nrows = len(self.rows)
        self.ncols = self.rows[0].len

    def __getitem__(self, key):
        if isinstance(key,int):
            return self.rows[key].copy()
        if isinstance(key,tuple):
            i,j = key
            if isinstance(j,slice) and isinstance(i,int):
                return self.rows[i].copy()
            if isinstance(i,slice) and isinstance(j,int):
                return self.T.rows[j].copy()
            if isinstance(i,int) and isinstance(j,int):
                return self.rows[i][j]
        return NotImplemented
    
    def __setitem__(self, key, value):
        if isinstance(key,tuple) and isinstance(value,Interval):
            assert len(key) == 2
            i,j = key
            assert isinstance(i,int) and isinstance(j,int)
            self.rows[i][j] = value.copy()
        else:
            raise NotImplementedError

    def __eq__(self, value):
        return all(r1 == r2 for r1,r2 in zip(self,value))
    
    def __repr__(self):
        return '\n'.join(['; '.join(i.__repr__() for i in r) for r in self])
    
    def __str__(self):
        return self.__repr__()
    
    @property
    def T(self):
        return IntervalMatrix(np.array([[i for i in r] for r in self.rows]).T)
    
    def __add__(self,other):
        if isinstance(other,Interval) or isinstance(other,float) or isinstance(other,int):
            return IntervalMatrix([[other + i for i in r] for r in self])
        if isinstance(other,IntervalMatrix):
            assert self.nrows == other.nrows and self.ncols == other.ncols
            return IntervalMatrix([r1+r2 for r1,r2 in zip(self,other)])
        if isinstance(other,np.ndarray):
            assert (self.nrows,self.ncols) == other.shape
            return IntervalMatrix([r1+r2 for r1,r2 in zip(self,other)])
        return NotImplemented
    
    __radd__ = __add__

    def __neg__(self):
        return IntervalMatrix([-r for r in self])
    
    def __sub__(self,other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __mul__(self,other):
        if isinstance(other,Interval) or isinstance(other,float) or isinstance(other,int):
            return IntervalMatrix([[other * i for i in r] for r in self])
        return NotImplemented
    
    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * (1/other)

    
    def __matmul__(self, other):
        if isinstance(other,IntervalMatrix):
            assert self.ncols == other.nrows
            return IntervalMatrix([[r.dot(c) for c in other.T] for r in self])
        if isinstance(other,np.ndarray):
            assert self.ncols == other.shape[0]
            return IntervalMatrix([[r.dot(c) for c in other.T] for r in self])
        if isinstance(other,IntervalVector):
            assert other.len == self.ncols
            return IntervalVector([r.dot(other) for r in self])
        return NotImplemented
    
    def __rmatmul__(self, left):
        if isinstance(left,np.ndarray):
            assert self.nrows == left.shape[1]
            return IntervalMatrix([[c.dot(r)  for c in self.T] for r in left])
        return NotImplemented
    
    def alpha(self):
        assert self.nrows == self.ncols
        betas = [self[i,i].lb - sum(max(abs(self[i,j].lb),abs(self[i,j].ub)) for j in range(self.ncols) if j != i) for i in range(self.nrows)]
        return max(0,-min(betas))
    
    
# Math function for numeric types, numpy arrays and Interval types

def exp(x):
    if isinstance(x, Interval) or isinstance(x,IntervalVector):
        return x.exp()
    return np.exp(x)

def log(x):
    if isinstance(x, Interval) or isinstance(x,IntervalVector):
        return x.log()
    return np.log(x)

def sin(x):
    if isinstance(x, Interval) or isinstance(x,IntervalVector):
        return x.sin()
    return np.sin(x)

def cos(x):
    if isinstance(x, Interval) or isinstance(x,IntervalVector):
        return x.cos()
    return np.cos(x)
