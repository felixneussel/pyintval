from interval_arithmetic import *
import pytest

# Tests for real intervals

def test_eq():
    assert Interval(0,1) == Interval(0,1)
    assert Interval(0.1,1) != Interval(0,1)
    assert -3.5 == Interval(-3.5,-3.5)


def test_bisect():
    a,b = Interval(-2,1).bisect()
    assert a == Interval(-2,-0.5) and b == Interval(-0.5,1)

def test_copy():
    a = Interval(0,1)
    b = a.copy()
    assert a == b
    a.lb = 0.1
    assert b == Interval(0,1)

def test_truediv():
    assert Interval(1,2) / 2 == Interval(0.5,1)
    assert Interval(1,2) / -2 == Interval(-1,-0.5)
    assert Interval(1,2) / Interval(1,2) == Interval(0.5,2)
    assert Interval(1,2) / Interval(-2,-1) == Interval(-2,-0.5)

def test_rtruediv():
    assert 2 / Interval(1,2) == Interval(1,2)
    assert -2 / Interval(1,2) == Interval(-2,-1)

def test_and():
    a = Interval(0,1)
    b = Interval(0.5,2)
    c = Interval(0.25,0.75)
    d = Interval(1.1,2)
    assert a & b == Interval(0.5,1)
    assert a & c == c
    with pytest.raises(ValueError):
        a & d

def test_sin():
    assert Interval(3*np.pi,5*np.pi).sin() == Interval(-1,1)
    assert Interval(5.75*np.pi,6.75*np.pi).sin() == Interval(np.sin(5.75*np.pi),1)
    assert Interval(-9*np.pi,-8.25*np.pi).sin().almost_equal(Interval(-1,0))
    assert Interval(-0.25*np.pi,0.25*np.pi).sin().almost_equal(Interval(np.sin(-0.25*np.pi),np.sin(0.25*np.pi)))
    assert Interval(-1.25*np.pi,-0.75*np.pi).sin().almost_equal(Interval(np.sin(-0.75*np.pi),np.sin(-1.25*np.pi)))
    assert Interval(np.pi,2.5*np.pi).sin().almost_equal(Interval(-1,1))

def test_pow():
    assert Interval(1,2)**(-2) == Interval(0.25,1)
    assert Interval(-3,-1)**(-1) == Interval(-1,-1/3)

# Tests for interval vectors

def test_init_v():
    a = IntervalVector([(0,1),(0,1)])
    assert all(a.lb == np.array([0,0]))
    assert all(a.ub == np.array([1,1]))
    assert np.allclose(a.width,np.sqrt(2))
    
    a,b = IntervalVector([(0,1),(0,1)])
    assert a == Interval(0,1) and b == Interval(0,1)

    v1,v2 = Interval(0,1),Interval(0,1)
    v = IntervalVector([v1,v2])
    w = v.copy()
    v1.lb = -2
    assert v == w


def test_eq_v():
    assert IntervalVector([(0,1),(0,2)]) == IntervalVector([(0,1),(0,2)])

def test_copy_v():
    a = IntervalVector([(0,1),(0,2)])
    b = a.copy()
    assert a == b
    a[0] = Interval(0.1,1)
    assert b == IntervalVector([(0,1),(0,2)])

def test_bisect_v():
    a,b = IntervalVector([(0,1),(0,2)]).bisect()
    assert a == IntervalVector([(0,1),(0,1)])
    assert b == IntervalVector([(0,1),(1,2)])

    a,b = IntervalVector([(-1,1),(0,2),(-0.5,3.5)]).bisect()
    assert a == IntervalVector([(-1,1),(0,2),(-0.5,1.5)])
    assert b == IntervalVector([(-1,1),(0,2),(1.5,3.5)])

def test_mul_v():
    a = IntervalVector([(0,1),(0,1)])
    assert a * (-2.5) == IntervalVector([(-2.5,0),(-2.5,0)])
    b = IntervalVector([(-2,1),(2,3)])
    assert a * b == IntervalVector([(-2,1),(0,3)])

    assert -2.5 * a == IntervalVector([(-2.5,0),(-2.5,0)])

    i = Interval(-4,4)
    assert i*a == IntervalVector([(-4,4),(-4,4)])
    assert a*i == IntervalVector([(-4,4),(-4,4)])

def test_dot():
    a = IntervalVector([(0,1),(0,1)])
    b = IntervalVector([(1,2),(2,3)])
    assert a.dot(b) == Interval(0,5)
    assert a.dot(-2*np.ones(2)) == Interval(-4,0)

def test_outer():
    a = IntervalVector([(0,1),(0,1)])
    b = IntervalVector([(1,2),(2,3)])
    res = a.outer(b)
    assert res == IntervalMatrix([
        [(0,2),(0,3)],
        [(0,2),(0,3)]
    ])
    assert IntervalVector([(-1,1),(1,2)]).outer() == IntervalMatrix([
        [(0,1),(-2,2)],
        [(-2,2),(1,4)]
    ])

def test_exp():
    assert exp(Interval(0,1)) == Interval(1,exp(1))
    assert exp(-2) == np.exp(-2)
    assert all(exp(np.array([-1,2])) == np.exp([-1,2]))
    assert all(exp([-1,0,2]) == np.exp([-1,0,2]))
    assert exp(IntervalVector([(0,1),(0,1)])) == IntervalVector([(1,np.e),(1,np.e)])

def test_log():
    assert log(Interval(1,np.e)) == Interval(0,1)
    assert log(2) == np.log(2)
    assert all(log(np.array([1,2])) == np.log([1,2]))
    assert all(log([1,2,3]) == np.log([1,2,3]))
    assert log(IntervalVector([(1,np.e),(1,np.e)])) == IntervalVector([(0,1),(0,1)])

def test_add_v():
    a = IntervalVector([(-1.5,1),(0,2)])
    res = IntervalVector([(0.5,3),(2,4)])
    assert a + a == IntervalVector([(-3,2),(0,4)])
    assert a + 2 == res
    assert 2.0 + a == res
    assert 2 * np.ones(2) + a == res
    assert a + 2*np.ones(2) == res

def test_sub_v():
    a = IntervalVector([(-1.5,1),(0,2)])
    res = IntervalVector([(-3.5,-1),(-2,0)])
    assert a - a == IntervalVector([(-2.5,2.5),(-2,2)])
    assert a - 2 == res
    assert a - 2*np.ones(2) == res
    res2 = IntervalVector([(1,3.5),(0,2)])
    assert 2.0 - a == res2
    assert 2 * np.ones(2) - a == res2

def test_rmatmul_v():
    x = IntervalVector([(-1,1),(2,3.5)])
    assert np.eye(2)@x == x

def test_and_v():
    v1 = IntervalVector([(0,1),(0,1)])
    v2 = IntervalVector([(-1,0.5),(1,2)])
    i = Interval(0.25,0.5)
    assert v1 & v2 == IntervalVector([(0,0.5),(1,1)])
    assert v1 & i == IntervalVector([(0.25,0.5),(0.25,0.5)])
    assert i & v1 == IntervalVector([(0.25,0.5),(0.25,0.5)])

# Tests for interval matrices

def test_init_m():
    A = IntervalMatrix([
        [(0,1),(0,1)],
        [(1,2),(1,2)],
        [(2,3),(2,3)]
    ])
    i1 = Interval(0,1)
    i2 = Interval(1,2)
    i3 = Interval(2,3)

    B = IntervalMatrix([
        [i1,i1],[i2,i2],[i3,i3]
    ])

    r1 = IntervalVector([i1,i1])
    r2 = IntervalVector([i2,i2])
    r3 = IntervalVector([i3,i3])

    C = IntervalMatrix([r1,r2,r3])

    assert A == B
    assert A == C
    assert B == C

    i1.lb = -1
    r2[0] = Interval(4,5)

    assert A == B
    assert A == C
    assert B == C

def test_transpose():
    A = IntervalMatrix([
        [(0,1),(0,1)],
        [(1,2),(1,2)],
        [(2,3),(2,3)]
    ])

    AT = IntervalMatrix([
        [(0,1),(1,2),(2,3)],
        [(0,1),(1,2),(2,3)]
    ])
    assert A.T == AT

def test_matmul_m():
    A = IntervalMatrix([
        [(0,1),(0,1)],
        [(1,2),(1,2)],
        [(2,3),(2,3)]
    ])
    B = IntervalMatrix([
        [(0,1),(0,1)],
        [(0,1),(0,1)]
    ])
    assert A@B == IntervalMatrix([
        [(0,2),(0,2)],
        [(0,4),(0,4)],
        [(0,6),(0,6)]
    ])
    assert A@np.eye(2) == A
    assert np.eye(3)@A == A
    assert B@IntervalVector([(0,1),(0,1)]) == IntervalVector([(0,2),(0,2)])

def test_getitem_m():
    A = IntervalMatrix([
        [(0,1),(0,1)],
        [(1,2),(1,2)],
        [(2,3),(2,3)]
    ])
    assert A[0] == IntervalVector([(0,1),(0,1)])
    assert A[:,1] == IntervalVector([(0,1),(1,2),(2,3)])
    assert A[0,:] == IntervalVector([(0,1),(0,1)])
    assert A[2,1] == Interval(2,3)

def test_alpha():
    X = IntervalVector([(-1,1),(0,1)])
    D2 = IntervalMatrix([
        [Interval(0,0),X[1].exp()],
        [X[1].exp(), X[0]*X[1].exp()]
    ])
    alpha = D2.alpha()
    assert np.allclose(alpha,2*np.e)

def test_add_m():
    A = IntervalMatrix([
        [(0,1),(0,1)],
        [(1,2),(1,2)]
    ])
    B = IntervalMatrix([
        [(0,0),(0,0)],
        [(0,0),(0,0)]
    ])
    assert A + B == A
    assert A + 0 == A
    assert A + Interval(0,0) == A
    assert A + np.zeros((2,2)) == A
    assert 0+ A  == A
    assert Interval(0,0) + A == A
    assert np.zeros((2,2)) + A == A

def test_mul_m():
    A = IntervalMatrix([
        [(0,1),(0,1)],
        [(1,2),(1,2)],
        [(2,3),(2,3)]
    ])
    i = Interval(-1,2)
    res = IntervalMatrix([
        [(-1,2),(-1,2)],
        [(-2,4),(-2,4)],
        [(-3,6),(-3,6)]
    ])
    assert A * 1 == A
    assert 1 * A == A
    assert i * A == res
    assert A * i == res
