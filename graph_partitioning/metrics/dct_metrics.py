import numpy as np
import math

def quantify_confusion(N, f=lambda x: x * (x - 1) * 0.5, exact=False):
    if len(N)<=0: return 0
    f = np.vectorize(f, otypes=[np.float])
    b2 = np.ones((N.shape[1], 1), dtype='float')
    b1 = np.ones((1, N.shape[0]), dtype='float')
    m1 = b1.dot(N)
    m2 = N.dot(b2)
    m = b1.dot(N).dot(b2)
    s1 = (b1.dot(f(m2)))
    s2 = (f(m1).dot(b2))
    n = f(m)
    I = (b1.dot(f(N)).dot(b2))
    if exact:
        E = (b1.dot(f(m2).dot(f(m1)) / f(m))).dot(b2)
    else:
        E = (b1.dot(f(m2.dot(m1) / m))).dot(b2)
    AG = float((s1 + s2 - 2 * I) / ((s1 + s2) - 2 * E))
    return 1 - AG



def nmi(e): return quantify_confusion(e, f=lambda x: x * math.log(x) if x > 0 else 0)
def kappa(e): return quantify_confusion(e, f=lambda x: x * x )
def ari(e): return quantify_confusion(e, f=lambda x: x * (x - 1), exact=True)
def ari2(e): return quantify_confusion(e, f=lambda x: x * (x - 1))

'''
def test():
    ct = np.array([[3, 1], [0, 3], [3, 0]])
    nct = ct*1.0/np.sum(ct)

    print nct
    print "NMI: " , nmi(ct), " == ", nmi(nct)
    print "kappa: " , kappa(ct), " == ", kappa(nct)

    # more variations, you can play with function f to get new measures
    print "other variations: ", ari(ct), ari2(ct)


if __name__ == '__main__':
    test()
    '''
