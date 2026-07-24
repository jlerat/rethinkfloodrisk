from scipy.special import gamma, digamma

class DualNum():
    def __init__(self, x, dx=0):
        if isinstance(x, DualNum):
            self.x = x.x
            self.dx = x.dx
        else:
            self.x = x
            self.dx = dx

    def __str__(self):
        return f"x={self.x} dx={self.dx}"

    def __add__(self, other):
        other = DualNum(other)
        return DualNum(self.x + other.x, self.dx + other.dx)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return DualNum(-self.x, -self.dx)

    def __sub__(self, other):
        other = DualNum(other)
        return DualNum(self.x - other.x, self.dx - other.dx)

    def __rsub__(self, other):
        o = self - other
        o.x *= -1
        o.dx *= -1
        return o

    def __mul__(self, other):
        other = DualNum(other)
        return DualNum(self.x * other.x,
                       self.x * other.dx + self.dx * other.x)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = DualNum(other)
        return DualNum(self.x / other.x,
                       (self.dx - self.x * other.dx / other.x) / other.x)

    def __pow__(self, alpha):
        return DualNum(self.x ** alpha,
                         self.dx * self.x ** (alpha - 1))

    def __rpow__(self, alpha):
        ax = alpha ** self.x
        return DualNum(ax, self.dx * math.log(alpha) * ax)

    def apply(self, fun, dfun):
        fx = fun(self.x)
        dfx = dfun(self.x)
        return DualNum(fx, self.dx * dfx)

    def gamma(self):
        dgamma = lambda x: gamma(x) * digamma(x)
        return self.apply(gamma, dgamma)

    def log(self):
        return self.apply(math.log, lambda x: 1./x)

    def exp(self):
        return self.apply(math.exp, math.exp)

