import numpy as np
import aesara
import aesara.tensor as aet
from aesara.tensor.var import TensorVariable
from biogeme.database import Database
from biogeme.expressions import Beta

class BetaShared(Beta):
    def __init__(self, name, value, lowerbound, upperbound, status):
        super().__init__(name, value, lowerbound, upperbound, status)

        self.sharedVar = aesara.shared(
            value=np.array(value).astype(aesara.config.floatX), name=name)

    def __call__(self):
        return self.sharedVar

    def __add__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return self.sharedVar + other
        else:
            return super().__add__(other)

    def __radd__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return self.sharedVar + other
        return super().__radd__(other)

    def __sub__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return self.sharedVar + other
        return super().__sub__(other)

    def __mul__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return self.sharedVar * other
        else:
            return super().__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return self.sharedVar / other
        return super().__truediv__(other)

    def __pow__(self, other):
        return super().__pow__(other)

    def __rsub__(self, other):
        return super().__rsub__(other)

    def __rmul__(self, other):
        return super().__rmul__(other)


class DatabaseShared(Database):
    def __init__(self, name, pandasDatabase, choiceVar):
        super().__init__(name, pandasDatabase)

        for v in self.variables:
            if v in self.data.columns:
                if self.variables[v].name == choiceVar:
                    self.variables[v].y = aet.ivector(self.variables[v].name)
                else:
                    self.variables[v].x = aet.matrix(self.variables[v].name)

    def get_x(self):
        list_of_x = []
        for var in self.variables:
            if hasattr(self.variables[var], "x"):
                list_of_x.append(self.variables[var].x)
        return list_of_x

    def get_x_data(self, index=None, batch_size=None, shift=None):
        x_data = []
        list_of_x = self.get_x()
        for x in list_of_x:
            if index == None:
                x_data.append(self.data[[x.name]])
            else:
                x_data.append(self.data[[x.name]][
                    index * batch_size + shift: (index+1) * batch_size + shift])
        return x_data

    def get_y(self):
        list_of_y = []
        for var in self.variables:
            if hasattr(self.variables[var], "y"):
                list_of_y.append(self.variables[var].y)
        return list_of_y

    def get_y_data(self, index=None, batch_size=None, shift=None):
        y_data = []
        list_of_y = self.get_y()
        for y in list_of_y:
            if index == None:
                y_data.append(self.data[y.name])
            else:
                y_data.append(self.data[y.name][
                    index * batch_size + shift: (index+1) * batch_size + shift])
        return y_data

    def inputs(self):
        return self.get_x() + self.get_y()
    
    def input_data(self, index=None, batch_size=None, shift=None):
        return self.get_x_data(index, batch_size, shift) + self.get_y_data(index, batch_size, shift)
    
    def autoscale(self, variables=None, verbose=False):
        for d in self.data:
            max_val = np.max(self.data[d])
            min_val = np.min(self.data[d])
            scale = 1.
            if variables is None:
                varlist = self.get_x()
            else:
                varlist = variables
            if d in varlist:
                if min_val >= 0.:
                    while max_val > 10:
                        scale *= 10.
                        self.data[d] = self.data[d] / scale
                        max_val = np.max(self.data[d])
                    if verbose:
                        print("scaling {} by {}".format(d, scale))
