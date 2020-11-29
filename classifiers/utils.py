from sklearn.preprocessing import OneHotEncoder

def encode(y):
    y_ = [[i] for i in y]
    y_ = OneHotEncoder(sparse=False).fit_transform(y_)
    return y_