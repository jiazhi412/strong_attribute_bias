import csv
from sklearn import preprocessing
import numpy as np
import torch


# statistics and bias split income, gender
def split_by_attr(xs, y_labels, a_labels):
    pp, pn, np, nn = [], [], [], []
    for i, (x, y, a) in enumerate(zip(xs, y_labels, a_labels)):
        if a == 1 and y == 1:
            pp.append(i)
        elif a == 1 and y == 0:
            pn.append(i)
        elif a == 0 and y == 1:
            np.append(i)
        elif a == 0 and y == 0:
            nn.append(i)
    return pp, pn, np, nn


def conditional_entropy(xs, y_labels, a_labels):
    pp, pn, np, nn = split_by_attr(xs, y_labels, a_labels)
    pp, pn, np, nn = len(pp), len(pn), len(np), len(nn)
    py = pp + pn
    ny = np + nn
    ap = np + pp
    an = nn + pn
    n = py + ny
    Pya = torch.tensor([[pp, pn], [np, nn]]) / n
    Pyca = torch.stack([torch.tensor([pp, pn]) / py, torch.tensor([np, nn]) / ny], dim=0)
    tmp = torch.log(Pyca)
    res = -(Pya * torch.log(Pyca)).sum()
    res = torch.nan_to_num(res)
    print(f"H(Y|A)={res}")
    return res.item()


# This function takes a column of the raw matrix and finds unique values for encoding
def uniqueItems(column):
    list = np.unique(column)
    fixedList = []
    for item in list:
        fixeListItem = [item]
        fixedList.append(fixeListItem)
    return fixedList


def newWidth(column):
    return len(np.unique(column))


def encodeColumn(oldCol, encoder):
    newCol = []
    for c in oldCol:
        c_array = np.array(c).reshape(-1, 1)
        newCol.append(encoder.transform(c_array).toarray())
    return np.array(newCol)


# Binarize the column with trueVal becoming +1 and everything else -1
def binarizeColumn(oldCol, trueVal):
    return [1 if x == trueVal else -1 for x in oldCol]


def data_processing(path):
    discrete_indices = [1, 3, 5, 6, 7, 8, 9, 13, 14]
    dicts = get_dict(path)
    # Read the raw data file into arrays
    with open(path) as rawDataFile:
        csvReader = csv.reader(rawDataFile, delimiter=",", quotechar="|")
        rows = []
        for row in csvReader:
            cols = []
            for i in range(len(row)):
                # Change the value here into a floating point number
                cur_dict = dicts[i]
                col = row[i]
                if i in discrete_indices:
                    value = float(cur_dict[col])
                else:
                    value = float(col)
                cols.append(value)
                i += 1
            rows.append(cols)

    rowCount = len(rows)
    colCount = len(rows[0])

    # read it into an ndarray
    arr = np.array(rows)

    # Data transformation by column:
    # Use One-Hot encoding for the categorical features
    # and use Gaussian normalization for numerical features
    #   (translate feature to have zero mean and unit variance)

    # 1 - age
    # 2 - workclass
    enc2 = preprocessing.OneHotEncoder()
    enc2.fit(uniqueItems(arr[:, 1]))
    # 3 - fnlwgt final weight
    # 4 - education
    enc4 = preprocessing.OneHotEncoder()
    enc4.fit(uniqueItems(arr[:, 3]))
    # 5 - educational-num
    # 6 - marital-status
    enc6 = preprocessing.OneHotEncoder()
    enc6.fit(uniqueItems(arr[:, 5]))
    # 7 - occupation
    enc7 = preprocessing.OneHotEncoder()
    enc7.fit(uniqueItems(arr[:, 6]))
    # 8 - relationship
    enc8 = preprocessing.OneHotEncoder()
    enc8.fit(uniqueItems(arr[:, 7]))
    # 9 - race
    enc9 = preprocessing.OneHotEncoder()
    enc9.fit(uniqueItems(arr[:, 8]))
    # 10 - gender
    enc10 = preprocessing.OneHotEncoder()
    enc10.fit(uniqueItems(arr[:, 9]))
    # 11 - capital-gain
    # 12 - capital-loss
    # 13 - hours-per-week
    # 14 - native-country
    enc14 = preprocessing.OneHotEncoder()
    enc14.fit(uniqueItems(arr[:, 13]))
    # 15 - income
    enc15 = preprocessing.OneHotEncoder()
    enc15.fit(uniqueItems(arr[:, 14]))

    # Create columns of new data
    col2 = encodeColumn(arr[:, 1], enc2)
    col4 = encodeColumn(arr[:, 3], enc4)
    col6 = encodeColumn(arr[:, 5], enc6)
    col7 = encodeColumn(arr[:, 6], enc7)
    col8 = encodeColumn(arr[:, 7], enc8)
    col9 = encodeColumn(arr[:, 8], enc9)
    col10 = encodeColumn(arr[:, 9], enc10)
    col14 = encodeColumn(arr[:, 13], enc14)
    col15 = encodeColumn(arr[:, 14], enc15)
    col1 = preprocessing.scale(arr[:, 0])  # numeric
    col3 = preprocessing.scale(arr[:, 2])  # numeric
    col5 = preprocessing.scale(arr[:, 4])  # numeric
    col11 = preprocessing.scale(arr[:, 10])  # numeric
    col12 = preprocessing.scale(arr[:, 11])  # numeric
    col13 = preprocessing.scale(arr[:, 12])  # numeric

    # the widths of the new columns
    w2 = newWidth(arr[:, 1])
    w4 = newWidth(arr[:, 3])
    w6 = newWidth(arr[:, 5])
    w7 = newWidth(arr[:, 6])
    w8 = newWidth(arr[:, 7])
    w9 = newWidth(arr[:, 8])
    w10 = newWidth(arr[:, 9])
    w14 = newWidth(arr[:, 13])
    w15 = newWidth(arr[:, 14])

    # Create a placeholder for new data
    newData = np.zeros((rowCount, w2 + w4 + w6 + w7 + w8 + w9 + w10 + w14 + w15 + 6))

    # populate the matrix with the new columns
    c = 0
    # index of current column (relative to old data)
    newData[:, c] = col1
    c = c + 1  # 1
    newData[:, c : c + w2] = col2.reshape((rowCount, w2))
    c = c + w2  # 2
    newData[:, c] = col3
    c = c + 1  # 3
    newData[:, c : c + w4] = col4.reshape((rowCount, w4))
    c = c + w4  # 4
    newData[:, c] = col5
    c = c + 1  # 5
    newData[:, c : c + w6] = col6.reshape((rowCount, w6))
    c = c + w6  # 6
    newData[:, c : c + w7] = col7.reshape((rowCount, w7))
    c = c + w7  # 7
    newData[:, c : c + w8] = col8.reshape((rowCount, w8))
    c = c + w8  # 8
    newData[:, c : c + w9] = col9.reshape((rowCount, w9))
    c = c + w9  # 9
    newData[:, c : c + w10] = col10.reshape((rowCount, w10))
    c = c + w10  # 10
    newData[:, c] = col11
    c = c + 1  # 11
    newData[:, c] = col12
    c = c + 1  # 12
    newData[:, c] = col13
    c = c + 1  # 13
    newData[:, c : c + w14] = col14.reshape((rowCount, w14))
    c = c + w14  # 14
    newData[:, c : c + w15] = col15.reshape((rowCount, w15))
    c = c + w15  # 15

    return newData


def quick_load(path):
    # Read the raw data file into arrays
    with open(path) as rawDataFile:
        csvReader = csv.reader(rawDataFile, delimiter=",")
        rows = []
        for row in csvReader:
            cols = []
            for col in row:
                # Change the value here into a floating point number
                value = float(col)
                cols.append(value)
            rows.append(cols)

    rowCount = len(rows)
    colCount = len(rows[0])

    # read it into an ndarray
    arr = np.array(rows)
    return arr


def get_dict(path):
    with open(path) as rawDataFile:
        csvReader = csv.reader(rawDataFile, delimiter=",", quotechar="|")
        char_indexs = [1, 3, 5, 6, 7, 8, 9, 13, 14]
        dicts = []
        for i in range(15):
            dicts.append(dict())

        for row in csvReader:
            for i in range(len(row)):
                if i in char_indexs:
                    col = row[i]
                    cur_dict = dicts[i]
                    if col not in cur_dict:
                        cur_dict[col] = len(cur_dict) + 1
                i += 1
    return dicts


def get_bias(data, bias_name):
    if bias_name == "label":
        bias_name = "income"
    corresponding_dict = {"age": (0, 1), "work_class": (1, 9), "final_weight": (10, 1), "education": (11, 16), "educational_num": (27, 1), "merital_status": (28, 7), "occupation": (35, 15), "relationship": (50, 6), "race": (56, 5), "sex": (61, 2), "capital_gain": (63, 1), "capital_loss": (64, 1), "hours_per_week": (65, 1), "native_country": (66, 42), "income": (108, 2)}
    address, length = corresponding_dict[bias_name]
    return data[:, address : address + length]


def perturb(data, bias_name):
    if bias_name == "label":
        bias_name = "income"
    corresponding_dict = {"age": (0, 1), "work_class": (1, 9), "final_weight": (10, 1), "education": (11, 16), "educational_num": (27, 1), "merital_status": (28, 7), "occupation": (35, 15), "relationship": (50, 6), "race": (56, 5), "sex": (61, 2), "capital_gain": (63, 1), "capital_loss": (64, 1), "hours_per_week": (65, 1), "native_country": (66, 42), "income": (108, 2)}
    address, length = corresponding_dict[bias_name]
    data[:, address : address + length] = torch.ones_like(data[:, address : address + length]) / length
    return data
