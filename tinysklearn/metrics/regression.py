def mean_absolute_error(y_true, y_pred):
    s = 0

    for i in range(len(y_true)):
        s += abs(y_true[i] - y_pred[i])

    return s / len(y_true)


def mean_squared_error(y_true, y_pred):
    s = 0
    for i in range(len(y_true)):
        d = y_true[i] - y_pred[i]
        s += d * d
    return s / len(y_true)

def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_t2rue, y_pred)
    return mse ** 0.5



def r2_score(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = 0
    ss_tot = 0

    for i in range(len(y_true)):
        ss_res += (y_true[i] - y_pred[i]) ** 2
        ss_tot += (y_true[i] - mean_y) ** 2
    if ss_tot == 0:
        return 0
    return 1 - (ss_res / ss_tot)




