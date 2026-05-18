# PROSTA REGUŁA
def predict_simple(row):
    if row['carat'] > 1:
        return 1
    else:
        return 0


# ZŁOŻONY MODEL (5 reguł)
def predict_complex(row):
    if row['carat'] > 1.5:
        return 1
    elif row['carat'] > 1 and row['cut'] in ['Premium', 'Ideal']:
        return 1
    elif row['color'] in ['D', 'E'] and row['carat'] > 0.5:
        return 1
    elif row['clarity'] in ['VVS1', 'IF']:
        return 1
    else:
        return 0


# REGRESJA
def predict_regression(row):
    if row['carat'] > 1.5:
        return 9000
    elif row['carat'] > 1:
        return 6000
    elif row['carat'] > 0.5:
        return 3000
    else:
        return 1000