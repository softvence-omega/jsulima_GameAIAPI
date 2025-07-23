def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_float(value, default=0.0):
    try:
        return round(float(value), 2)
    except (ValueError, TypeError):
        return round(default, 2)


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0