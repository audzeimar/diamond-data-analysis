import seaborn as sns


def load_data():
    # Wczytuje zbiór danych diamonds.

    try:
        return sns.load_dataset('diamonds')
    
    except Exception:
        from plotnine.data import diamonds
        return diamonds.copy()
