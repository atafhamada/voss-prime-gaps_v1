#!/usr/bin/env python3
"""تحليل نتائج VOSS"""
import os, pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')

def main():
    hist_path = os.path.join(RESULTS, 'gap_histogram.csv')
    if not os.path.exists(hist_path):
        print('Not found: ' + hist_path)
        return
    df = pd.read_csv(hist_path)
    print('Total gaps: ' + str(df['count'].sum()))
    print('Gap types : ' + str(len(df)))
    print()
    print('Top 10:')
    print(df.nlargest(10, 'count').to_string(index=False))

if __name__ == '__main__':
    main()
