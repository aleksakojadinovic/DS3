import numpy as np
import sys
import os

BASE_CMD = 'python3 transport.py -i'

def get_run_command(fname):
    return f'{BASE_CMD} {fname}'

def run_and_fetch(filepath):
    os.system(get_run_command(filepath) + ' > tmp')
    return open('tmp', 'r').read()

def open_and_fetch(filepath):
    return open(filepath, 'r').read()

if __name__ == '__main__':
    input_files = map(lambda x: os.path.join('examples', x), os.listdir('./examples'))
    result_files = map(lambda x: os.path.join('results', x), os.listdir('./results'))

    results = map(run_and_fetch, input_files)
    results = list(map(float, results))
    print(f'Running tests...')
    expected_results = map(open_and_fetch, result_files)
    expected_results = list(map(float, expected_results))
    print('=============================================')


    passed = 0
    failed = 0
    total = len(results)
    for i, (r, er) in enumerate(zip(results, expected_results)):
        if r == er:
            passed += 1
        else:
            failed += 1
            print(f'Failed for example {i}, expected {r} got {er}')

    if failed > 0:
        print('===========================================')
    print(f'Summary: ')
    print(f'Total: {total}')
    print(f'Passed: {passed}')
    print(f'Failed: {failed}')


    

    
