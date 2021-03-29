"""Parses input for linear constraint problem"""
import numpy as np
class LPInputParser:
    def __init__(self, filepath, report_mode='out'):
        self.filepath   = filepath
        self.done       = False
        self.numpy_data = None
        self.list_data  = None
        self.file_content = []
        self.up_to_date = False
        if report_mode not in ['exceptions', 'out']:
            raise ValueError('`report_mode` can be either `exceptions` or `out`')
        self.report_mode = report_mode

    def report_parse_error(self, msg):
        err_msg = f'Parse error: {msg}'
        if self.report_mode == 'exceptions':
            raise ValueError(f'Parse error encountered:{msg}')
        else:
            return {'errors': [err_msg], 'data': None}

    def fetch_file_content(self):
        file = open(self.filepath, "r")
        lines = file.readlines()
        lines = (line.strip() for line in lines)
        lines = (line for line in lines if line)
        lines = list(lines)
        file.close()
        if len(lines) == len(self.file_content) and all(a == b for a, b in zip(lines, self.file_content)):
            return
        
        self.file_content = lines
        self.up_to_date = False

 
    def get_numpy(self):
        if self.numpy_data is not None:
            return self.numpy_data
        raise NotImplementedError()

    def get_list(self):
        if self.list_data is not None:
            return self.list_data
        raise NotImplementedError()
