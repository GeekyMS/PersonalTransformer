COUNTER = {'flops': 0, 'bytes': 0}

def count(flops, bytes_moved):
    COUNTER['flops'] += flops
    COUNTER['bytes'] += bytes_moved

def reset():
    COUNTER['flops'] = 0
    COUNTER['bytes'] = 0