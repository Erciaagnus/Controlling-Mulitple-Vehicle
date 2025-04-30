import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/henricus/Controlling-Mulitple-Vehicle/install/cpnav'
