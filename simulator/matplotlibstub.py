import platform
import matplotlib
if platform.system() == 'Darwin':
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')
