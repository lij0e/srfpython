#!/usr/bin/env python

import numpy as np
import sys
for f in sys.argv[1:]:
    assert f.endswith('.npy')
for f in sys.argv[1:]:
    a = np.load(f)
    print(f, a.shape, a.dtype, a[:10])
