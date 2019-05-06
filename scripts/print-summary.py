#!/usr/bin/python3

import subprocess
import os

dname = '../images/'

files = ['ra12-2014-summary.png',
         'ra12-2015-summary.png',
         'ra15-summary.png',
         'nrl1-summary.png',
         'nrl2-summary.png',
         'nrl3-summary.png',
         'nrl4-summary.png',
         'nrl5-summary.png']


for ff in files[2:3]:
    if os.path.isfile(dname + ff):
        subprocess.call('convert ' + dname + ff
                        + ' -background white -alpha remove new.png',
                        shell=True)
        subprocess.run(['lp', '-d', 'Burt428',
                        '-o', 'orientation-requested=4',
                        'new.png'])
