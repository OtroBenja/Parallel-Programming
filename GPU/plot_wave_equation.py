#Define functions

import pandas as pd
import numpy as np

#Import the data from the file
def read_data(filename):
    #Open file and read all first 11 lines
    data = open(filename)
    text = ''
    params = {}

    params1 = ['time','deltaR']
    for i in range(2):
        row   = data.readline()
        text += row
        params.update({params1[i]:float(row.split(':')[1])})
    params2 = ['maxR','iterations','nB','nT']
    for i in range(4):
        row   = data.readline()
        text += row
        params.update({params2[i]:int(row.split(':')[1])})
    params.update({'text':text})
    #Read data for x, u and u_t
    rdata = list(map(float,(data.readline().split(','))[:-1]))
    alldata = data.readlines()
    data.close()
    len_2 = int(len(alldata)/2)
    if(len_2*2 !=len(alldata)):
        raise ValueError(f'{len_2}*2 must be equal to {len(alldata)}')

    for i in range(2*len_2):
        alldata[i] = list(map(float,(alldata[i].split(','))[:-1]))
    udata = np.array(alldata[:len_2])
    u_tdata = np.array(alldata[len_2:])

    return rdata, udata, u_tdata, params

import matplotlib.animation as animation
import matplotlib.pyplot as plt

#Animation in paralel of scalar fields
def animate_data(filename, ipf = 1):
    r,u,u_t,params = read_data(filename)

    IPF = ipf #Iterations per frame
    iterations = len(u)
    if 2*ipf>iterations:
        raise ValueError(f'ipf = {ipf} must be at least half of iterations = {iterations}')
    FRAMES = int(iterations/IPF)

    fig, ax = plt.subplots(2,1,figsize=(12,12),dpi=100)

    Uplot  = ax[0].plot(r,  u[0])[0]
    UTplot = ax[1].plot(r,u_t[0])[0]
    ax[1].set_ylim( -ax[0].get_ylim()[1],ax[0].get_ylim()[1])
    #Make limits for phi plot based on maximum initial difference from external point
    deltaLim = max(u[0][-1] -ax[0].get_ylim()[0], ax[0].get_ylim()[1] -u[0][-1])
    ax[0].set_ylim(u[0][-1]-deltaLim,u[0][-1]+deltaLim)
    ax[0].set_title('$u$')
    ax[1].set_title('$\partial_t u$')

    #Create a function that changes the data every frame
    def animate(i):
        it = i*IPF
        Uplot.set_data(r,u[it])
        UTplot.set_data(r,u_t[it])
        return None

    #Make the animation
    anim = animation.FuncAnimation(fig, animate, frames=FRAMES,
                                    interval=100, repeat_delay=3000)

    anim.save(filename[:-3]+'gif',writer='pillow')

from os import listdir

#Search last file to plot

folder_files = listdir()
folder_files = [f for f in folder_files if (f.startswith('Output') and f.endswith('.dat'))]
final_file = folder_files[-1]
animate_data('Output_162319.dat',ipf=2)