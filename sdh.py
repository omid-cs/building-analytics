
# coding: utf-8

# In[1]:

import os
import sys
from smap.archiver.client import SmapClient
from smap.contrib import dtutil
import requests
import btrdb

from scipy import stats
from scipy.ndimage import *
from scipy.signal import periodogram
from scipy.signal import welch
from scipy.signal import argrelextrema
from scipy.signal import hilbert
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.nonparametric.api as nparam


from scipy.cluster.hierarchy import fcluster, linkage, cophenet
from scipy.spatial.distance import pdist
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import numpy as np
import scipy.fftpack as fft
from math import *
import time
import datetime
from datetime import timedelta

import operator

import matplotlib.pyplot as plt
from matplotlib import pylab

from numpy.linalg import norm
from dtw import dtw

from smap.archiver.client import SmapClient
from mpl_toolkits.axes_grid1 import host_subplot

import pyeemd
from pyeemd.utils import plot_imfs

get_ipython().magic(u'matplotlib inline')
#%pylab inline


# In[2]:


####################
# GLOBAL CONSTANTS #
####################

startDate = "2016-03-14T07:00:00"
endDate = "2016-06-06T07:00:00"

# startDate = "2015-07-22T07:00:00"
# endDate = "2015-08-29T07:00:00"

# startDate = "2015-07-06T07:00:00"
# endDate = "2015-07-26T07:00:00"

# startDate = "2012-09-10T07:00:00"
# endDate = "2012-11-26T07:00:00"
# endDate = "2012-12-10T07:00:00"

firstday = 0 # i.e. Monday
daylbl = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_labels = daylbl[firstday:] + daylbl[:firstday]

smap_client = SmapClient("http://castle.cs.berkeley.edu:8079")

startDate_dt = datetime.datetime.strptime(startDate, "%Y-%m-%dT%H:%M:%S")
endDate_dt = datetime.datetime.strptime(endDate, "%Y-%m-%dT%H:%M:%S") - datetime.timedelta(seconds=1)

startDate_dt_local = startDate_dt - timedelta(hours=7)
endDate_dt_local = endDate_dt - timedelta(hours=7)


data_num_weeks = (endDate_dt_local - startDate_dt_local).days / 7 + 1
print "Number of weeks : ", data_num_weeks

start = btrdb.date(startDate)
end = btrdb.date(endDate)

pw = 39

oat_uuid_stanley = "b8e57175-78c1-43db-85a7-928e0e7545c1"

windowsize = 10
windowsizestring = str(windowsize) + 'min'

period = windowsize * 60 

# number of periods per day
numperiods = ( 60 * 24 ) / windowsize 

data = {}

fill_method_string = "ffill"

avg_energy_loss = {}
avg_weekly_occupancy_bitmap = {}

schedule_nighttime = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 92, 93, 94, 95 }

room_list = {}
tsdata = {}

maxnumreadings = 0


# In[107]:

# canny edge detection algorithm

def edgeDetector( data, scales, thresholds, bw, pltFlag, sagFlag=True):
    # Smooth out data using a Gaussian kernel
    if bw != 0:
        sData = list(filters.gaussian_filter(data, bw))
    else:
        sData = data
    
    # Create the derivative scale space (minima and maxima of the derivative correspond to transitions)
    # Scale is the standard deviation of the Gaussian distribution that its derivative is convolved with data
    dData = createGaussScaleSpace( sData, scales )
    
    # Find local minima and maxima of the most coarse scale
    # Parameter 'scale' determines the length of the window (2*scale) used to find local minima/maxima
    # A local optimum is considered only if the ratio of it to the global optimum is greater than the specified 'threshold'
    if sagFlag:
        minmax = findLocalExtrema_Sag( dData[-1], thresholds[-1], scales[-1])
    else:
        minmax = findLocalExtrema_Surge( dData[-1], thresholds[-1], scales[-1])
    minmaxIdx = np.nonzero(minmax)[0]
    
    # Refine local min/max through scale space
    for i in reversed(range(len(scales)-1)):
        if sagFlag:
            minmax = findLocalExtremaInRange_Sag( dData[i], thresholds[i], scales[i], minmaxIdx)
        else:
            minmax = findLocalExtremaInRange_Surge( dData[i], thresholds[i], scales[i], minmaxIdx)
        minmaxIdx = np.nonzero(minmax)[0]
    
    if pltFlag:
        pylab.rcParams['figure.figsize'] = (30.0, 10.0)
#         plt.figure()
#         plt.plot(dData[-1])
#         plt.xlim(0, len(dData[-1]))
#         numdays = len(dData[-1])/numperiods
#         labels = day_labels*(numdays/7)+day_labels[0:(numdays%7)]
#         plt.xticks(np.arange(0, len(dData[-1]), numperiods), labels, rotation=90)
#         plt.plot([0.5 * mm * max(dData[-1]) for mm in minmax], 'r')
#         plt.show()
        
        fig, ax = plt.subplots()
        plt.plot(data);
        plt.xlim(0, len(data))
        numdays = len(data)/numperiods
        labels = day_labels*(numdays/7)+day_labels[0:(numdays%7)]
        plt.xticks(np.arange(0, len(data), numperiods), labels, rotation=90)
        plt.ylabel('Damper Position (Percent)')
        plt.plot([1.0 * mm * max(data) for mm in minmax], 'r')
        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
        
        plt.show()
    
    starts = []
    ends = []
    for i in range(len(minmaxIdx)):
        if i%2 == 0:
            starts.append(minmaxIdx[i])
        else:
            ends.append(minmaxIdx[i])
    
#     if len(starts)>len(ends):
#         ends.append(len(data)-1)
    return zip(starts,ends)

def createGaussScaleSpace( data, scales ):
    space = []
    for sidx in range(len(scales)):
        sigma = scales[sidx]
        width = 3 # extend the kernel by 3*sigma from its center

        prange = list(range(1, 2*width*sigma + 2))
        center = prange[(len(prange)/2)]

        # first derivative of the Gaussian kernel
        kernel = [(1/(sigma*sqrt(2*pi))) * exp(-((point-center)**2)/float(2*(sigma**2))) * (-((point-center)/float(sigma**2))) for point in prange]
        
        # pad data
        padData = [data[1]]*len(kernel) + data + [data[-1]]*len(kernel)
        fData = np.convolve(padData, kernel, mode='same')

        # Resize the filtered data
        offset = float(len(fData) - len(data)) / 2
        fData = fData[offset-1:offset+len(data)-1]
        
        space.append(fData)
    return space
    
def findLocalExtrema_Sag( data, threshold, scale ):
    cautious = True
    
    rdataMax = [float(d)/max(data) for d in data]
    rdataMin = [float(d)/min(data) for d in data]
    
    winmax = [False]*len(data)
    winmin = [False]*len(data)
    maxima = [False]*len(data)
    minima = [False]*len(data)
    
    for i in range(scale,len(data)-scale):
        winmax[i] = max(rdataMax[i-scale:i+scale+1])
        winmin[i] = max(rdataMin[i-scale:i+scale+1])
    
        maxima[i] = rdataMax[i] >= threshold and rdataMax[i] >= winmax[i]
        minima[i] = rdataMin[i] >= threshold and rdataMin[i] >= winmin[i]
    
    # merge boolean lists
    lastmaximaidx = 0   # last end index
    lastminimaidx = 0   # last start index
    maxminlst = []
    for i in range(len(maxima)):
        # Found a local minimum: potential occ start time
        if minima[i]: 
            # Add if the previous occ interval has ended
            if lastmaximaidx >= lastminimaidx:
                if (i%numperiods) >= (numperiods/6): # after 4am
                    maxminlst.append(True)
                    lastminimaidx = i
                else:
                    maxminlst.append(False)
                    
#                 maxminlst.append(True)
#                 lastminimaidx = i
                    
            # If previous occ interval started on a different day has not ended yet, 
            # then add an end time for that day and start a new occ interval
            elif (i/numperiods)-(lastminimaidx/numperiods) >= 1:
#                 lastmaximaidx = ((lastminimaidx/numperiods)+1)*numperiods-1
#                 maxminlst[lastmaximaidx] = True

                maxminlst[lastminimaidx] = False
                if (i%numperiods) >= (numperiods/6): # after 4am
                    maxminlst.append(True)
                    lastminimaidx = i
                else:
                    maxminlst.append(False)
                    
#                 maxminlst.append(True)
#                 lastminimaidx = i
                    
            # If previous occ interval started on the same day has not ended yet, ignore the local minimum
            else:
                maxminlst.append(False)
        
        # Found a local maximum: potential occ end time
        elif maxima[i]:
            # Check if an occ interval has started earlier
            if lastmaximaidx < lastminimaidx:
                if cautious:
                    # If started on a different day, end it on the same day and ignore the present local maximum
                    if (i/numperiods)-(lastminimaidx/numperiods) >= 1:
#                         lastmaximaidx = ((lastminimaidx/numperiods)+1)*numperiods-1
#                         # end time is the same as start time
#                         if maxminlst[lastmaximaidx]:
#                             lastmaximaidx = lastmaximaidx + 1
#                             maxminlst[lastmaximaidx] = True
#                         else:
#                             maxminlst[lastmaximaidx] = True

                        maxminlst[lastminimaidx] = False

                        maxminlst.append(False)
                    # If started on the same day, end it now
                    else:
                        maxminlst.append(True)
                        lastmaximaidx = i
                else:
                    maxminlst.append(True)
                    lastmaximaidx = i
            # If an occ interval has not yet started and the previous one ended on a different day, ignore the present one
            elif (i/numperiods)-(lastmaximaidx/numperiods) >= 1 and lastmaximaidx>0:
                maxminlst.append(False)
            # If an occ interval has not yet started and the previous one ended on the same day, update the end time
            elif lastmaximaidx>0:
                maxminlst[lastmaximaidx] = False
                maxminlst.append(True)
                lastmaximaidx = i
            # Ignore if no occ interval has been detected yet
            else:
                maxminlst.append(False)

        else:
            maxminlst.append(False)
    
    return maxminlst

def findLocalExtrema_Surge( data, threshold, scale ):
    cautious = True
    
    rdataMax = [float(d)/max(data) for d in data]
    rdataMin = [float(d)/min(data) for d in data]
    
    winmax = [False]*len(data)
    winmin = [False]*len(data)
    maxima = [False]*len(data)
    minima = [False]*len(data)
    
    for i in range(scale,len(data)-scale):
        winmax[i] = max(rdataMax[i-scale:i+scale+1])
        winmin[i] = max(rdataMin[i-scale:i+scale+1])
    
        maxima[i] = rdataMax[i] >= threshold and rdataMax[i] >= winmax[i]
        minima[i] = rdataMin[i] >= threshold and rdataMin[i] >= winmin[i]
    
    # merge boolean lists
    lastmaximaidx = 0   # last start index
    lastminimaidx = 0   # last end index
    maxminlst = []
    for i in range(len(maxima)):
        # Found a local minimum: potential occ end time
        if minima[i]: 
            # Check if an occ interval has started earlier
            if lastmaximaidx > lastminimaidx:
                if cautious:
                    # If started on a different day, end it on the same day and ignore the present local maximum
                    if (i/numperiods)-(lastmaximaidx/numperiods) >= 1:
#                         lastminimaidx = ((lastmaximaidx/numperiods)+1)*numperiods-1
#                         # end time is the same as start time
#                         if maxminlst[lastminimaidx]:
#                             lastminimaidx = lastminimaidx + 1
#                             maxminlst[lastminimaidx] = True
#                         else:
#                             maxminlst[lastminimaidx] = True

                        maxminlst[lastmaximaidx] = False

                        maxminlst.append(False)
                    # If started on the same day, end it now  
                    else:
                        maxminlst.append(True)
                        lastminimaidx = i
                else:
                    maxminlst.append(True)
                    lastminimaidx = i
            # If an occ interval has not yet started and the previous one ended on a different day, ignore the present one
            elif (i/numperiods)-(lastminimaidx/numperiods) >= 1 and lastminimaidx>0:
                maxminlst.append(False)
            # If an occ interval has not yet started and the previous one ended on the same day, update the end time
            elif lastminimaidx>0:
                maxminlst[lastminimaidx] = False
                maxminlst.append(True)
                lastminimaidx = i
            # Ignore if no occ interval has been detected yet
            else:
                maxminlst.append(False)
        
        # Found a local maximum: potential occ start time
        elif maxima[i]:
            # Add if the previous occ interval has ended
            if lastmaximaidx <= lastminimaidx:
                if (i%numperiods) >= (numperiods/6): # after 4am
                    maxminlst.append(True)
                    lastmaximaidx = i
                else:
                    maxminlst.append(False)
                    
#                 maxminlst.append(True)
#                 lastmaximaidx = i
                    
            # If previous occ interval started on a different day has not ended yet, 
            # then add an end time for that day and start a new occ interval
            elif (i/numperiods)-(lastmaximaidx/numperiods) >= 1:
#                 lastminimaidx = ((lastmaximaidx/numperiods)+1)*numperiods-1
#                 maxminlst[lastminimaidx] = True

                maxminlst[lastmaximaidx] = False
                if (i%numperiods) >= (numperiods/6): # after 4am
                    maxminlst.append(True)
                    lastmaximaidx = i
                else:
                    maxminlst.append(False)

#                 maxminlst.append(True)
#                 lastmaximaidx = i
        
            # If previous occ interval started on the same day has not ended yet, ignore the local minimum
            else:
                maxminlst.append(False)
        # Ignore if not a local optimum
        else:
            maxminlst.append(False)
    
    return maxminlst

def findLocalExtremaInRange_Sag( data, threshold, scale, regions ):
    rdataMax = [float(d)/max(data) for d in data]
    rdataMin = [float(d)/min(data) for d in data]
    
    winmax = [False]*len(data)
    winmin = [False]*len(data)
    maxima = [False]*len(data)
    minima = [False]*len(data)
    
    # create a sliding window min & max
    for i in range(scale,len(data)-scale):
        winmax[i] = max(rdataMax[i-scale:i+scale+1])
        winmin[i] = max(rdataMin[i-scale:i+scale+1])
        
    for r in range(len(regions)):
        for i in range(max(scale, regions[r]-scale), min(len(data)-scale, regions[r]+scale)):
            maxima[i] = (rdataMax[i] >= threshold) and (rdataMax[i] >= winmax[i])
            minima[i] = (rdataMin[i] >= threshold) and (rdataMin[i] >= winmin[i])
    
    # merge boolean lists
    lastmaximaidx = 0
    lastminimaidx = 0
    maxminlst = []
    for i in range(len(maxima)):
        if minima[i]: 
            if lastmaximaidx >= lastminimaidx:
                maxminlst.append(True)
                lastminimaidx = i
#             elif rdataMin[i] > rdataMin[lastminimaidx] and lastminimaidx>0:
#                 maxminlst[lastminimaidx] = False
#                 maxminlst.append(True)
#                 lastminimaidx = i
            else:
                maxminlst.append(False)
        elif maxima[i]:
            if lastmaximaidx < lastminimaidx:
                maxminlst.append(True)
                lastmaximaidx = i
#             elif rdataMax[i] > rdataMax[lastmaximaidx] and lastmaximaidx>0:
#                 maxminlst[lastmaximaidx] = False
#                 maxminlst.append(True)
#                 lastmaximaidx = i
            else:
                maxminlst.append(False)
        else:
            maxminlst.append(False)
    
    return maxminlst

def findLocalExtremaInRange_Surge( data, threshold, scale, regions ):
    rdataMax = [float(d)/max(data) for d in data]
    rdataMin = [float(d)/min(data) for d in data]
    
    winmax = [False]*len(data)
    winmin = [False]*len(data)
    maxima = [False]*len(data)
    minima = [False]*len(data)
    
    # create a sliding window min & max
    for i in range(scale,len(data)-scale):
        winmax[i] = max(rdataMax[i-scale:i+scale+1])
        winmin[i] = max(rdataMin[i-scale:i+scale+1])
        
    for r in range(len(regions)):
        for i in range(max(scale, regions[r]-scale), min(len(data)-scale, regions[r]+scale)):
            maxima[i] = (rdataMax[i] >= threshold) and (rdataMax[i] >= winmax[i])
            minima[i] = (rdataMin[i] >= threshold) and (rdataMin[i] >= winmin[i])
    
    # merge boolean lists
    lastmaximaidx = 0
    lastminimaidx = 0
    maxminlst = []
    for i in range(len(maxima)):
        if minima[i]: 
            if lastmaximaidx > lastminimaidx:
                maxminlst.append(True)
                lastminimaidx = i
#             elif rdataMin[i] > rdataMin[lastminimaidx] and lastminimaidx>0:
#                 maxminlst[lastminimaidx] = False
#                 maxminlst.append(True)
#                 lastminimaidx = i
            else:
                maxminlst.append(False)
        elif maxima[i]:
            if lastmaximaidx <= lastminimaidx:
                maxminlst.append(True)
                lastmaximaidx = i
#             elif rdataMax[i] > rdataMax[lastmaximaidx] and lastmaximaidx>0:
#                 maxminlst[lastmaximaidx] = False
#                 maxminlst.append(True)
#                 lastmaximaidx = i
            else:
                maxminlst.append(False)
        else:
            maxminlst.append(False)
    
    return maxminlst

def mergeTwoSetsofTuples( listoftuples1, listoftuples2 ):
    # sort based on the start time
    s = sorted(listoftuples1 + listoftuples2)

    mergedlist = [ s[0] ]
    for sitem in s[1:]:
        if mergedlist[-1][1] < sitem[0]:
            mergedlist.append(sitem)
        else:
            temp = mergedlist[-1][0]
            del mergedlist[-1]
            mergedlist.append((temp,sitem[1]))

    return mergedlist
    
def invertSetofTuples( listoftuples, alltimeslots ):
    invertedlist = []

    lasttime = int(alltimeslots[0])
    for i in range(len(listoftuples)):
        invertedlist.append((lasttime,listoftuples[i][0]-1))
        lasttime = listoftuples[i][1]+1

    invertedlist.append((lasttime,int(alltimeslots[-1])))
    return invertedlist

def getIndicesWithinIntervals( listoftuples ):
    idxList = []
    for atuple in listoftuples:
        idxList.extend(range(atuple[0],atuple[1]+1))
    return idxList

def reconstructSignal( signal1, idx1, signal2, idx2 ):
    sigout = [0]*(len(idx1) + len(idx2))
    for i in range(len(idx1)):
        sigout[int(idx1[i])] = signal1[i]
    for i in range(len(idx2)):
        sigout[int(idx2[i])] = signal2[i]
    return sigout


# In[4]:

#Debug
# t, x1 = getIndicativeSignalForZone('240 CLASSROOM')
# edgeDetector( x1, [3], [.05], 0, True, False)


# In[102]:

# Gets the data for a uuid and returns a windowed resampled pandas series
def getAndFormatData(zone, stream_name, uuid):
    global maxnumreadings
    r = requests.get("http://{}:{}/data/uuid/{}?starttime={}&endtime={}&pw={}&unitoftime=ns"
            .format("castle.cs.berkeley.edu", 9000, uuid, start, end, pw))

    sensor_dict = {}

    for x in r.json()[0]["XReadings"]:
        if datetime.datetime.fromtimestamp(int(x[0])/1000) > startDate_dt_local and datetime.datetime.fromtimestamp(int(x[0])/1000) < endDate_dt_local:
            sensor_dict[datetime.datetime.fromtimestamp(int(x[0])/1000)] = x[3] 

    sensor_dict[startDate_dt_local] = sensor_dict[sorted(sensor_dict)[0]]
    sensor_dict[endDate_dt_local] = sensor_dict[sorted(sensor_dict, reverse=True)[0]]
    if "numreadings" not in room_list[zone]:
        room_list[zone]["numreadings"] = {}
    room_list[zone]["numreadings"][stream_name] = len(sensor_dict)
    if maxnumreadings < len(sensor_dict):
        maxnumreadings = len(sensor_dict)
        
    pd_sensor = pd.Series(sensor_dict)
    resampled_p = pd_sensor.resample(windowsizestring, how=np.mean, fill_method=fill_method_string )
    return resampled_p

# Formats the pandas objects into a data array
def addToData(name, pandas_series, data, first=False):
    indices = pandas_series.index
    for i in range(len(pandas_series)):
        if first==True:
            data[time.mktime(indices[i].to_datetime().timetuple())] = {}
        
        data[time.mktime(indices[i].to_datetime().timetuple())][name] = pandas_series[i]

    return data


def getSortedNamedStream(zone, stream_name):  
    
    if stream_name in tsdata[zone]:
        #print "Returning cached data"
        #print "Returned length of data : ", len(tsdata[zone][stream_name])
        return tsdata[zone][stream_name]
    
    print "Getting data for zone : %s , stream_name : %s" % (zone, stream_name)
    stream_uuid = room_list[zone][stream_name]
    pd_stream = getAndFormatData(zone, stream_name, stream_uuid)
    
    data = {}
    addToData(stream_name, pd_stream, data, True)
    
    retV = []
    
    for t in sorted(data):
        retV.append((t, data[t][stream_name]))
        
    retT = range(len(retV))
    print "Length of series %s:%s is %d " % (zone, stream_name, len(retT) )
    retV = sorted(retV, key=lambda k:k[0])
    times = [ int(t) for t in range(len(retV))]
    values = [ v for (t,v) in retV ]
    return (times, values)
def isDataMissing(zone):
    global maxnumreadings
    if zone not in room_list:
        return True
    
    for stream in room_list[zone]["numreadings"]:
        if room_list[zone]["numreadings"][stream] < 0.2 * maxnumreadings:
            return True
        
    return False

# segmentation algorithm
def runSegmentationForZone(t, x1, plotFlag, USE_ALL=False, sagFlag=True, bw=0):

    if USE_ALL:
    # use the entire data set in model parameter estimation
        training_set = range(len(t))
    else:
    # use a subset of the data set in model parameter estimation
        training_set = range(numperiods*7)

    intervals1 = edgeDetector( [x1[training_idx] for training_idx in training_set], [6], [.15], bw, plotFlag, sagFlag)
#     intervals2 = edgeDetector( [x3[training_idx] for training_idx in training_set], [1, 4, 8], [.15, .15, .15], 5, plotFlag)

    occupied_intervals = intervals1
#     occupied_intervals = mergeTwoSetsofTuples(intervals1,intervals2)
    
    # print occupied_intervals
    unoccupied_intervals = invertSetofTuples(occupied_intervals, [t[training_idx] for training_idx in training_set])
    # print unoccupied_intervals
    unoccupied_idx = getIndicesWithinIntervals(unoccupied_intervals)
    # print unoccupied_idx

    if not USE_ALL:
        occupied_intervals.append((int(t[training_set[-1]+1]),int(t[-1])))
    occupied_idx = getIndicesWithinIntervals(occupied_intervals)
    return (occupied_intervals, occupied_idx, unoccupied_intervals, unoccupied_idx)


# In[6]:

offices = ['266C OFFICE' , '330F OFFICE' , '330E OFFICE' , '330C OFFICE' , '330B OFFICE' , '356D OFFICE' , '356F OFFICE' , '413 OFFICE' , '446 OFFICE' , '421 OFFICE' , '424 OFFICE' , '423 OFFICE' , '448 OFFICE' , '464 OFFICE' , '510 OFFICE' , '511 OFFICE' , '515 OFFICE' ,  '558 OFFICE' , '640 OFFICE' , '646 OFFICE' , '621C OFFICE' , '621A OFFICE' , '652A OFFICE' , '656B OFFICE' , '725 OFFICE' , '721 OFFICE' , '722 OFFICE' , '756 OFFICE' , '766 OFFICE']
open_offices = ['356 OPEN OFFICE' , '432 OPEN OFFICE' , '434 OPEN OFFICE' , '440 OPEN OFFICE' , '450 OPEN OFFICE' , '460 OPEN OFFICE' , '466 OPEN OFFICE' , '468 OPEN OFFICE' , '532 OPEN OFFICE' , '538 OPEN OFFICE' , '548 OPEN OFFICE' , '550 OPEN OFFICE' , '568 OPEN OFFICE' , '566 OPEN OFFICE' , '736 OPEN OFFICE' , '738 OPEN OFFICE' , '750 OPEN OFFICE' , '768 OPEN OFFICE']
conference_rooms = ['368 CONFERENCE' , '458 CONFERENCE' , '554 CONFERENCE' , '630 CONFERENCE' , '730 CONFERENCE']
class_rooms = ['240 CLASSROOM', '250 CLASSROOM', '254 CLASSROOM']
selected_zones = offices+open_offices+conference_rooms


# In[7]:

#############################################
# POPULATING METADATA AND DATA OF ALL ZONES #
#############################################

results = smap_client.query("select distinct Metadata/RoomId where Metadata/SourceName='SutardjaDaiHall'")

count = 1
for room in results:
    try:
        room = str(room)
        print "Doing room : %s (%d/%d)" % (room, count, len(results))
        count += 1
        if room == "all":
            continue
        ret = smap_client.query("select uuid where Metadata/RoomId='%s' and Metadata/sensor='room_temperature'" % room)
        if len(ret) == 0:
            continue
        temp_uuid = str(ret[0]["uuid"])
        ret = smap_client.query("select uuid where Metadata/RoomId='%s' and Metadata/sensor='room_damper_position'" %room)
        if len(ret) == 0:
            continue
        flow_uuid = str(ret[0]["uuid"])
        ret = smap_client.query("select uuid where Metadata/RoomId='%s' and Metadata/sensor='room_valve_position'" % room)
        if len(ret) == 0:
            continue
        reheat_uuid = str(ret[0]["uuid"])
        ret = smap_client.query("select uuid where Metadata/RoomId='%s' and Metadata/sensor='room_airflow'" % room)
        if len(ret) == 0:
            continue
        volume_uuid = str(ret[0]["uuid"]) 
        ret = smap_client.query("select uuid where Metadata/RoomId='%s' and Metadata/sensor='room_minimum_airflow'" % room)
        if len(ret) == 0:
            continue
        minflow_uuid = str(ret[0]["uuid"]) 

        room_list[room] = {}
        room_list[room]["temp"] = temp_uuid
        room_list[room]["flow"] = flow_uuid
        room_list[room]["reheat"] = reheat_uuid
        room_list[room]["volume"] = volume_uuid
        room_list[room]["minflow"] = minflow_uuid
        
        tsdata[room] = {}
        tsdata[room]["reheat"] = getSortedNamedStream(room,"reheat")
        tsdata[room]["flow"] = getSortedNamedStream(room, "flow")
        tsdata[room]["volume"] = getSortedNamedStream(room,"volume")
        tsdata[room]["minflow"] = getSortedNamedStream(room, "minflow")
        
    except:
#         raise
        print "Error pulling in data..."
        if room in room_list:
            del room_list[room]
        if room in data:
            del tsdata[room]
        continue

print "Max number of readings : ", maxnumreadings
for room in results:
    if isDataMissing(room):
        if room in room_list:
            print "Deleting room : %s due to missing data. Number of datapoints : %s" %(room, str(room_list[room]["numreadings"]))
            print room_list[room]["numreadings"]
            del room_list[room]
        if room in tsdata:
            del tsdata[room]


# In[8]:

def removeSpikes(t, flow_values, vol_values, minflow_values, block = 4 , threshold = 1.13): 
    bw = block/2
    instants_to_delete = []
    for i in range(len(t)):
        if vol_values[i] < minflow_values[i]:
            instants_to_delete.append(i)
    
    for i in instants_to_delete:
        if i<bw:
            flow_values[i] = np.median(flow_values[0:i+bw])
        else:
            flow_values[i] = np.median(flow_values[i-bw:i+bw])
    
    flow_values = despike(flow_values, block, threshold)
    
    return flow_values

def despike(signal, block = 4 , threshold = 1.10):
    bw = block/2
    for i in range(len(signal)):
        if i<bw or i>=len(signal)-bw:
            continue
        if signal[i] > threshold * max(signal[i-bw:i-1]) and signal[i] > threshold * max(signal[i+1:i+bw]):
            signal[i] = np.median(signal[i-bw:i+bw])
    
    return signal

def getIndicativeSignalForZone(zone):
    t, xf  = tsdata[zone]['flow']
    _, xv  = tsdata[zone]['volume']
    _, xmf = tsdata[zone]['minflow']
    
    xf1 = removeSpikes(t, xf, xv, xmf)
    
    return t, xf1


# In[9]:

pylab.rcParams['figure.figsize'] = (30.0, 10.0)
def plotSpectrum(zone):
    t, fx = getIndicativeSignalForZone(zone)
    n = len(fx)
    
    Fk = fft.rfft(fx)/n # Fourier coefficients (divided by n)
    nu = fft.rfftfreq(n, period) # Natural frequencies

#     Fk = fft.fftshift(Fk) # Shift zero freq to center
#     nu = fft.fftshift(nu) # Shift zero freq to center
    
    plt.plot(nu,abs(Fk),'r') # plotting the spectrum
    
    plt.xlim(-0.000001,max(nu))
    plt.xlabel('Freq (Hz)', size = 'x-large')
    plt.ylabel('|Y(freq)|', size = 'x-large')
    
    plt.show()

def plotPowerSpectralDensity(zone):
    t, fx = getIndicativeSignalForZone(zone)
#     f, Pxx_den = periodogram(fx, 1.0/period)
    f, Pxx_den = welch(fx, 1.0/period)
    plt.semilogy(f, Pxx_den)
    plt.ylim([1e-4, 1e8])
    plt.xlim(0,max(f))
    plt.xticks(np.arange(0, max(f), 1.0/(24*3600)), rotation=90)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()
    
#     f, Pxx_spec = welch(fx, 1.0/period, 'flattop', scaling='spectrum')
#     plt.figure()
#     plt.semilogy(f, np.sqrt(Pxx_spec))
#     plt.xlim(0,max(f))
#     plt.xticks(np.arange(0, max(f), 1.0/(24*3600)), rotation=90)
#     plt.xlabel('frequency [Hz]')
#     plt.ylabel('Linear spectrum [V RMS]')
#     plt.show()
#     print np.sqrt(Pxx_spec.max())

def decompose(zone, pltFlag=True):   
    pylab.rcParams['figure.figsize'] = (30.0, 20.0)
    t, x = getIndicativeSignalForZone(zone)
    imfs = pyeemd.ceemdan(x)
    # imfs = pyeemd.eemd(x)

    if pltFlag:
        f, ax = plt.subplots(len(imfs)+1,1,sharex=True)
        ax[0].plot(x)
        ax[0].set_xlim(0, len(x))
        ax[0].set_ylabel('signal')
        for item in ([ax[0].title, ax[0].xaxis.label, ax[0].yaxis.label] + ax[0].get_xticklabels() + ax[0].get_yticklabels()):
                item.set_fontsize(16)
                
        for i in range(len(imfs)):
            ax[i+1].plot(imfs[i])
            ax[i+1].set_xlim(0, len(imfs[i]))
            numdays = len(imfs[i])/numperiods
            labels = day_labels*(numdays/7)+day_labels[0:(numdays%7)]
            if i!= len(imfs)-1:
                ax[i+1].set_ylabel('IMF'+str(i))
                for k in range(numdays/7):
                    ax[i+1].axvspan(k*7*numperiods+(5-firstday)*numperiods, (k+1)*7*numperiods-firstday*numperiods, facecolor='r', alpha=0.1)
            else:
                ax[i+1].set_ylabel('trend')
            ax[i+1].yaxis.set_label_coords(-0.015, 0.5) 
            ax[i+1].set_xticks(np.arange(0, len(imfs[i]), numperiods))
            ax[i+1].set_xticklabels(labels, rotation=90)
        plt.show()
    return imfs

def hilb(s, unwrap=False):
    H = hilbert(s)
    amp = np.abs(H)
    phase = np.angle(H)
    if unwrap: phase = np.unwrap(np.angle(H))
    return amp, phase

def computeInstFreq(imfs, pltFlag=False):
    pylab.rcParams['figure.figsize'] = (30.0, 20.0)
#     imfs = decompose(zone, False)
    pers = []
    
    if pltFlag:
        f, ax = plt.subplots(len(imfs)-1,1,sharex=True)
        
    for i in range(len(imfs)-1):
        signal = imfs[i]
        t = range(len(signal))

        A, phase = hilb(signal, unwrap=True)
        F = np.diff(phase) / ((2.0*np.pi) * period)
        
        if pltFlag:
            ax[i].plot(t[1:], F)
            ax[i].set_xlim(0.0, len(signal))
            numdays = len(signal)/numperiods
            labels = day_labels*(numdays/7)+day_labels[0:(numdays%7)]
            ax[i].set_xticks(np.arange(0, len(imfs[i]), numperiods))
            ax[i].set_xticklabels(labels, rotation=90)
        
#         per = np.average([1/(60*ff) for ff in F])
        per = 1/(np.average(F)*60)
        print per
        pers.append(per) # in minute

    if pltFlag:
        plt.show()
        
    return pers

def binIMFs(pers, debugFlag=True, th1=120, th2=960):
    binFast = []
    binMid = []
    binSlow = []
    
    i=0
    for per in pers:
        if per <= th1:
            binFast.append(i)
        elif per <= th2:
            binMid.append(i)
        else:
            binSlow.append(i)
        i+=1
        
    if debugFlag:
        print binFast
        print binMid
        print binSlow
    
    return binFast, binMid, binSlow
    
def aggregateIMFs(imfs, binFast, binMid, binSlow, pltFlag=True):
    pylab.rcParams['figure.figsize'] = (30.0, 10.0)
    if len(binFast)>0:
        fastFs = imfs[binFast[-1]]
        for idx in range(len(binFast)-1):
            fastFs = fastFs + imfs[binFast[idx]]
    else:
        fastFs = [0]*len(imfs[0])
            
    if len(binMid)>0:
        midFs = imfs[binMid[-1]]
        for idx in range(len(binMid)-1):
            midFs = midFs + imfs[binMid[idx]]
    else:
        midFs = [0]*len(imfs[0])
    
    if len(binSlow)>0:
        slowFs = imfs[binSlow[-1]]
        for idx in range(len(binSlow)-1):
            slowFs = slowFs + imfs[binSlow[idx]]
    else:
        slowFs = [0]*len(imfs[0])
    
    sig = imfs[-1]
    for i in range(len(imfs)-1):
        sig += imfs[i]
    
    if pltFlag:
        f, ax = plt.subplots(5,1,sharex=True)
        ax[0].plot(fastFs) # Plot Cosine terms
        ax[0].set_ylabel('ts < 2hr', size = 'x-large')
        ax[1].plot(midFs) # Plot Sine terms
        ax[1].set_ylabel('2hr < ts < 16hr', size = 'x-large')
        ax[2].plot(slowFs) # Plot spectral power
        ax[2].set_ylabel('ts > 16hr', size = 'x-large')
        ax[3].plot(fastFs+midFs) # Plot spectral power
        ax[3].set_ylabel('ts < 16hr', size = 'x-large')
        ax[4].plot(sig)
        ax[4].set_ylabel('signal', size = 'x-large')

        ax[4].set_xlim(0.0, len(imfs[0]))
        numdays = len(imfs[0])/numperiods
        labels = day_labels*(numdays/7)+day_labels[0:(numdays%7)]
        ax[4].set_xticks(np.arange(0, len(imfs[0]), numperiods))
        ax[4].set_xticklabels(labels, rotation=90)
        
        for axes in ax:
            for k in range(numdays/7):
                axes.axvspan(k*7*numperiods+(5-firstday)*numperiods, (k+1)*7*numperiods-firstday*numperiods, facecolor='r', alpha=0.1)
        
        plt.show()
    
    return fastFs, midFs, slowFs

def extractMidRangeFreqsForZone(zone, verbose=True):
#     if verbose:
#         plotSpectrum(zone)
#         plotPowerSpectralDensity(zone)
    imfs = decompose(zone, verbose)
    binFast, binMid, binSlow = binIMFs(computeInstFreq(imfs, False), verbose) 
    fastFs, midFs, slowFs = aggregateIMFs(imfs, binFast, binMid, binSlow, verbose)
    
    times = range(len(midFs))
    tsdata[zone]["indicative-midFreqs"] = (times, midFs)
    
def extractMidRangeFreqs():
    count = 1
    for zone in tsdata:
        print 'Doing zone: ', zone , "... (%d/%d)" % (count, len(tsdata))
        extractMidRangeFreqsForZone(zone, False)
        print 'Finished doing zone: '+str(zone)
        count += 1


# In[10]:

# extractMidRangeFreqsForZone('250 CLASSROOM')
# extractMidRangeFreqsForZone('COMP LAB 200')
# extractMidRangeFreqsForZone('621C OFFICE')
# extractMidRangeFreqsForZone('CYBERCAFE 210')
# extractMidRangeFreqsForZone('254 CLASSROOM')
# extractMidRangeFreqsForZone('730 CONFERENCE')
# extractMidRangeFreqsForZone('656 CUBICLES')

# conference_rooms = ['368 CONFERENCE' , '458 CONFERENCE' , '554 CONFERENCE' , '630 CONFERENCE' , '730 CONFERENCE']


# In[11]:

# def removeSpikes(t, flow_values, vol_values, minflow_values):   
#     instants_to_delete = []
#     for i in range(len(t)):
#         if vol_values[i] < minflow_values[i]:
#             instants_to_delete.append(i)
    
#     for i in instants_to_delete:
#         if i < 3:
#             flow_values[i] = np.median(flow_values[0:i+3])
#         else:
#             flow_values[i] = np.median(flow_values[i-3:i+3])

#     return flow_values

def convertOccupancyToBitmap(occupied_idx, t, numperiods):
    occupancy_bitmap = [0]*int(t[-1]+1)
    for idx in occupied_idx:
        occupancy_bitmap[idx] = 1

    weekly_occupancy_bitmap = [0]*(numperiods*7)

    for tidx in range(int(t[-1]+1)):
        weekly_occupancy_bitmap[tidx%(numperiods*7)] += occupancy_bitmap[tidx]

    return weekly_occupancy_bitmap

def applySchedule(unoccupied_idx, occupied_idx, t, numperiods, schedule_night):
    unoccupied_idx_schedule = list(unoccupied_idx[:])
    occupied_idx_schedule = list(occupied_idx[:])

    for tm in t:
        if (tm % numperiods) in schedule_night:
            if tm in occupied_idx:
                occupied_idx_schedule.remove(tm)
                unoccupied_idx_schedule.append(tm)
                
    unoccupied_idx_schedule = sorted(unoccupied_idx_schedule)
    
    return unoccupied_idx_schedule, occupied_idx_schedule

def calculateEnergyLoss(zone, schedule_night, useIMFs = False, plotFlag = True):    
    if zone not in tsdata:
        print "No zone in data dictionary"
        return None, None, None
    t, xf = tsdata[zone]['flow']
    _, xrv = tsdata[zone]['reheat']
    _, xv = tsdata[zone]['volume']
    _, xmf = tsdata[zone]['minflow']
    
    xf1 = removeSpikes(t, xf, xv, xmf)
    
    if useIMFs:
        t, xf_imfs = tsdata[zone]['indicative-midFreqs']
        occupied_intervals, occupied_idx, unoccupied_intervals, unoccupied_idx = runSegmentationForZone(t, xf_imfs, plotFlag, True, False)
    else:
        occupied_intervals, occupied_idx, unoccupied_intervals, unoccupied_idx = runSegmentationForZone(t, xf1, plotFlag, True, False)

#     unoccupied_idx_schedule, occupied_idx_schedule = applySchedule(unoccupied_idx, occupied_idx, t, numperiods, schedule_night)
    unoccupied_idx_schedule = unoccupied_idx
    occupied_idx_schedule = occupied_idx
    weekly_occupancy_bitmap = convertOccupancyToBitmap(occupied_idx_schedule, t, numperiods)

    totalenergyloss = 0
    count = 0
    
    energyloss = np.zeros(len(weekly_occupancy_bitmap))
    weekly_energy_loss = np.zeros(data_num_weeks)

    for tm in t:
        if xf[tm] > 0 and xrv[tm] > 0:
            energyloss[tm%(numperiods*7)] += xrv[tm] * windowsize
            totalenergyloss += xrv[tm] * windowsize
            weekly_energy_loss[int(tm / (numperiods * 7))] += (xrv[tm] * windowsize)

    room_list[zone]["defaultSch-loss"] = totalenergyloss
    room_list[zone]["defaultSch-loss-weekly"] = weekly_energy_loss
    
    print "Total energy loss in ", zone, " is ", sum(energyloss)
    print "totalenergyloss : ", totalenergyloss
    number_of_weeks = len(t)/(numperiods*7)
    avg_energy_loss[zone] = [ float(i) / number_of_weeks for i in energyloss ]
    avg_weekly_occupancy_bitmap[zone] = [ float(i) / number_of_weeks for i in weekly_occupancy_bitmap]

    return weekly_occupancy_bitmap, number_of_weeks, occupied_idx_schedule

def plotOccupancy(weekly_occupancy_bitmap, number_of_weeks, zone):
    plt.xlim(0, numperiods*7)
    plt.ylim(0, 1)
    plt.plot(range(numperiods*7), [a/float(number_of_weeks) for a in weekly_occupancy_bitmap])
    plt.xticks(np.arange(0, numperiods*7, numperiods), day_labels, rotation=90)
    plt.xlabel('Day of Week')
    plt.ylabel('Occupancy')
    plt.title('SutardjaDai ' + str(zone) + ' between '+str(startDate_dt_local)+' and '+str(endDate_dt_local))
    plt.show()
            
def calculateEnergyLossAllZones(plotFlag=False, useIMFs = False):
    print "Calculating energy loss for all zones"
    print "Number of zones : " , len(room_list)
    global schedule_nighttime
    
    count = 1
    for zone in room_list:
        try:
            print "Doing zone : ", zone , "(%d/%d)" % (count, len(room_list))
            count += 1
            
            # run Canny detector on occupancy indicative signal or midFreq IMF signal
            weekly_occupancy_bitmap, numweeks, occupied_idx_schedule_zone = calculateEnergyLoss(zone, schedule_nighttime, useIMFs, plotFlag)

            if "occ" not in room_list[zone]:
                room_list[zone]["occ"] = []
            room_list[zone]["occ"] = list(occupied_idx_schedule_zone)
            print "Energy loss is ", room_list[zone]["defaultSch-loss"]
            print "Calculated occupancy too"
        except:
            raise
            print "Error doing zone : ", zone


# In[ ]:

extractMidRangeFreqs()


# In[13]:

calculateEnergyLossAllZones(False, False)
# calculateEnergyLossAllZones(False, True)


# In[14]:

pylab.rcParams['figure.figsize'] = (20.0, 40.0)
def plotHeatMap(ifNormal=True, anomalous_zones=[]):
    
    global avg_energy_loss
    global avg_weekly_occupancy_bitmap
    
    anonymize = False
    
    if ifNormal:
        energylossvalues = np.zeros((len(avg_energy_loss)-len(anomalous_zones), numperiods * 7) )
        occvalues = np.zeros((len(avg_energy_loss)-len(anomalous_zones), numperiods * 7) )
    else:
        energylossvalues = np.zeros((len(anomalous_zones), numperiods * 7) )
        occvalues = np.zeros((len(anomalous_zones), numperiods * 7) )
    count = 0
    
    zone_names = []
    for zone in sorted(avg_energy_loss, key=lambda k: fsum(avg_energy_loss[k])):
        if ifNormal==True and zone in anomalous_zones:
            continue
        elif ifNormal==False and zone not in anomalous_zones:
            continue
        else:
            energylossvalues[count, :] = avg_energy_loss[zone][:]
            occvalues[count, :] = avg_weekly_occupancy_bitmap[zone][:]
            count += 1
            
            if anonymize:
                zone_names.append('zone '+str(count))
            else:
                zone_names.append(zone)
            
    row_labels = list(zone_names)
    col_labels = day_labels
    
#     fig, ax = plt.subplots()
#     heatmap = ax.pcolor(energylossvalues, cmap=plt.cm.Reds)
#     plt.colorbar(heatmap)
    
#     ax.set_xticks(np.arange(0,energylossvalues.shape[1],numperiods), minor=False)
#     ax.set_yticks(np.arange(energylossvalues.shape[1])+0.5, minor=False)
    
#     #ax.invert_yaxis()
#     ax.xaxis.tick_top()
    
#     ax.set_ybound(lower = 0, upper = energylossvalues.shape[0])
#     ax.set_xbound(lower = 0, upper = energylossvalues.shape[1])

#     ax.set_xticklabels(col_labels, minor=False)
#     ax.set_yticklabels(row_labels, minor=False)
#     plt.title("Average Energy Loss [SDH Hall]", y=1.05)
#     for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#         item.set_fontsize(20)
#     plt.show()
    
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(occvalues, cmap=plt.cm.YlOrRd)
    ax.grid(which='major', axis='x', linestyle='-')
    
    cbaxes = fig.add_axes([0.91, 0.125, 0.01, 0.775]) 
    plt.colorbar(heatmap, cax = cbaxes)  
#     plt.colorbar(heatmap)
    
    ax.set_xticks(np.arange(0,energylossvalues.shape[1],numperiods), minor=False)
    ax.set_yticks(np.arange(energylossvalues.shape[1])+0.5, minor=False)
    
    #ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    ax.set_ybound(lower = 0, upper = occvalues.shape[0])
    ax.set_xbound(lower = 0, upper = occvalues.shape[1])

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_title("Average Occupancy [SDH Hall]", y=1.01)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    plt.show()
    
plotHeatMap()


# In[108]:

pylab.rcParams['figure.figsize'] = (20.0, 10.0)

import matplotlib.ticker as ticker

def convert2WDWE(weekly_occ):
    outputWD = [0]*numperiods
    outputWE = [0]*numperiods
    
    for i in range(numperiods*7):
        if i/numperiods < 5:
            outputWD[i%numperiods] += weekly_occ[i]
        else:
            outputWE[i%numperiods] += weekly_occ[i]
            
    for i in range(numperiods):
        outputWD[i] = outputWD[i]/5
        outputWE[i] = outputWE[i]/2
    return outputWD, outputWE

def plotWDvsWE(ifNormal=True, anomalous_zones=[]):
    
    global avg_weekly_occupancy_bitmap
    
    anonymize = True
    
    if ifNormal:
        occvaluesWD = np.zeros((len(avg_weekly_occupancy_bitmap)-len(anomalous_zones), numperiods) )
        occvaluesWD1 = np.zeros((len(avg_weekly_occupancy_bitmap)-len(anomalous_zones), numperiods) )
        occvaluesWE = np.zeros((len(avg_weekly_occupancy_bitmap)-len(anomalous_zones), numperiods) )
        occvaluesWE1 = np.zeros((len(avg_weekly_occupancy_bitmap)-len(anomalous_zones), numperiods) )
    else:
        occvaluesWD = np.zeros((len(anomalous_zones), numperiods) )
        occvaluesWD1 = np.zeros((len(anomalous_zones), numperiods) )
        occvaluesWE = np.zeros((len(anomalous_zones), numperiods) )
        occvaluesWE1 = np.zeros((len(anomalous_zones), numperiods) )
        
    count = 0
    
    zone_names = []
    for zone in sorted(avg_energy_loss, key=lambda k: fsum(avg_energy_loss[k])):
        if ifNormal==True and zone in anomalous_zones:
            continue
        elif ifNormal==False and zone not in anomalous_zones:
            continue
        else:
            occvaluesWD[count, :], occvaluesWE[count, :] = convert2WDWE(avg_weekly_occupancy_bitmap[zone][:])
            count += 1
            
            if anonymize:
                zone_names.append('zone '+str(count))
            else:
                zone_names.append(zone)
    
    startsWD = [0]*(count)
    startsWD1 = [0]*(count)
    endsWD = [0]*(count)
    endsWD1 = [0]*(count)
    lenWD = [0]*(count)
    lenWD1 = [0]*(count)
    
    startsWE = [0]*(count)
    startsWE1 = [0]*(count)
    endsWE = [0]*(count)
    endsWE1 = [0]*(count)
    lenWE = [0]*(count)
    lenWE1 = [0]*(count)
    
    percentile = .1 # 10th percentile
    
    # WEEKDAY
    
    for i in range(count):
        ecdf = np.cumsum(occvaluesWD[i,:])
        pct_s = np.sum(occvaluesWD[i,:])*percentile
        pct_e = np.sum(occvaluesWD[i,:])*(1-percentile)
        for j in range(len(ecdf)):
            if j == 0:
                continue
            if ecdf[j]>=pct_s and ecdf[j-1]<pct_s:
                startsWD[i] = j
            if ecdf[j]>=pct_e and ecdf[j-1]<pct_e:
                endsWD[i] = j
        lenWD[i]=endsWD[i]-startsWD[i]

    indexes = range(count)
    zone_namesWD = []
    indexes.sort(key=lambda k: lenWD[k], reverse=True)
    for i in range(count):
        lenWD1[i] = lenWD[indexes[i]]
    bins = [0]
    multiplier = max(lenWD1)/50
    for i in range(count-1):
        if lenWD1[i]>0 and lenWD1[i+1]==0:
            bins.append(i)
            break
        if lenWD1[i]>=multiplier*50 and lenWD1[i+1]<multiplier*50:
            bins.append(i)
            multiplier -= 1
    bins = [0,-1]
    for i in range(len(bins)-1):
        if bins[i+1] == -1:
            indexes[bins[i]:] = sorted(indexes[bins[i]:], key=lambda k: startsWD[k], reverse=False)
        else:
            indexes[bins[i]: bins[i+1]] = sorted(indexes[bins[i]: bins[i+1]], key=lambda k: startsWD[k], reverse=False)
    for i in range(count):
        occvaluesWD1[i,:] = occvaluesWD[indexes[i],:]
        startsWD1[i] = startsWD[indexes[i]]
        endsWD1[i]   = endsWD[indexes[i]]
        zone_namesWD.append( zone_names[indexes[i]] )
    
    for i in range(count):
        if startsWD1[i]==0 and endsWD1[i]==numperiods-1:
            occvaluesWD1[i,:]=0
            continue
        for j in range(numperiods):
            if j<startsWD1[i] or j>=endsWD1[i]:
                occvaluesWD1[i,j]=0
                
    # WEEKEND
    
    for i in range(count):
        ecdf = np.cumsum(occvaluesWE[i, :])
        pct_s = np.sum(occvaluesWE[i, :])*percentile
        pct_e = np.sum(occvaluesWE[i, :])*(1-percentile)
        for j in range(len(ecdf)):
            if j == 0:
                continue
            if ecdf[j]>=pct_s and ecdf[j-1]<pct_s:
                startsWE[i] = j
            if ecdf[j]>=pct_e and ecdf[j-1]<pct_e:
                endsWE[i] = j
        lenWE[i]=endsWE[i]-startsWE[i]
                
    indexes = range(count)
    zone_namesWE = []
    indexes.sort(key=lambda k: lenWE[k], reverse=True)
    for i in range(count):
        lenWE1[i] = lenWE[indexes[i]]
    bins = [0]
    multiplier = max(lenWE1)/50
    for i in range(count-1):
        if lenWE1[i]>0 and lenWE1[i+1]==0:
            bins.append(i)
            break
        if lenWE1[i]>=multiplier*50 and lenWE1[i+1]<multiplier*50:
            bins.append(i)
            multiplier -= 1
    bins = [0,-26]
    for i in range(len(bins)-1):
        if bins[i+1] == -1:
            indexes[bins[i]:] = sorted(indexes[bins[i]:], key=lambda k: startsWE[k], reverse=False)
        else:
            indexes[bins[i]: bins[i+1]] = sorted(indexes[bins[i]: bins[i+1]], key=lambda k: startsWE[k], reverse=False)
    for i in range(count):
        occvaluesWE1[i,:] = occvaluesWE[indexes[i],:]
        startsWE1[i] = startsWE[indexes[i]]
        endsWE1[i]   = endsWE[indexes[i]]
        zone_namesWE.append( zone_names[indexes[i]] )
    
    for i in range(count):
        if startsWE1[i]==0 and endsWE1[i]==numperiods-1:
            occvaluesWE1[i,:]=0
            continue
        for j in range(numperiods):
            if j<startsWE1[i] or j>=endsWE1[i]:
                occvaluesWE1[i,j]=0            
    
    
    col_labels = ["12am", "4am", "8am", "12pm", "4pm", "8pm"]
    
    # PLOT WEEKDAY
    
    if anonymize:
        row_labels = []
        
    else:
        row_labels = list(zone_namesWD)

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(occvaluesWD1, cmap=plt.cm.Greys)
    ax.grid(which='minor', axis='x', linestyle='-')
    
    for i in range(count):
        plt.axvline(x=startsWD1[i], ymin=i/float(count), ymax=float(i+1)/count, color='k')
        plt.axvline(x=endsWD1[i], ymin=i/float(count), ymax=float(i+1)/count, color='k')

    plt.axvline(x=np.mean(filter(None,startsWD1)), ymin=0, ymax=occvaluesWD1.shape[0], color='r', linewidth=2, linestyle='dashed')
    plt.axvline(x=np.mean(filter(None,endsWD1)), ymin=0, ymax=occvaluesWD1.shape[0], color='r', linewidth=2, linestyle='dashed')
        
    cbaxes = fig.add_axes([0.91, 0.125, 0.01, 0.775]) 
    plt.colorbar(heatmap, cax = cbaxes)
    
    ax.set_yticks(np.arange(occvaluesWD1.shape[0])+0.5, minor=False)
#     ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    ax.set_xticks(np.arange(0, occvaluesWE1.shape[1], occvaluesWD1.shape[1]/6))
    
    ax.set_ybound(lower = 0, upper = occvaluesWD1.shape[0])
    ax.set_xbound(lower = 0, upper = occvaluesWD1.shape[1])

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_title("Weekday Occupancy", y=1.005)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(26)
    plt.show()
    
    # PLOT WEEKEND
    
    if anonymize:
        row_labels = []
        
    else:
        row_labels = list(zone_namesWE)
        
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(occvaluesWE1, cmap=plt.cm.Greys)
    ax.grid(which='minor', axis='x', linestyle='-')
    
    for i in range(count):
        plt.axvline(x=startsWE1[i], ymin=i/float(count), ymax=float(i+1)/count, color='k')
        plt.axvline(x=endsWE1[i], ymin=i/float(count), ymax=float(i+1)/count, color='k')

    plt.axvline(x=np.mean(filter(None,startsWE1)), ymin=0, ymax=occvaluesWE1.shape[0], color='r', linewidth=2, linestyle='dashed')
    plt.axvline(x=np.mean(filter(None,endsWE1)), ymin=0, ymax=occvaluesWE1.shape[0], color='r', linewidth=2, linestyle='dashed')
    
    cbaxes = fig.add_axes([0.91, 0.125, 0.01, 0.775]) 
    plt.colorbar(heatmap, cax = cbaxes)
    
    ax.set_yticks(np.arange(occvaluesWE1.shape[0])+0.5, minor=False)
#     ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    ax.set_xticks(np.arange(0, occvaluesWE1.shape[1], occvaluesWE1.shape[1]/6))
    
    ax.set_ybound(lower = 0, upper = occvaluesWE1.shape[0])
    ax.set_xbound(lower = 0, upper = occvaluesWE1.shape[1])
    
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_title("Weekend Occupancy", y=1.005)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(26)
    plt.show()

plotWDvsWE()


# In[37]:

pylab.rcParams['figure.figsize'] = (10.0, 10.0)
def identifyNormalZones(occvalues,threshold,minlen=8,maxlen=56):
    for day in range(5):
        occ_intervals = []
        for idx in range(numperiods):
#             if idx < 24 or idx >= 92:
#                 continue
            if occvalues[day*numperiods+idx]>threshold:
                occ_intervals.append(idx)
        if len(occ_intervals)>=maxlen or len(occ_intervals)<=minlen:
            return False
    return True

def identifyZonesWithNightTimeOccupancy(occvalues,threshold,maxnum=12):
    for day in range(7):
        occ_night_intervals = []
        for idx in range(numperiods):
            if (idx < 24 or idx >= 92) and occvalues[day*numperiods+idx]>threshold:
                occ_night_intervals.append(idx)
        if len(occ_night_intervals)>=maxnum:
            return True
    return False

def identifyAnomalousZones(occvalues,threshold,minlen=8,maxlen=81):
    for day in range(5):
        occ_intervals = []
        for idx in range(numperiods):
            if occvalues[day*numperiods+idx]>threshold:
                occ_intervals.append(idx)
        if len(occ_intervals)>=maxlen or len(occ_intervals)<=minlen:
            return True
    return False

def getLengthofOccupiedInterval(zone, timerange, threshold):
    occ_intervals = []
    for idx in range(numperiods*7):
        if idx in timerange and avg_weekly_occupancy_bitmap[zone][idx]>=threshold:
            occ_intervals.append(idx)
    return occ_intervals

remaining_zones = []
normal_zones = []
anomalous_zones = []

always_occupied = []
always_unoccupied = []
for zone in sorted(room_list):
    count_occ = 0
    count_unocc = 0
    try:
        for i in range(len(avg_weekly_occupancy_bitmap[zone])):
            if avg_weekly_occupancy_bitmap[zone][i] >= 0.75*max(avg_weekly_occupancy_bitmap[zone]):
                count_occ += 1
            if avg_weekly_occupancy_bitmap[zone][i] <= 1.25*min(avg_weekly_occupancy_bitmap[zone]):
                count_unocc += 1
                
        if count_occ > 0.75 * len(avg_weekly_occupancy_bitmap[zone]):
            always_occupied.append(zone)
        elif count_unocc > 0.75 * len(avg_weekly_occupancy_bitmap[zone]):
            always_unoccupied.append(zone)
    except:
        continue

for zone in sorted(room_list):
    if zone in always_occupied or zone in always_unoccupied:
        continue
    else:
        remaining_zones.append(zone)

############

MIDNIGHT_TO_SIX = range(numperiods/4)
MIDNIGHT_TO_SIX_7DAYS = []
for day in range(7):
    for i in range(len(MIDNIGHT_TO_SIX)):
        MIDNIGHT_TO_SIX_7DAYS.append(day*numperiods+MIDNIGHT_TO_SIX[i])

# _7DAYS = range(numperiods*7)
_7DAYS = MIDNIGHT_TO_SIX_7DAYS

maxratio = 0
for zone in sorted(remaining_zones):
    occ_overnight = [avg_weekly_occupancy_bitmap[zone][tt] for tt in _7DAYS]
    ratio = max(occ_overnight)/np.mean(occ_overnight)
    if ratio > maxratio:
        maxratio = ratio

# feature1 = [fsum([avg_weekly_occupancy_bitmap[zone][tt] for tt in _7DAYS])/float(len(_7DAYS)) for zone in sorted(remaining_zones)]
feature1 = [fsum([avg_weekly_occupancy_bitmap[zone][tt] for tt in _7DAYS]) for zone in sorted(remaining_zones)]
feature2 = [max([avg_weekly_occupancy_bitmap[zone][tt] for tt in _7DAYS])/(np.mean([avg_weekly_occupancy_bitmap[zone][tt] for tt in _7DAYS])*maxratio) for zone in sorted(remaining_zones)]
feature2 = [1 if isnan(f) else f for f in feature2]
# feature3 = [len(getLengthofOccupiedInterval(zone, _7DAYS, 0.05))/float(len(_7DAYS)) for zone in sorted(remaining_zones)]

# observations with 2 features
X = []
for i in range(len(remaining_zones)):
#     X.append([feature1[i],feature2[i],feature3[i]])
    X.append([feature1[i],feature2[i]])
X = np.array(X)

# generate the linkage matrix
# Z = linkage(X, 'ward')
Z = linkage(X, 'complete')
# Z = linkage(X, 'single')

c, coph_dists = cophenet(Z, pdist(X))

# force two clusters
k=2
clusters = fcluster(Z, k, criterion='maxclust')
# print clusters

if True:
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')

#     ax.scatter(X[:,0], X[:,1], X[:,2], zdir='z', depthshade=True, c=clusters, cmap='prism')

#     ax.set_xlabel('feature 1')
#     ax.set_ylabel('feature 2')
#     ax.set_zlabel('feature 3')
    
    plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism') 
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.show()

to_delete = []
index = 0
for zone in sorted(remaining_zones):
    if clusters[index]==2:
        normal_zones.append(zone)
        to_delete.append(zone)
    elif clusters[index]==1:
        anomalous_zones.append(zone)
        to_delete.append(zone)
    index+=1

for zone in to_delete:
    remaining_zones.remove(zone)

# print normal_zones
# print remaining_zones
    
############
  
anomalous_zones.extend(remaining_zones) 

pylab.rcParams['figure.figsize'] = (20.0, 40.0)
plotHeatMap(False, normal_zones)
pylab.rcParams['figure.figsize'] = (20.0, 5.0)
plotHeatMap(False, anomalous_zones)
pylab.rcParams['figure.figsize'] = (20.0, 5.0)
plotHeatMap(False, always_occupied)
pylab.rcParams['figure.figsize'] = (20.0, 10.0)
plotHeatMap(False, always_unoccupied)

print "Num normal zones" , len(normal_zones)

print "Always occupied zones (anomolous)", len(always_occupied)

print "Never occupied (anomolous)", len(always_unoccupied)

print "Remaining anomalous zones", len(anomalous_zones)

print "Number of remaining zones", len(remaining_zones)

normal_zones.extend(always_unoccupied)

print "Percentage of normal zones", len(normal_zones)/float(len(room_list))


# In[22]:

# class Schedule(object):

#     def __init__(self, schedule_name):
#         self.name = schedule_name
#         self.schedule = [] # the nighttime schedule
#         print "Created schedule : %s" % (schedule_name)
#         self.zone = None
#         self.t = None
#         self.xf = None
#         self.xrv = None
        
#     def setZone(self, zone):
#         self.zone = zone
#         self.t, self.xf = getSortedNamedStream(self.zone, "flow")
#         _, self.xrv = getSortedNamedStream(self.zone, "reheat")
#         self.schedule = []

#     def computeSchedule(self):
#         self.applySchedule()

#     def applySchedule(self):
#         room_list[self.zone][self.name] = self.schedule        
        
#     def calculateEnergyLossForSchedule(self, zone):
#         self.setZone(zone)
#         self.computeSchedule()
#         energyloss = 0
#         violations = 0
#         weekly_loss = np.zeros(data_num_weeks)
#         for tm in self.t:
#             if self.xf[tm] > 0 and self.xrv[tm] > 0 and tm in self.schedule:
#                 energyloss += self.xrv[tm] * windowsize
#                 weekly_loss[int(tm / (numperiods * 7))] += (self.xrv[tm] * windowsize)
#             if tm in self.schedule and tm in room_list[zone]["occ"]:
#                 violations += 1
            

#         room_list[zone][self.name+"-loss"] = energyloss
#         room_list[zone][self.name+"-loss-weekly"] = weekly_loss

#         room_list[zone][self.name+"-violations"] = float(violations)
#         print "Calculated %s & %s for zone : %s " % (self.name+"-loss", self.name+"-violations", zone )

#     def __str__(self):
#         return ", ".join('{}'.format(*k) for k in enumerate(self.schedule))
    
# class BestSchedule(Schedule):
#     def computeSchedule(self):
#         self.schedule = []
#         occupied_idx_foralldays = room_list[self.zone]["occ"]
#         for tm in self.t:
#             if tm not in occupied_idx_foralldays:
#                 self.schedule.append(tm)
#         self.applySchedule()
        
# class NoSchedule(Schedule):
#     def computeSchedule(self):
#         self.schedule = []
#         self.applySchedule()

# class LoiterTimeSchedule(Schedule):
#     def setAggressiveness(self, percentile):
#         self.aggressiveness = percentile
        
#     def numWeeksToTrain(self, numWeeks):
#         self.numWeeksToTrain = numWeeks
    
#     def sameScheduleForAllDays(self, flag=True):
#         if flag == True:
#             self.sameScheduleForAllDays = True
#         else:
#             self.sameScheduleForAllDays = False
        
#     def getOccupiedSchedule(self, weeks, dayNum=None):
#         starts = []
#         ends = []
#         prev_idx = -2

#         # your function to find the occupancy start and end times
#         for idx in room_list[self.zone]["occ"]:
#             # only use data which is in the current set of weeks for training
#             if idx / (numperiods * 7) not in weeks:
#                 continue
#             if prev_idx!=idx-1:
#                 starts.append(idx)
#                 if prev_idx>=0:
#                     ends.append(prev_idx)
#             prev_idx = idx
        
#         ends.append(prev_idx)
        
#         if dayNum == None:
#             # find the modulo, to convert this to a generic day
#             final_starts = [ i % numperiods for i in starts ]
#             final_ends = [ i % numperiods for i in ends ]
            
#             # this is the case where you have the same schedule for all days in the week
# #             print "Non weekend start times : ", final_non_weekend_starts
# #             print "Non weekend end time : ", final_non_weekend_ends
#             if len(final_starts) == 0 or len(final_ends) == 0:
#                 return [ (numperiods, numperiods) ]
#             start_time = np.nanpercentile(final_starts,self.aggressiveness)
#             end_time = np.nanpercentile(final_ends,100-self.aggressiveness)+1
#             if end_time > numperiods:
#                 end_time = numperiods - 0.01
# #             print "Chosen start time : %d , end time : %d " % (start_time, end_time)
#             #occupied_idx_foraday = np.arange(start_time, end_time,1)
#             if end_time < start_time:
#                 return [ (None, None) ]
#             return [ (start_time, end_time) ]
#         else:
#             # find the module , to convert to generic week !
#             final_starts = [ i % (numperiods * 7) for i in starts ]
#             final_ends = [ i % (numperiods * 7) for i in ends ]
#             # this is the case where you are calculating a different schedule for each day of the week
#             occupied_schedule = []
# #             print "Non weekend start times : ", final_non_weekend_starts
# #             print "Non weekend end time : ", final_non_weekend_ends
#             for day in range(dayNum):
#                 starts_day = [ (idx % numperiods) for idx in final_starts if idx >= numperiods * (day) and idx < numperiods * (day + 1)]
#                 ends_day = [ (idx % numperiods) for idx in final_ends if idx >= numperiods * (day) and idx < numperiods * (day + 1)]
#                 if len(starts_day) == 0 or len(ends_day) == 0:
# #                     print "Day : %d , no start_time or end_time" % day, starts_day , ends_day
#                     occupied_schedule.append((numperiods, numperiods))
#                     continue
# #                 print "Day : ", day, "Start times : ", starts_day
# #                 print "Day : ", day, "End times : ", ends_day
#                 start_time = np.nanpercentile(starts_day,self.aggressiveness)
#                 end_time = np.nanpercentile(ends_day ,100-self.aggressiveness) + 1
#                 if end_time > numperiods:
#                     end_time = numperiods - 0.01
#                 if end_time < start_time:
#                     occupied_schedule.append((None, None))
#                     continue
#                 #occupied = np.arange(start_time, end_time ,1)
# #                 print "Day : %d , Chosen start time : %d , end time : %d " % (day, start_time, end_time)
#                 occupied_schedule.append( (start_time, end_time))
                
#             return occupied_schedule
    
#     def calculateEnergyLossForSchedule(self, zone):
#         self.setZone(zone)
#         self.computeSchedule()
#         energyloss = 0
#         violations = 0
#         weekend_violations = 0
#         nighttime_violations = 0
#         othertime_violations = 0
        
#         tot_num_weeks = int(self.t[-1] / ( numperiods * 7 ))
#         if (numperiods * 7 * tot_num_weeks) < self.t[-1]:
#             tot_num_weeks += 1
            
#         num_start_violations = 0
#         num_end_violations = 0
        
#         totdays = 0
#         reheat_key = self.name+"-"+str(self.numWeeksToTrain)+"-"+str(self.aggressiveness) + "-reheat"
#         room_list[zone][reheat_key] = np.zeros(tot_num_weeks * 7 * numperiods)
#         for tm in range(tot_num_weeks * 7 * numperiods):
#             if self.xrv[tm] > 0:
#                 room_list[zone][reheat_key][tm] = 1
        
#         for day in range((tot_num_weeks*7)):

#             totdays += 1
#             if self.schedule[day][0] == None or self.schedule[day][1] == None:
#                 continue
            
#             schedule_start = self.schedule[day][0] + (numperiods * day)
#             schedule_end = self.schedule[day][1] + (numperiods * day)
            
#             occ_day = [ tm  for tm in room_list[zone]["occ"] if tm >= day * numperiods and tm < (day+1)* numperiods ]
#             if len(occ_day) > 0:
#                 occ_start = occ_day[0]
#                 occ_end = occ_day[-1]
# #                 print "Zone: %s Occupancy : (%d, %d, %d, %d)" % (self.zone, occ_start, occ_end, occ_start % numperiods, occ_end % numperiods)
# #                 print "Zone: %s Schedule  : (%d, %d, %d, %d)" % (self.zone, schedule_start, schedule_end, schedule_start % numperiods , schedule_end % numperiods)
#                 for tm in range( day * numperiods , (day+1) * numperiods):
#                     if tm < min(occ_start, schedule_start):
#                         if self.xf[tm] > 0 and self.xrv[tm] > 0:
#                             energyloss += self.xrv[tm] * windowsize
#                             room_list[zone][reheat_key][tm] = 0
                            
#                     elif tm > max(occ_end, schedule_end):
#                             energyloss += self.xrv[tm] * windowsize
#                             room_list[zone][reheat_key][tm] = 0
                
#                 if occ_start < schedule_start:
#                     num_start_violations += int(schedule_start - occ_start)
# #                     print "Start violation" , int(schedule_start - occ_start)

#                 if occ_end > schedule_end:          
#                     num_end_violations += int(occ_end - schedule_end)
# #                     print "End violation" , int(occ_end - schedule_end)
#             else:
#                 for tm in range( day * numperiods , (day+1) * numperiods):
#                     if tm < schedule_start:
#                         if self.xf[tm] > 0 and self.xrv[tm] > 0:
#                             energyloss += self.xrv[tm] * windowsize
#                             room_list[zone][reheat_key][tm] = 0
#                     elif tm > schedule_end:
#                             energyloss += self.xrv[tm] * windowsize
#                             room_list[zone][reheat_key][tm] = 0
        
#         print " Start violations : %d , End violations : %d " % (num_start_violations, num_end_violations)
#         tot_violations = num_start_violations + num_end_violations
#         room_list[zone][self.name+"-loss"] = energyloss
#         room_list[zone][self.name+"-violations"] = tot_violations
#         room_list[zone][self.name+"-violations-start"] = float(num_start_violations) 
#         room_list[zone][self.name+"-violations-end"] = float(num_end_violations) 
#         room_list[zone][self.name+"-violations-totdays"] = totdays
#         #print room_list[zone][reheat_key]

#     def computeSchedule(self):
#         self.schedule = []

#         # calculating total number of weeks in the sensor data
#         tot_num_weeks = int(self.t[-1] / ( numperiods * 7 ))
#         if (numperiods * 7 * tot_num_weeks) < self.t[-1]:
#             tot_num_weeks += 1

            
#         # if total number of weeks < the numbers of weeks you have asked it to train for, apply the default schedule
#         if tot_num_weeks < self.numWeeksToTrain:
# #             print "Warning: the numbers of weeks you have asked it to train for is greater than the number of weeks for which data is available"
#             self.applySchedule()
#             return
        
       
#         for i in range(self.numWeeksToTrain * 7):
#             self.schedule.append((None, None))
        
#         if self.sameScheduleForAllDays:
#             print "Same schedule for all days"
# #             Case where you impose the same schedule for all the days in a week      
#             for i in range(tot_num_weeks - self.numWeeksToTrain ):
#                 weeks = np.arange(i, i + self.numWeeksToTrain)
# #                 print "Training for weeks : ", weeks
#                 occupied_schedule = self.getOccupiedSchedule(weeks)
# #                 print "Returned occupied schedule : ", occupied_schedule
#                 for day in range(7):
#                     self.schedule.append(occupied_schedule[0])
# #             print "Final schedule : ", self.schedule
#             self.applySchedule()
#             return
#         else:
#             # Case where there are different schedules on each day of the week
#             print "Different schedule for each day"
#             for i in range(tot_num_weeks - self.numWeeksToTrain ):
#                 weeks = np.arange(i, i + self.numWeeksToTrain)
# #                 print "Training for weeks : ", weeks

#                 # occupied schedule returns an array of occupancy arrays, one corresponding to each day
#                 occupied_schedule = self.getOccupiedSchedule(weeks, 7)
                     
# #                 print "Returned occupied schedule : ", occupied_schedule
#                 for day in range(7):
#                     self.schedule.append(occupied_schedule[day])
                    
# #             print "Final schedule : ", self.schedule
#             self.applySchedule()
#             return


# In[23]:

class Schedule(object):

    def __init__(self, schedule_name):
        self.name = schedule_name
        self.schedule = [] # the nighttime schedule
        print "Created schedule : %s" % (schedule_name)
        self.zone = None
        self.t = None
        self.xf = None
        self.xrv = None
        self.aggressiveness = 0
        self.numWeeksToTrain = 0
        
    def setZone(self, zone):
        self.zone = zone
        self.t, self.xf = getSortedNamedStream(self.zone, "flow")
        _, self.xrv = getSortedNamedStream(self.zone, "reheat")
        self.schedule = []

    def computeSchedule(self):
        self.applySchedule()

    def applySchedule(self):
        room_list[self.zone][self.name] = self.schedule        
        
    def calculateEnergyLossForSchedule(self, zone):
        self.setZone(zone)
        self.computeSchedule()
        energyloss = 0
        violations = 0
        weekly_loss = np.zeros(data_num_weeks)
        for tm in self.t:
            if self.xf[tm] > 0 and self.xrv[tm] > 0 and tm in self.schedule:
                energyloss += self.xrv[tm] * windowsize
                weekly_loss[int(tm / (numperiods * 7))] += (self.xrv[tm] * windowsize)
            if tm in self.schedule and tm in room_list[zone]["occ"]:
                violations += 1
            

        room_list[zone][self.name+"-loss"] = energyloss
        room_list[zone][self.name+"-loss-weekly"] = weekly_loss

        room_list[zone][self.name+"-violations"] = float(violations)
        print "Calculated %s & %s for zone : %s " % (self.name+"-loss", self.name+"-violations", zone )

    def __str__(self):
        return ", ".join('{}'.format(*k) for k in enumerate(self.schedule))
    
class BestSchedule(Schedule):
    def computeSchedule(self):
        self.schedule = []
        occupied_idx_foralldays = room_list[self.zone]["occ"]
        for tm in self.t:
            if tm not in occupied_idx_foralldays:
                self.schedule.append(tm)
        self.applySchedule()
        
class NoSchedule(Schedule):
    def computeSchedule(self):
        self.schedule = []
        self.applySchedule()

class LoiterTimeSchedule(Schedule):
    def setAggressiveness(self, percentile):
        self.aggressiveness = percentile
        
    def numWeeksToTrain(self, numWeeks):
        self.numWeeksToTrain = numWeeks
    
    def sameScheduleForAllDays(self, flag=True):
        if flag == True:
            self.sameScheduleForAllDays = True
        else:
            self.sameScheduleForAllDays = False
        
    def getOccupiedSchedule(self, weeks, dayNum=None):
        starts = []
        ends = []
        prev_idx = -2

        # your function to find the occupancy start and end times
        for idx in room_list[self.zone]["occ"]:
            # only use data which is in the current set of weeks for training
            if idx / (numperiods * 7) not in weeks:
                continue
            if prev_idx!=idx-1:
                starts.append(idx)
                if prev_idx>=0:
                    ends.append(prev_idx)
            prev_idx = idx
        
        ends.append(prev_idx)

        if dayNum == None:
            # find the modulo, to convert this to a generic day
            final_starts = [ i % numperiods for i in starts ]
            final_ends = [ i % numperiods for i in ends ]
            
            # this is the case where you have the same schedule for all days in the week
#             print "Non weekend start times : ", final_non_weekend_starts
#             print "Non weekend end time : ", final_non_weekend_ends
            if len(final_starts) == 0 or len(final_ends) == 0:
                return [ (numperiods, numperiods) ]
            start_time = np.nanpercentile(final_starts,self.aggressiveness)
            end_time = np.nanpercentile(final_ends,100-self.aggressiveness)+1
            if end_time > numperiods:
                end_time = numperiods - 0.01
#             print "Chosen start time : %d , end time : %d " % (start_time, end_time)
            #occupied_idx_foraday = np.arange(start_time, end_time,1)
            if end_time < start_time:
                return [ (None, None) ]
            return [ (start_time, end_time) ]
        else:
            # find the module , to convert to generic week !
            final_starts = [ i % (numperiods * 7) for i in starts ]
            final_ends = [ i % (numperiods * 7) for i in ends ]
            # this is the case where you are calculating a different schedule for each day of the week
            occupied_schedule = []
#             print "Non weekend start times : ", final_non_weekend_starts
#             print "Non weekend end time : ", final_non_weekend_ends
            for day in range(dayNum):
                starts_day = [ (idx % numperiods) for idx in final_starts if idx >= numperiods * (day) and idx < numperiods * (day + 1)]
                ends_day = [ (idx % numperiods) for idx in final_ends if idx >= numperiods * (day) and idx < numperiods * (day + 1)]
                if len(starts_day) == 0 or len(ends_day) == 0:
#                     print "Day : %d , no start_time or end_time" % day, starts_day , ends_day
                    occupied_schedule.append((numperiods, numperiods))
                    continue
#                 print "Day : ", day, "Start times : ", starts_day
#                 print "Day : ", day, "End times : ", ends_day
                start_time = np.nanpercentile(starts_day,self.aggressiveness)
                end_time = np.nanpercentile(ends_day ,100-self.aggressiveness) + 1
                if end_time > numperiods:
                    end_time = numperiods - 0.01
                if end_time < start_time:
                    occupied_schedule.append((None, None))
                    continue
                #occupied = np.arange(start_time, end_time ,1)
#                 print "Day : %d , Chosen start time : %d , end time : %d " % (day, start_time, end_time)
                occupied_schedule.append( (start_time, end_time))
                
            return occupied_schedule
    
    def calculateEnergyLossForSchedule(self, zone):
        self.setZone(zone)
        self.computeSchedule()
        energyloss = 0
        violations = 0
        weekend_violations = 0
        nighttime_violations = 0
        othertime_violations = 0
        
        tot_num_weeks = int(self.t[-1] / ( numperiods * 7 ))
        if (numperiods * 7 * tot_num_weeks) < self.t[-1]:
            tot_num_weeks += 1
            
        num_start_violations = 0
        num_end_violations = 0
        
        totdays = 0
        reheat_key = self.name+"-"+str(self.numWeeksToTrain)+"-"+str(self.aggressiveness) + "-reheat"
#         energy_savings_key = self.name+"-"+str(self.numWeeksToTrain)+"-"+str(self.aggressiveness) + "-energysavings"
        violations_key = self.name+"-"+str(self.numWeeksToTrain)+"-"+str(self.aggressiveness) + "-violations"
        
        
        room_list[zone][reheat_key] = np.zeros(tot_num_weeks * 7 * numperiods) 
        room_list[zone][violations_key] = np.zeros(tot_num_weeks * 7 * numperiods)
        
        for tm in range(tot_num_weeks * 7 * numperiods):
            if self.xrv[tm] > 0:
                room_list[zone][reheat_key][tm] = 1
#                 room_list[zone][energy_savings_key][tm] = 1
                
        for day in range((tot_num_weeks*7)):

            totdays += 1
            if self.schedule[day][0] == None or self.schedule[day][1] == None:
                continue
            
            schedule_start = self.schedule[day][0] + (numperiods * day)
            schedule_end = self.schedule[day][1] + (numperiods * day)
            
            occ_day = [ tm  for tm in room_list[zone]["occ"] if tm >= day * numperiods and tm < (day+1)* numperiods ]
            if len(occ_day) > 0:
                occ_start = occ_day[0]
                occ_end = occ_day[-1]
#                 print "Zone: %s Occupancy : (%d, %d, %d, %d)" % (self.zone, occ_start, occ_end, occ_start % numperiods, occ_end % numperiods)
#                 print "Zone: %s Schedule  : (%d, %d, %d, %d)" % (self.zone, schedule_start, schedule_end, schedule_start % numperiods , schedule_end % numperiods)
                for tm in range( day * numperiods , (day+1) * numperiods):
                    if tm < min(occ_start, schedule_start):
                        if self.xf[tm] > 0 and self.xrv[tm] > 0:
                            energyloss += self.xrv[tm] * windowsize
                            room_list[zone][reheat_key][tm] = 0
                    elif tm > max(occ_end, schedule_end):
                            energyloss += self.xrv[tm] * windowsize
                            room_list[zone][reheat_key][tm] = 0
                
                if occ_start < schedule_start:
                    num_start_violations += int(schedule_start - occ_start)
                    room_list[zone][violations_key][occ_start:schedule_start] = 1
#                     print "Start violation" , int(schedule_start - occ_start)

                if occ_end > schedule_end:          
                    num_end_violations += int(occ_end - schedule_end)
                    room_list[zone][violations_key][schedule_end:occ_end] = 1
#                     print "End violation" , int(occ_end - schedule_end)
            else:
#                 print "Zone: No occupancy" 
#                 print "Zone: %s Schedule  : (%s, %s, %s)" % (self.zone, str(schedule_start), str(schedule_end))
                for tm in range( day * numperiods , (day+1) * numperiods):
                    if tm < schedule_start:
                        if self.xf[tm] > 0 and self.xrv[tm] > 0:
                            energyloss += self.xrv[tm] * windowsize
                            room_list[zone][reheat_key][tm] = 0
                    elif tm > schedule_end:
                            energyloss += self.xrv[tm] * windowsize
                            room_list[zone][reheat_key][tm] = 0
        
#         print " Start violations : %d , End violations : %d " % (num_start_violations, num_end_violations)
        tot_violations = num_start_violations + num_end_violations
        room_list[zone][self.name+"-loss"] = energyloss
        room_list[zone][self.name+"-violations"] = tot_violations
        room_list[zone][self.name+"-violations-start"] = float(num_start_violations) 
        room_list[zone][self.name+"-violations-end"] = float(num_end_violations) 
        room_list[zone][self.name+"-violations-totdays"] = totdays
        #print room_list[zone][reheat_key]

    def computeSchedule(self):
        self.schedule = []

        # calculating total number of weeks in the sensor data
        tot_num_weeks = int(self.t[-1] / ( numperiods * 7 ))
        if (numperiods * 7 * tot_num_weeks) < self.t[-1]:
            tot_num_weeks += 1

            
        # if total number of weeks < the numbers of weeks you have asked it to train for, apply the default schedule
        if tot_num_weeks < self.numWeeksToTrain:
#             print "Warning: the numbers of weeks you have asked it to train for is greater than the number of weeks for which data is available"
            self.applySchedule()
            return
        
       
        for i in range(self.numWeeksToTrain * 7):
            self.schedule.append((None, None))
        
        if self.sameScheduleForAllDays:
#             print "Same schedule for all days"
#             Case where you impose the same schedule for all the days in a week      
            for i in range(tot_num_weeks - self.numWeeksToTrain ):
                weeks = np.arange(i, i + self.numWeeksToTrain)
#                 print "Training for weeks : ", weeks
                occupied_schedule = self.getOccupiedSchedule(weeks)
#                 print "Returned occupied schedule : ", occupied_schedule
                for day in range(7):
                    self.schedule.append(occupied_schedule[0])
#             print "Final schedule : ", self.schedule
            self.applySchedule()
            return
        else:
            # Case where there are different schedules on each day of the week
#             print "Different schedule for each day"
            for i in range(tot_num_weeks - self.numWeeksToTrain ):
                weeks = np.arange(i, i + self.numWeeksToTrain)
#                 print "Training for weeks : ", weeks

                # occupied schedule returns an array of occupancy arrays, one corresponding to each day
                occupied_schedule = self.getOccupiedSchedule(weeks, 7)
                     
#                 print "Returned occupied schedule : ", occupied_schedule
                for day in range(7):
                    self.schedule.append(occupied_schedule[day])
                    
#             print "Final schedule : ", self.schedule
            self.applySchedule()
            return


# In[24]:

import random

class FlexibleSchedule(Schedule):
    def __init__(self, schedule_name):
        self.name = schedule_name
        self.schedule = [] # the nighttime schedule
        print "Created schedule : %s" % (schedule_name)
        self.zone = None
        self.t = None
        self.xf = None
        self.xrv = None
        self.aggressiveness = 0
        self.numWeeksToTrain = 0
        
    def setAggressiveness(self, percentile):
        self.aggressiveness = percentile

    def setNumWeeksToTrain(self, numWeeks):
        self.numWeeksToTrain = numWeeks

    def setScheduleType(self, scheduleType):
        self.scheduleType = scheduleType

    def setNaiveScheduleTime(self, weekday_occ_start, weekday_occ_end):
        self.naive_occ_start_weekday = weekday_occ_start
        self.naive_occ_end_weekday = weekday_occ_end
        
    def getStartEnds(self, weeks):
        starts = []
        ends = []
        prev_idx = -2

        # your function to find the occupancy start and end times
        for idx in room_list[self.zone]["occ"]:
            # only use data which is in the current set of weeks for training
            if idx / (numperiods * 7) not in weeks:
                continue
            if prev_idx!=idx-1:
                starts.append(idx)
                if prev_idx>=0:
                    ends.append(prev_idx)
            prev_idx = idx

        ends.append(prev_idx)    
        return [ starts, ends ]

    def trackingSameOccSchedule(self, weeks):
        [ starts, ends ] = self.getStartEnds(weeks)
        # find the modulo, to convert this to a generic day
        final_starts = [ i % numperiods for i in starts ]
        final_ends = [ i % numperiods for i in ends ]
        occ_day = None
        # this is the case where you have the same schedule for all days in the week
#             print "Non weekend start times : ", final_non_weekend_starts
#             print "Non weekend end time : ", final_non_weekend_ends
        if len(final_starts) == 0 or len(final_ends) == 0:
            occ_day = (numperiods, numperiods) 
        else:
            start_time = np.nanpercentile(final_starts,self.aggressiveness)
            end_time = np.nanpercentile(final_ends,100-self.aggressiveness)+1
            if end_time > numperiods:
                end_time = numperiods - 0.01
    #             print "Chosen start time : %d , end time : %d " % (start_time, end_time)
            #occupied_idx_foraday = np.arange(start_time, end_time,1)
            if end_time < start_time:
                occ_day =  (None, None) 
            else:
                occ_day = (start_time, end_time)
        occupied_schedule = []
        for i in range(7):
            occupied_schedule.append(occ_day)
        return occupied_schedule

    def trackingDiffOccSchedule(self, weeks):
        # find the module , to convert to generic week !
        [ starts, ends] = self.getStartEnds(weeks)
        final_starts = [ i % (numperiods * 7) for i in starts ]
        final_ends = [ i % (numperiods * 7) for i in ends ]
        # this is the case where you are calculating a different schedule for each day of the week
        occupied_schedule = []
#             print "Non weekend start times : ", final_non_weekend_starts
#             print "Non weekend end time : ", final_non_weekend_ends
        for day in range(7):
            starts_day = [ (idx % numperiods) for idx in final_starts if idx >= numperiods * (day) and idx < numperiods * (day + 1)]
            ends_day = [ (idx % numperiods) for idx in final_ends if idx >= numperiods * (day) and idx < numperiods * (day + 1)]
            if len(starts_day) == 0 or len(ends_day) == 0:
#                     print "Day : %d , no start_time or end_time" % day, starts_day , ends_day
                occupied_schedule.append((numperiods, numperiods))
                continue
#                 print "Day : ", day, "Start times : ", starts_day
#                 print "Day : ", day, "End times : ", ends_day
            start_time = np.nanpercentile(starts_day,self.aggressiveness)
            end_time = np.nanpercentile(ends_day ,100-self.aggressiveness) + 1
            if end_time > numperiods:
                end_time = numperiods - 0.01
            if end_time < start_time:
                occupied_schedule.append((None, None))
                continue
            #occupied = np.arange(start_time, end_time ,1)
#                 print "Day : %d , Chosen start time : %d , end time : %d " % (day, start_time, end_time)
            occupied_schedule.append( (start_time, end_time))
        
        return occupied_schedule

    def weekdayWeekendSchedule(self, weeks):
        #print "Started calculating weekdayweekend schedule"
        occupied_schedule = []
        [ starts, ends] = self.getStartEnds(weeks)
        #print "Starts , " , starts, "Ends ,", ends

        # do for weekdays
        final_starts_weekday = [ i % numperiods for i in starts if (i % (numperiods * 7)) / numperiods < 5 ]
        final_ends_weekday = [ i % numperiods for i in ends if (i % (numperiods * 7)) / numperiods < 5  ]
        weekday_occ_schedule = None
#             print "Non weekend start times : ", final_non_weekend_starts
#             print "Non weekend end time : ", final_non_weekend_ends
        if len(final_starts_weekday) == 0 or len(final_ends_weekday) == 0:
            weekday_occ_schedule = (numperiods, numperiods) 
        else:
            start_time = np.nanpercentile(final_starts_weekday,self.aggressiveness)
            end_time = np.nanpercentile(final_ends_weekday,100-self.aggressiveness)+1
            if end_time > numperiods:
                end_time = numperiods - 0.01
#             print "Chosen start time : %d , end time : %d " % (start_time, end_time)
            #occupied_idx_foraday = np.arange(start_time, end_time,1)
            if end_time < start_time:
                weekday_occ_schedule = (None, None)
            else:
                weekday_occ_schedule = (start_time, end_time) 
        for i in range(5):
            occupied_schedule.append(weekday_occ_schedule)  

        # do for weekends
        final_starts_weekend = [ i % numperiods for i in starts if (i % (numperiods * 7)) / numperiods >= 5 ]
        final_ends_weekend = [ i % numperiods for i in ends if (i % (numperiods * 7)) / numperiods >= 5  ]
        weekend_occ_schedule = None
#             print "Non weekend start times : ", final_non_weekend_starts
#             print "Non weekend end time : ", final_non_weekend_ends
        if len(final_starts_weekend) == 0 or len(final_ends_weekend) == 0:
            weekend_occ_schedule = (numperiods, numperiods) 
        else:
            start_time = np.nanpercentile(final_starts_weekend,self.aggressiveness)
            end_time = np.nanpercentile(final_ends_weekend,100-self.aggressiveness)+1
            if end_time > numperiods:
                end_time = numperiods - 0.01
#             print "Chosen start time : %d , end time : %d " % (start_time, end_time)
            #occupied_idx_foraday = np.arange(start_time, end_time,1)
            if end_time < start_time:
                weekend_occ_schedule = (None, None)
            else:
                weekend_occ_schedule = (start_time, end_time) 
        for i in range(2):
            occupied_schedule.append(weekend_occ_schedule) 
        return occupied_schedule

    def calculateEnergyLossForSchedule(self, zone):
        self.setZone(zone)
        self.computeSchedule()
        print "Computed schedule"
        energyloss = 0
        violations = 0
        weekend_violations = 0
        nighttime_violations = 0
        othertime_violations = 0
        
        tot_num_weeks = int(self.t[-1] / ( numperiods * 7 ))
        if (numperiods * 7 * tot_num_weeks) < self.t[-1]:
            tot_num_weeks += 1
            
        num_start_violations = 0
        num_end_violations = 0
        
        totdays = 0
        reheat_key = self.name+"-"+str(self.numWeeksToTrain)+"-"+str(self.aggressiveness) + "-reheat"
#         energy_savings_key = self.name+"-"+str(self.numWeeksToTrain)+"-"+str(self.aggressiveness) + "-energysavings"
        violations_key = self.name+"-"+str(self.numWeeksToTrain)+"-"+str(self.aggressiveness) + "-violations"
        
        room_list[zone][reheat_key] = np.zeros(tot_num_weeks * 7 * numperiods) 
        room_list[zone][violations_key] = np.zeros(tot_num_weeks * 7 * numperiods)
        
        for tm in range(tot_num_weeks * 7 * numperiods):
            if self.xrv[tm] > 0:
                room_list[zone][reheat_key][tm] = 1
#                 room_list[zone][energy_savings_key][tm] = 1
                
#         print "Number of entries in schedule : ", len(self.schedule) , self.name
#         print "Total number of days : ", tot_num_weeks*7
        for day in range((tot_num_weeks*7)):

            totdays += 1
#             print self.schedule[day]
            if self.schedule[day][0] == None or self.schedule[day][1] == None:
                continue
            
            schedule_start = self.schedule[day][0] + (numperiods * day)
            schedule_end = self.schedule[day][1] + (numperiods * day)
            
            occ_day = [ tm  for tm in room_list[zone]["occ"] if tm >= day * numperiods and tm < (day+1)* numperiods ]
            if len(occ_day) > 0:
                occ_start = occ_day[0]
                occ_end = occ_day[-1]
#                 print "Zone: %s Occupancy : (%d, %d, %d, %d)" % (self.zone, occ_start, occ_end, occ_start % numperiods, occ_end % numperiods)
#                 print "Zone: %s Schedule  : (%d, %d, %d, %d)" % (self.zone, schedule_start, schedule_end, schedule_start % numperiods , schedule_end % numperiods)
                for tm in range( day * numperiods , (day+1) * numperiods):
                    if tm < min(occ_start, schedule_start):
                        if self.xf[tm] > 0 and self.xrv[tm] > 0:
                            energyloss += self.xrv[tm] * windowsize
                            room_list[zone][reheat_key][tm] = 0
                    elif tm > max(occ_end, schedule_end):
                            energyloss += self.xrv[tm] * windowsize
                            room_list[zone][reheat_key][tm] = 0
                
                if occ_start < schedule_start:
                    num_start_violations += int(schedule_start - occ_start)
                    room_list[zone][violations_key][occ_start:schedule_start] = 1
#                     print "Start violation" , int(schedule_start - occ_start)

                if occ_end > schedule_end:          
                    num_end_violations += int(occ_end - schedule_end)
                    room_list[zone][violations_key][schedule_end:occ_end] = 1
#                     print "End violation" , int(occ_end - schedule_end)
            else:
#                 print "Zone: No occupancy" 
#                 print "Zone: %s Schedule  : (%s, %s, %s)" % (self.zone, str(schedule_start), str(schedule_end))
                for tm in range( day * numperiods , (day+1) * numperiods):
                    if tm < schedule_start:
                        if self.xf[tm] > 0 and self.xrv[tm] > 0:
                            energyloss += self.xrv[tm] * windowsize
                            room_list[zone][reheat_key][tm] = 0
                    elif tm > schedule_end:
                            energyloss += self.xrv[tm] * windowsize
                            room_list[zone][reheat_key][tm] = 0
        
#         print " Start violations : %d , End violations : %d " % (num_start_violations, num_end_violations)
        tot_violations = num_start_violations + num_end_violations
        room_list[zone][self.name+"-loss"] = energyloss
        room_list[zone][self.name+"-violations"] = tot_violations
        room_list[zone][self.name+"-violations-start"] = float(num_start_violations) 
        room_list[zone][self.name+"-violations-end"] = float(num_end_violations) 
        room_list[zone][self.name+"-violations-totdays"] = totdays
        #print room_list[zone][reheat_key]

    def computeSchedule(self):
        self.schedule = []

        # calculating total number of weeks in the sensor data
        tot_num_weeks = int(self.t[-1] / ( numperiods * 7 ))
        if (numperiods * 7 * tot_num_weeks) < self.t[-1]:
            tot_num_weeks += 1

        # if total number of weeks < the numbers of weeks you have asked it to train for, apply the default schedule
        if tot_num_weeks < self.numWeeksToTrain:
#             print "Warning: the numbers of weeks you have asked it to train for is greater than the number of weeks for which data is available"
            self.applySchedule()
            return

        for i in range(self.numWeeksToTrain * 7):
            self.schedule.append((None, None))
        
        if self.scheduleType == "SAME_SCHEDULE_ALL_DAYS":
            print "===Same schedule for all days==="
#             Case where you impose the same schedule for all the days in a week 
#             print "Calculating same schedule for each day for zone ",  self.zone 
#             print "Initial number of None entries in schedule", len(self.schedule)
            for i in range(tot_num_weeks - self.numWeeksToTrain ):
                weeks = np.arange(i, i + self.numWeeksToTrain)
#                 print "Training for weeks : ", weeks
                occupied_schedule = self.trackingSameOccSchedule(weeks)
#                 print "Number of entries returned by function call ", len(occupied_schedule)
                self.schedule.extend(occupied_schedule)
#                 print "Total number of entries in occupied schedule after extending ", len(self.schedule)
#                 print "Returned occupied schedule : ", occupied_schedule
#             print "Final schedule : ", self.schedule
            self.applySchedule()
            return
        elif self.scheduleType == "DIFF_SCHEDULE_ALL_DAYS":
            # Case where there are different schedules on each day of the week
#             print "Different schedule for each day"
            print "===Calculating different schedule for each day for zone===",  self.zone
            for i in range(tot_num_weeks - self.numWeeksToTrain ):
                weeks = np.arange(i, i + self.numWeeksToTrain)
#                 print "Training for weeks : ", weeks

                # occupied schedule returns an array of occupancy arrays, one corresponding to each day
                occupied_schedule = self.trackingDiffOccSchedule(weeks)
                     
#                 print "Returned occupied schedule : ", occupied_schedule

                self.schedule.extend(occupied_schedule)
                    
#             print "Final schedule : ", self.schedule
            self.applySchedule()
            return
        elif self.scheduleType == "WEEKEND_WEEKDAY_SCHEDULE":
            print "===Calculating weekend weekday schedule for zone====",  self.zone
            for i in range(tot_num_weeks - self.numWeeksToTrain ):
                weeks = np.arange(i, i + self.numWeeksToTrain)
#                 print "Training for weeks : ", weeks

                # occupied schedule returns an array of occupancy arrays, one corresponding to each day
                occupied_schedule = self.weekdayWeekendSchedule(weeks)
                     
#                 print "Returned occupied schedule : ", occupied_schedule

                self.schedule.extend(occupied_schedule)
            self.applySchedule()
            return
        elif self.scheduleType == "RANDOM_STATIC_SCHEDULE_WEEKEND_WEEKDAY":
            print "===Calculting random static schedule for zone===", self.zone
            i = random.randint(0, tot_num_weeks - 2)
            weeks = np.arange(i, i + 2)
#                 print "Training for weeks : ", weeks

            # occupied schedule returns an array of occupancy arrays, one corresponding to each day
            occupied_schedule = self.weekdayWeekendSchedule(weeks)

#                 print "Returned occupied schedule : ", occupied_schedule
            for i in range(self.numWeeksToTrain, tot_num_weeks):
                self.schedule.extend(occupied_schedule)
            
            self.applySchedule()
            return
        
        elif self.scheduleType == "NAIVE_SCHEDULE":
            print "===Calculting naive schedule for zone===", self.zone
            for i in range(self.numWeeksToTrain*7, tot_num_weeks*7):
                if i % 7 < 5 :
                    self.schedule.append((self.naive_occ_start_weekday, self.naive_occ_end_weekday))
                else:
                    self.schedule.append((numperiods, numperiods))
            self.applySchedule()
            return
        else:
            print "=============NOT VALID SCHEDULE =========== PANIC"


# In[25]:

bestSchedule = BestSchedule("bestSch")

naiveSchedule = FlexibleSchedule("naiveSch")
naiveSchedule.setNumWeeksToTrain(0)
naiveSchedule.setScheduleType("NAIVE_SCHEDULE")
naiveSchedule.setNaiveScheduleTime(numperiods/4, numperiods - ((numperiods/24)*4)) #6am-8pm
        
staticSchedule = FlexibleSchedule("staticSch")
staticSchedule.setNumWeeksToTrain(0)
staticSchedule.setScheduleType("RANDOM_STATIC_SCHEDULE_WEEKEND_WEEKDAY")
    
allSchedules = [bestSchedule, naiveSchedule, staticSchedule ]

count = 1

print "Number of normal zones : ", len(normal_zones)
for zone in sorted(normal_zones):
    try:
        if "occ" in room_list[zone]:
            if len(room_list[zone]["occ"])>0:
                print "Doing zone : ", zone , "(%d/%d)" % (count, len(normal_zones))
                map(lambda x: x.calculateEnergyLossForSchedule(zone), allSchedules) 
                #print "Best schedule energy saved : ", room_list[zone]["bestSch-loss-weekly"]
    except:
        raise
        print "Error doing zone : ", zone
    count += 1


# In[ ]:

# # for zone in sorted(room_list):
# #     try:
# #         if "occ" in room_list[zone]:
# #             if len(room_list[zone]["occ"])>0:
# #                 print zone, room_list[zone]["bestSch-loss"]/room_list[zone]["defaultSch-loss"], room_list[zone]["aggressiveSameSch-loss"]/room_list[zone]["defaultSch-loss"], room_list[zone]["aggressiveSch-loss"]/room_list[zone]["defaultSch-loss"] 
# # #                 print zone, room_list[zone]["bestSch-violations"], room_list[zone]["aggressiveSameSch-violations"], room_list[zone]["aggressiveSch-violations"]
# #     except:
# #         print "Error doing zone : ", zone
        
# # for zone in sorted(room_list):
# #     try:
# #         if "occ" in room_list[zone]:
# #             if len(room_list[zone]["occ"])>0:
# #                 print zone, room_list[zone]["bestSch-violations"], room_list[zone]["aggressiveSameSch-violations"], room_list[zone]["aggressiveSch-violations"]
# #     except:
# #         raise
# #         print "Error doing zone : ", zone
        
# print "zone,best_savings,similar_sch_savings,per_day_customized_sch_savings,best_violations,similar_sch_violations,per_day_customized_sch_savings"
# print "number of rooms : ", len(normal_zones)
# num_days_of_exp = (data_num_weeks - numTrainingWeeks)  * 7
# for zone in sorted(normal_zones):
#     try:
#         if "occ" in room_list[zone]:
#             if len( room_list[zone]["occ"])>0:
#                 default_sched_loss = sum(room_list[zone]["defaultSch-loss-weekly"][numTrainingWeeks:])
#                 best_energy_savings = sum(room_list[zone]["bestSch-loss-weekly"][numTrainingWeeks:])
#                 best_energy_savings_percent = (best_energy_savings * 100.0) / default_sched_loss
#                 same_schedule_savings_percent = (room_list[zone]["aggressiveSameSch-loss"] * 100.0)/default_sched_loss
#                 diff_day_schedule_savings_percent = (room_list[zone]["aggressiveSch-loss"] * 100.0)/default_sched_loss
                
#                 print "%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\t\t%.2f,%.2f" % ( zone, best_energy_savings_percent , \
#                                                                               same_schedule_savings_percent , diff_day_schedule_savings_percent, \
#                         float(room_list[zone]["bestSch-violations"])/ float(num_days_of_exp), \
#                         float(room_list[zone]["aggressiveSameSch-violations"]) / float(num_days_of_exp * numperiods), \
#                         float(room_list[zone]["aggressiveSch-violations"]) / float(num_days_of_exp* numperiods) , \
#                         float(room_list[zone]["aggressiveSameSch-violations-start"]) , \
#                         float(room_list[zone]["aggressiveSameSch-violations-end"]) )   
# #                 print zone, diff_day_schedule_savings,
# #                 print float(room_list[zone]["aggressiveSch-violations"] * 100.0) / float(len(room_list[zone]["occ"])) , 
# #                 print float(room_list[zone]["aggressiveSch-violations-weekend"] * 100.0) ,
# #                 print float(room_list[zone]["aggressiveSch-violations-nighttime"] * 100.0) ,
# #                 print float(room_list[zone]["aggressiveSch-violations-other"] * 100.0)
#     except:
#         raise
#         print "Error doing zone : ", zone


# In[ ]:

# improvement1 = []
# improvement2 = []
# label = []
# for zone in sorted(room_list):
#     try:
#         if "occ" in room_list[zone]:
#             if len(room_list[zone]["occ"])>0:
#                 improvement1.append(100*room_list[zone]["aggressiveSch-loss"]/room_list[zone]["defaultSch-loss"] )
#                 improvement2.append(100*room_list[zone]["aggressiveSameSch-loss"]/room_list[zone]["defaultSch-loss"] )
#                 label.append(zone)
#     except:
#         print "Error doing zone : ", zone


# In[ ]:

# pylab.rcParams['figure.figsize'] = (20.0, 5.0)
# plt.figure()
# bar_width = 0.35
# plt.bar(np.arange(len(label)), improvement1, bar_width, color='green', alpha=0.25, label='Same schedule for each weekday')
# plt.bar(np.arange(len(label))+ bar_width, improvement2, bar_width, color='red', alpha=0.25, label='Same schedule for all weekdays')
# plt.ylim([0,100])
# plt.xticks(np.arange(len(label))+bar_width, label, rotation='vertical')
# plt.xlim([0,len(label)])
# plt.title("Sutardja Dai")
# plt.ylabel("Percentage")
# plt.legend(loc='upper left')
# plt.show()


# In[26]:

# setAggressiveness => Same thing as the percentile you were choosing in your class
# numWeeksToTrain => How many preceeding weeks you should train on. The first numWeeks will be the default schedule. 
# numWeeksToTrain => If this value > tot number of weeks in data , it will apply the default schedule to the entire period
# sameScheduleForAllDays => True : same schedule for all weekdays , False : different schedule for each day of the week
# in the output : violations is the total number of periods for which our schedule was in place when there was occupancy

final_stats = {}
for t in [ 1, 4, 8 ]:
    num_days_of_exp = (data_num_weeks - t)  * 7
    for a in [ 10, 15, 20]:
        key = str(t) + "-" + str(a)
        final_stats[key] = {}

        aggressiveSameSchedule = FlexibleSchedule("aggressiveSameSch")
        aggressiveSameSchedule.setAggressiveness(a)
        aggressiveSameSchedule.setNumWeeksToTrain(t)
        aggressiveSameSchedule.setScheduleType("SAME_SCHEDULE_ALL_DAYS")

        aggressiveSchedule = FlexibleSchedule("aggressiveSch")
        aggressiveSchedule.setAggressiveness(a)
        aggressiveSchedule.setNumWeeksToTrain(t)
        aggressiveSchedule.setScheduleType("DIFF_SCHEDULE_ALL_DAYS")
    
        weekdayWeekendSchedule = FlexibleSchedule("weekdayWeekendSchedule")
        weekdayWeekendSchedule.setAggressiveness(a)
        weekdayWeekendSchedule.setNumWeeksToTrain(t)
        weekdayWeekendSchedule.setScheduleType("WEEKEND_WEEKDAY_SCHEDULE")
        
        #allSchedules = [aggressiveSameSchedule, aggressiveSchedule]
        allSchedules = [ aggressiveSameSchedule, aggressiveSchedule, weekdayWeekendSchedule ]
        count = 0
        print "Doing trainingweeks = %d , aggressiveness = %d" % ( t, a)
        for zone in sorted(normal_zones):
        #for zone in ["300T"]:
            try:
                if "occ" in room_list[zone]:
                    if len(room_list[zone]["occ"])>0:
                        print " t = %d a = %d Doing zone : %s (%d/%d)" % (t, a, zone, count, len(normal_zones))
                        map(lambda x: x.calculateEnergyLossForSchedule(zone), allSchedules) 
                        default_sched_loss = sum(room_list[zone]["defaultSch-loss-weekly"][t:])
                        best_energy_savings = sum(room_list[zone]["bestSch-loss-weekly"][t:])
                        best_energy_savings_percent = (best_energy_savings * 100.0) / default_sched_loss
                        
                        same_schedule_savings_percent = (room_list[zone]["aggressiveSameSch-loss"] * 100.0)/ best_energy_savings
                        diff_day_schedule_savings_percent = (room_list[zone]["aggressiveSch-loss"] * 100.0)/ best_energy_savings
                        weekday_weekend_schedule_savings_percent = (room_list[zone]["weekdayWeekendSchedule-loss"] * 100.0)/ best_energy_savings
                        
                        same_schedule_savings_violations = float(room_list[zone]["aggressiveSameSch-violations"]) / float(num_days_of_exp * numperiods)
                        diff_day_schedule_savings_violations =  float(room_list[zone]["aggressiveSch-violations"]) / float(num_days_of_exp* numperiods)
                        weekday_weekend_schedule_savings_violations =  float(room_list[zone]["weekdayWeekendSchedule-violations"]) / float(num_days_of_exp* numperiods)
                        if np.isnan(same_schedule_savings_percent) or np.isnan(diff_day_schedule_savings_percent) or np.isnan(best_energy_savings) or np.isnan(weekday_weekend_schedule_savings_percent):
                            print "==================== ALARM =============================="
                            print same_schedule_savings_percent
                            print diff_day_schedule_savings_percent
                            print weekday_weekend_schedule_savings_percent
                            print best_energy_savings
                            print "========================================================="
                        final_stats[key][zone] = (same_schedule_savings_violations, diff_day_schedule_savings_violations, weekday_weekend_schedule_savings_violations,                                                   default_sched_loss, best_energy_savings, room_list[zone]["aggressiveSameSch-loss"], room_list[zone]["aggressiveSch-loss"], room_list[zone]["weekdayWeekendSchedule-loss"] )
            except:
                raise
                print "Error doing zone : ", zone
            count += 1


# In[27]:

print "#Training Weeks, Aggressiveness, Oracle, SS-energy, DS-energy, WW-energy, SS-violations, DS-violations, WW-violations"
#numzones = len(normal_zones)
savings_data = {}
violations_data = {}
for t in [ 1, 4, 8 ]:
    
    savings_data[t] = {}
    violations_data[t] = {}
    for a in [ 10, 15, 20]:
        numzones = 0
        savings_data[t][a] = {}
        violations_data[t][a] = {}
        key = str(t) + "-" + str(a)
        avg_best_energy = 0
        avg_same_energy = 0
        avg_diff_energy = 0
        avg_ww_energy = 0
        avg_same_violations = 0
        avg_diff_violations = 0
        avg_ww_violations = 0
        for zone in normal_zones:
#             continueFlag = False
#             for tupleid in range(len(final_stats[key][zone])):
#                 #print v
#                 if np.isnan(final_stats[key][zone][tupleid]):
#                     continueFlag = True
#                     break
#             if continueFlag:
#                 continue
           
#             print zone, final_stats[key][zone]
            if np.isnan(final_stats[key][zone][6]) or np.isnan(final_stats[key][zone][5]) or final_stats[key][zone][3] == 0                     or final_stats[key][zone][4] == 0 or np.isnan(final_stats[key][zone][7]):
                continue
            
            avg_same_energy += ( final_stats[key][zone][5] * 100.0 / final_stats[key][zone][3] )
            avg_diff_energy += ( final_stats[key][zone][6] * 100.0 / final_stats[key][zone][3] )
            avg_ww_energy += ( final_stats[key][zone][7] * 100.0 / final_stats[key][zone][3] )
            
            avg_same_violations += final_stats[key][zone][0]
            avg_diff_violations += final_stats[key][zone][1]
            avg_ww_violations += final_stats[key][zone][2]
            
            default_sched_loss = sum(room_list[zone]["defaultSch-loss-weekly"][t:])
            best_energy_savings = sum(room_list[zone]["bestSch-loss-weekly"][t:])
            

            best_energy_savings_percent = (best_energy_savings * 100.0) / default_sched_loss
            avg_best_energy += best_energy_savings_percent
            numzones += 1
            
        savings_data[t][a]["best"] = avg_best_energy/numzones
        savings_data[t][a]["same"] = avg_same_energy/numzones
        savings_data[t][a]["diff"] = avg_diff_energy/numzones
        savings_data[t][a]["ww"] = avg_ww_energy/numzones
        
        violations_data[t][a]["same"] = (avg_same_violations*100)/numzones
        violations_data[t][a]["diff"] = (avg_diff_violations*100.0)/numzones
        violations_data[t][a]["ww"] = (avg_ww_violations*100.0)/numzones
        
        print "%d,%d,  %.1f,  %.1f,  %.1f,  %.1f,  %.1f,  %.1f,  %.1f" % (t, a, avg_best_energy/numzones, avg_same_energy/numzones , avg_diff_energy/numzones , avg_ww_energy/numzones , (avg_same_violations*100)/numzones, (avg_diff_violations*100.0)/numzones, (avg_ww_violations*100.0)/numzones)

print "BestSch-energy, StaticSch-energy, NaiveSch-energy, StaticSch-violations, NaiveSch-violations"

bestSchEnergy = 0
staticSchEnergy = 0
staticSchViolations = 0
naiveSchEnergy = 0
naiveSchViolations = 0
numzones = 0
for zone in normal_zones:
    if room_list[zone]["bestSch-loss"] == 0 or room_list[zone]["defaultSch-loss"] == 0:
        continue
    staticSchEnergy += (room_list[zone]["staticSch-loss"] * 100.0 / room_list[zone]["defaultSch-loss"])
    staticSchViolations += float(room_list[zone]["staticSch-violations"] * 100) /float(data_num_weeks* 7 * numperiods)
    naiveSchEnergy += (room_list[zone]["naiveSch-loss"] * 100.0 / room_list[zone]["defaultSch-loss"])
    naiveSchViolations += float(room_list[zone]["naiveSch-violations"] * 100) /float(data_num_weeks* 7 * numperiods)
    bestSchEnergy += (room_list[zone]["bestSch-loss"] * 100.0 / room_list[zone]["defaultSch-loss"])
    numzones += 1
print "%.1f,  %.1f,  %.1f,  %.1f,  %.1f," % (bestSchEnergy/numzones, staticSchEnergy/numzones, naiveSchEnergy/numzones,                                             staticSchViolations/numzones, naiveSchViolations/numzones)

#print sum(room_list[zone]["bestSch-loss-weekly"])


# In[111]:

from matplotlib import cm

pylab.rcParams['figure.figsize'] = (24.0, 18.0)
N = 3
ind = np.arange(N)  # the x locations for the groups
width = 0.2        # the width of the bars

fig, ax = plt.subplots()

count = 0
extraspace = 0.0
numstreams = 0
patterns = ['-', '\\', '//']
colors=['grey', 'silver', 'white', 'maroon', 'orangered', 'lightsalmon', 'darkblue', 'cornflowerblue', 'cyan']
mapping = { "diff" : "Per-Day" , "ww" : "WW", "same" : "Weekly"}
for t in sorted(savings_data):
    for scheme in [ "diff", "ww", "same"]:
        arr = []
        for a in sorted(savings_data[t]):
            arr.append(savings_data[t][a][scheme])
#             arr.append(violations_data[t][a][scheme])

        rects = ax.bar(ind + (count+1) * width + extraspace + (numstreams%3)*width/4, arr, width/2, color=colors[numstreams%9] , label='#WND=' + str(t) + ', schedule=' + mapping[scheme], hatch=patterns[numstreams%3])
        numstreams += 1
    count += 1
    
    extraspace += 0.05
# add some text for labels, title and axes ticks
ax.set_ylabel('Energy Saved in Reheat (%)')
ax.set_xlabel('Aggressiveness (percentile)')

ax.set_xticks(ind + (width * (count+2) + extraspace)/2 )
ax.set_xticklabels(('10', '15', '20'))

plt.ylim([0,100])
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(40)
#plt.legend(loc=9, bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize = 40)

plt.show()


# In[112]:

from matplotlib import cm

pylab.rcParams['figure.figsize'] = (24.0, 18.0)
N = 3
ind = np.arange(N)  # the x locations for the groups
width = 0.2         # the width of the bars

fig, ax = plt.subplots()

count = 0
extraspace = 0
numstreams = 0
colors=['grey', 'silver', 'white', 'maroon', 'orangered', 'lightsalmon', 'darkblue', 'cornflowerblue', 'cyan']
for t in sorted(savings_data):
    for scheme in [ "diff", "ww", "same"]:
        arr = []
        for a in sorted(violations_data[t]):
            arr.append(violations_data[t][a][scheme])
#             arr.append(violations_data[t][a][scheme])

        rects = ax.bar(ind + (count+1) * width + extraspace + (numstreams%3)*width/4, arr, width/2, color=colors[numstreams%9] , label='#WND=' + str(t) + ', schedule=' + scheme, hatch=patterns[numstreams%3])
        numstreams += 1
    count += 1
    
    extraspace += 0.05
# add some text for labels, title and axes ticks
ax.set_ylabel('Occupancy Comfort Violations (%)')
ax.set_xlabel('Aggressiveness (percentile)')

ax.set_xticks(ind + (width * (count+2) + extraspace)/2 )
ax.set_xticklabels(('10', '15', '20'))

plt.ylim([0,20])
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(40)
#plt.legend(loc=9, bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=40)

plt.show()


# In[114]:

pylab.rcParams['figure.figsize'] = (15.0, 10.0)
ahudata = {}

tot_reheat = np.zeros(numperiods)
tot_occ = np.zeros(numperiods)
tot_occ_var = np.zeros(numperiods)
#reheat_key = "aggressiveSch-4-20-reheat"
reheat_keys = []
#for t in [ 4 ]:
for t in [ 8 ]:
#     for a in [ 20 ]:
    for a in [ 10, 20]:
        reheat_keys.append("aggressiveSch-" + str(t) + "-" + str(a) + "-reheat")
        reheat_keys.append("aggressiveSameSch-" + str(t) + "-" + str(a) + "-reheat")
max_training_weeks = 8

schedule_reheat = np.zeros((len(reheat_keys), numperiods))
reheat_sums = np.zeros(len(reheat_keys))
print "Number of normal zones : ", len(normal_zones)

for zone in normal_zones:
    
    ahudata[zone] = {}
    ahudata[zone]["reheat"] = np.zeros(numperiods)
    ahudata[zone]["occ"] = np.zeros(numperiods)
    ahudata[zone]["occ-std"] = np.zeros(numperiods)
    times, flow = getSortedNamedStream(zone, "flow")
    _,   reheat = getSortedNamedStream(zone, "reheat")
    tot_num_weeks = times[-1] / ( numperiods * 7 )
    if (numperiods * 7 * tot_num_weeks) < times[-1]:
        tot_num_weeks += 1
    #print "Doing zone", zone, tot_num_weeks
        
    for idx in room_list[zone]["occ"]:
        if (idx % (numperiods * 7) ) / numperiods <5 :
            ahudata[zone]["occ"][idx % numperiods] += 1
            
    for tm in times:
        if reheat[tm] > 0:
            if (tm % (numperiods * 7) ) / numperiods <5 :
                ahudata[zone]["reheat"][tm % numperiods] += 1
                
    #print "Length : ", len(room_list[zone][reheat_key])  
    
    for key in reheat_keys:
        ahudata[zone][key] = np.zeros(numperiods)
        training_weeks = int(key.split("-")[1].strip())
        for tm in range(numperiods* max_training_weeks*7, numperiods * tot_num_weeks * 7):
            if room_list[zone][key][tm] > 0:
                  if (tm % (numperiods * 7) ) / numperiods <5 :
                    ahudata[zone][key][tm % numperiods] += 1
                    if tm % numperiods < numperiods / 6:
                        print "Strange" , zone, tm, tm % numperiods, room_list[zone][key][tm]
        ahudata[zone][key] /= float((tot_num_weeks-max_training_weeks) * 5)
        
    ahudata[zone]["occ"] /= float(tot_num_weeks * 5)
    for i in range(numperiods):
        ahudata[zone]["occ-std"][i] = np.sqrt(np.power((1 - ahudata[zone]["occ"][i]),2)*ahudata[zone]["occ"][i])
    
    ahudata[zone]["reheat"] /= float(tot_num_weeks * 5)
    #for key in reheat_keys:
    #    ahudata[zone][key] /= float(tot_num_weeks * 5)
    #if prev_idx!=idx-1:
    tot_reheat += ahudata[zone]["reheat"]
    tot_occ += ahudata[zone]["occ"]
    tot_occ_var += [ahudata[zone]["occ-std"][i]*ahudata[zone]["occ-std"][i] for i in range(numperiods)]

    for i in range(len(reheat_keys)):
        schedule_reheat[i] += ahudata[zone][reheat_keys[i]]
        #if i > 1 and sum(ahudata[zone][reheat_keys[i]]) > sum(ahudata[zone][reheat_keys[i-2]]):
        #    print "zone", zone, reheat_keys[i] ,  sum(ahudata[zone][reheat_keys[i]]) , reheat_keys[i-2], sum(ahudata[zone][reheat_keys[i-2]])
        #reheat_sums[i] += sum(ahudata[zone][reheat_keys[i]])


fig, ax1 = plt.subplots()
pylab.rcParams['figure.figsize'] = (10.0, 10.0)
ax1.plot([tot_reheat[k]/max(tot_reheat) for k in range(numperiods)], 'm', label="total energy consumed for reheat")
ax2 = ax1.twinx()
ax2.plot([100*tot_occ[k]/len(normal_zones) for k in range(numperiods)], 'k--', label="building occupancy envelope")
ax2.fill_between(range(numperiods),[100*(tot_occ[k]-1.96*np.sqrt(tot_occ_var[k]/float(tot_num_weeks*5)))/len(normal_zones) for k in range(numperiods)],[100*(tot_occ[k]+1.96*np.sqrt(tot_occ_var[k]/float(tot_num_weeks*5)))/len(normal_zones) for k in range(numperiods)], facecolor='#F0F8FF', alpha=0.5, edgecolor='#8F94CC', linewidth=1, linestyle='dashed')

# print [100*tot_occ[k]/len(normal_zones) for k in range(numperiods)]
# print [100*(tot_occ[k]-1.96*np.sqrt(tot_occ_var[k]/float(tot_num_weeks*5)))/len(normal_zones) for k in range(numperiods)]
# print [100*(tot_occ[k]+1.96*np.sqrt(tot_occ_var[k]/float(tot_num_weeks*5)))/len(normal_zones) for k in range(numperiods)]


ax1.set_ylim(0, 1)
ax1.set_ylabel("Normalized Energy Consumption")
ax2.set_ylim(0, 100)
ax2.set_ylabel("Occupied Zones (pct.)", rotation=270, labelpad=15)

for i in range(len(reheat_keys)):
    ax1.plot([schedule_reheat[i][k]/max(tot_reheat) for k in range(numperiods)], label=reheat_keys[i])
    #print reheat_keys[i], reheat_sums[i]
for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels()):
    item.set_fontsize(20)
for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] + ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(20)
        
plt.xlim(0, numperiods-1)
plt.xticks(np.arange(0, numperiods, numperiods/4), ["12am", "6am", "12pm", "6pm"])
plt.title("Occupacy & Relative Energy Consumption on Weekdays")
ax1.legend(loc=2, fontsize = 'x-large')
ax2.legend(loc=1, fontsize = 'x-large')
plt.show()


# In[ ]:

# improvement_lst1 = []
# improvement_lst2 = []
# improvement_lst3 = []
# improvement_lst4 = []

# violation_lst1 = []
# violation_lst2 = []
# violation_lst3 = []
# violation_lst4 = []

# such_zones = []

# count = 0
# for a in [ 5, 10, 15, 20 ]:
#     for zone in sorted(normal_zones):
#         if "nan" in str(final_stats[key][zone]):
#                 continue
#         if final_stats["8-" + str(a)][zone][2] < final_stats["4-" + str(a)][zone][2] and \
#             final_stats["4-" + str(a)][zone][2] < final_stats["2-" + str(a)][zone][2] and \
#             final_stats["2-" + str(a)][zone][2] < final_stats["1-" + str(a)][zone][2]:
#                 improvement_lst1.append(final_stats["1-" + str(a)][zone][0])
#                 improvement_lst2.append(final_stats["2-" + str(a)][zone][0])
#                 improvement_lst3.append(final_stats["4-" + str(a)][zone][0])
#                 improvement_lst4.append(final_stats["8-" + str(a)][zone][0])
                
#                 violation_lst1.append(final_stats["1-" + str(a)][zone][2])
#                 violation_lst2.append(final_stats["2-" + str(a)][zone][2])
#                 violation_lst3.append(final_stats["4-" + str(a)][zone][2])
#                 violation_lst4.append(final_stats["8-" + str(a)][zone][2])
                
#                 such_zones.append(zone)
                
#                 print zone , "\n",final_stats["8-" + str(a)][zone], final_stats["4-" + str(a)][zone], final_stats["2-" + str(a)][zone], final_stats["1-" + str(a)][zone]
#                 print "\n"
#                 count += 1
                
# print "number of such zones : ", count

# plt.figure()
# bar_width = 0.2
# plt.bar(np.arange(len(such_zones)), violation_lst1, bar_width, color='red', alpha=0.25, label='1 week of learning')
# plt.bar(np.arange(len(such_zones))+ bar_width, violation_lst2, bar_width, color='orange', alpha=0.25, label='2 weeks of learning')
# plt.bar(np.arange(len(such_zones))+ 2*bar_width, violation_lst3, bar_width, color='yellow', alpha=0.25, label='4 weeks of learning')
# plt.bar(np.arange(len(such_zones))+ 3*bar_width, violation_lst4, bar_width, color='green', alpha=0.25, label='8 weeks of learning')
# plt.xticks(np.arange(len(such_zones))+2*bar_width, such_zones, rotation='vertical')
# plt.xlim([0,len(such_zones)])
# plt.title("Sutardja Dai")
# plt.ylabel("Percentage")
# plt.legend(loc='upper left')
# plt.show()


# In[115]:

pylab.rcParams['figure.figsize'] = (30.0, 10.0)
def plotApparentOccupancy(zone):
    global avg_weekly_occupancy_bitmap
    
    data = avg_weekly_occupancy_bitmap[zone][:]
    fig, ax = plt.subplots()
    ax.plot(data,'g--',linewidth=3);
    ax.set_xticks(np.arange(0,len(data),numperiods/2), minor=False)
    ax.set_xbound(lower = 0, upper = len(data))
    ax.set_ybound(lower = 0, upper = 1.1)
    ax.grid(which='major', axis='x', linestyle=':')
    col_label = []
    for i in range(len(day_labels)):
        col_label.append(day_labels[i])
        col_label.append(" ")
            
    ax.set_xticklabels(col_label,rotation=45)

    
    ax.set_title("Apparent Occupancy")
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(26)
    plt.show()    


# In[116]:

# For Debuging:
global schedule_nighttime

# test_zone = 'COMP LAB 200'
# test_zone = '222 WOMENS BATH'
# test_zone = '621C OFFICE'
# test_zone = '250 CLASSROOM'
# test_zone = '254 CLASSROOM'
# test_zone = '730 CONFERENCE'
test_zone = 'CYBERCAFE 210'

extractMidRangeFreqsForZone(test_zone, False)

use_IMFs = True

calculateEnergyLoss(test_zone, schedule_nighttime, use_IMFs)
plotApparentOccupancy(test_zone)

use_IMFs = False

calculateEnergyLoss(test_zone, schedule_nighttime, use_IMFs)
plotApparentOccupancy(test_zone)


# In[ ]:



