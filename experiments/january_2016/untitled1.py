# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 07:17:45 2016

@author: sala
"""

def readserial(wfo, dt):
    
    wf = wfo.copy()
    #Threshold for logic
    loglimit=1.45

    #Bit period in seconds
    period = 20e-9

    #Bit period in vector indexes (NOT integer for better result)
    step = period / dt;

    #Half range for bit identification (RELATIVE)
    idrange = step * 0.1;

    #Contrast enhancement/waveform digitalization
    wf[wf<loglimit] = 0
    wf[wf>=loglimit] = 1

    #Search for startbit
    startbit = np.where(wf>0.5)[0][0];

    #First bit position (middle of first bit)
    startpos = startbit + 1.22 * step;


#Recover binary number
    bin = []
    points = []
    for i in range(32):
        index = int(round(startpos + i * step))
        
        #Check waveform overrun
        if (index > len(wf)):
            break;
        
        
        #Scan ID range
        r1 = int(round(index - idrange))
        r2 = int(round(index + idrange))
        #Check and correct waveform overrun
        if (r1 < 0):
            r1 = 0    
        
        if (r2 > len(wf)):
            r2=len(wf)
        
        
        #Gather data
        bit = 0
        #print r1, r2
        for k in range(r1, r2, 1):
            #points=[points,
            #print k, wf[k]
            if (int(wf[k]) == 1):
                #print wf[k]
                bit = 1
                break;

                
        #Add next binary component (bit)
        bin.append(bit);
        #print bin
        
    #Convert binary to decimal number
    num = 0
    for i in range(len(bin)):
        num = num + bin[i] * 2 ** i;
    
    return num, wf
