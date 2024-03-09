# VELO Pixel Calibration Analysis
## Heterogeneity of VELO’s pixels response to a monochromatic Fe55 radioactive source
The Vertex Locator (VELO) is a detector forming part of the Large Hadron Collider beauty experiment, at CERN, composed of 256x256 pixel squares, tasked with the particle detection when crossing a pixel. After its recent replacement, the pixels in the VELO need to be calibrated. In the study of the thesis, the heterogeneity of the pixels and the nature of the discrepancy was inspected to determine the calibration needs of the VELO. In what follows below, the research project is not discussed, rather, an overview of the approach to the data analysis performed to draw conclusions, with the goal of showcasing 

### Before starting
With the calibration purposes, some data was taken with a single module of 256x256 pixels, exposed to a radiation source and recording the particle hits on each pixel. To record a particle hit, the electric charge produced with its passing should be above a threshold, otherwise no hit is recorded. This is done to avoid recording false hits from electric fluctuations. The data available, then, consisted of a 256x256 csv file per acquisition run and set threshold, containing the amount of hits on each pixel for a fixed period of time. This was done for a large amount of runs per threshold and a large amount of thresholds. Additionally, some small python mathematical model and filtering functions were provided by another member of the team, that I will not showcase here for privacy. The function names are ```fitfunction```, ```check_badscan``` and ```get_mask```, you may see them in the code snippets that follow. 

#### My task
Equipped with this data, it was my responsability to load, transform and fit the data to a mathematical model describing the amount of hits on each pixel for each threshold. More importantly, I should then evaluate the goodness of the fit on each pixel individually and compare it to the fit performed to the pixel grid as a whole. From this I would then draw conclusions on the difference between the pixels, if any, and the reasons for this. 

### Data loading and transforming
To load and transform the data into the required format, a function ```fluxpixelthr``` is written. The function loads all the csv files corresponding to each acquisition of each threshold for each run and computes the flux (particle hits per unit of time) for each pixel and for the grid as a whole (ASIC). The proper error analysis is performed and stored separately. Then, a 256x256 csv file is generated and stored for the flux of particles on each pixel and the flux uncertainties for each threshold. The file loading and saving are done with numpy. Below you can find the code snippet for this function. Note that some parts have been omitted for data protection.

<details>

<summary>Click to see the code snippet</summary>
  
```python
def fluxpixelthr(filepath, fileprefix, ST, nacq, fromThr, toThr, stepThr, CUT_BAD=True, bad_cut=2):
    '''
    Creates a list of flux per pixel for each threshold, the uncertainty of the flux per pixel for each threshold,
    the total flux on the ASIC per threshold and the uncertainty of the total flux on the ASIC per threshold.
    The analyzed data files are named in the form <filepath+fileprefix+str(ST)+'_THR_'+str(thr)+'_'+str(acq)+'.csv'>
    (e.g. /data/bfys/LucasvdH/VELOdata/coolsource/Module0_VP0-1_ECS_data_ST_1s310ms_THR_1320_3.csv).
    The saved files will be named in the form <filepath+'<datatype>'+fileprefix+str(ST)+'_THR_'+str(thr)+'.csv'>, 
    where datatype is Fluxperpixel, FluxperASIC, UncertaintyFluxperpixel or UncertaintyFluxperASIC.

    Arguments:
    - filepath (str): Filepath (folder) to the data (e.g. '/data/bfys/LucasvdH/VELOdata/coolsource/').
    - fileprefix (str): Prefix of the data files (e.g. 'Module0_VP0-1_ECS_data_ST_').
    - ST (str): shutter time (e.g. '1s310ms').
    - nacq (int): Number of acquisitions per threshold (e.g. 100).
    - fromThr (int): Starting threshold of the scan (e.g. 1500).
    - toThr (int): Ending threshold of the scan (e.g. 1600).
    - stepThr (int): Increment of the threshold between scans (e.g. 5).
    - CUT_BAD (bool): Default True. Skip data acquisitions that contain bad data. 
    - bad_cut (float, int): Default 2. Used to identify bad data acquisitions. 
    If flux per pixel on the last columns exceeds bad_cut, it is faulty. 
    '''
    Thrs=[]
    Flux=[]
    F_ASIC=np.zeros(((toThr-fromThr)/stepThr+1, 1))
    unc_F_ASIC=np.zeros(((toThr-fromThr)/stepThr+1, 1))
    mask=get_mask(filepath, fileprefix)

    st=ST.split('ms')
    st=st[0].split('s')
    sec=float(st[0])
    milisec=float(st[1])/1000
    acqtime=sec+milisec
    
    print('\n\nANALAYZING for {}{}{}*.csv'.format(filepath, fileprefix, str(ST)))
    for thr in range(fromThr, toThr+1, stepThr):
        Thrs.append(thr)
        print('Analyzing for thr={}'.format(thr))
        
        #t_hits: Intermediary variable to sum the total hits, thereafter converted to flux
        # t_hits=np.ma.masked_where(mask>0, np.zeros((256, 256), int))
        t_hits=np.zeros((256, 256))
        goodacq=0

        for acq in range(nacq):
            hits=np.loadtxt(filepath+fileprefix+str(ST)+'_THR_'+str(thr)+'_'+str(acq)+'.csv', dtype=int, delimiter=',')
            hits=np.ma.masked_where(mask>0, hits)
            means, sigmas, BAD=check_badscan(hits, bad_cut)
            
            #If the acquisition is faulty, it skips it. For some thresholds, all acquisitions are bad
            if (CUT_BAD==True and BAD==True):
                continue
            
            t_hits+=hits
            goodacq+=1
        
        totalacqtime=goodacq*acqtime
        Flux.append(t_hits/totalacqtime)
        #Saves the flux on each pixel and, seperately, its uncertainty, as a 256x256 array
        np.savetxt(filepath+'Fluxperpixel_'+fileprefix+str(ST)+'_THR_'+str(thr)+'.csv', t_hits/totalacqtime, delimiter=',')
        np.savetxt(filepath+'UncertaintyFluxperpixel_'+fileprefix+str(ST)+'_THR_'+str(thr)+'.csv', np.sqrt(t_hits)/totalacqtime, delimiter=',')
        
        #Saves the total flux on the ASIC per threshold and, separately, its uncertainty, as a numberofthrs x 1 array
        F_ASIC[(thr-fromThr)/stepThr][0]=np.nansum(t_hits)/totalacqtime
        unc_F_ASIC[(thr-fromThr)/stepThr][0]=np.nansum(np.sqrt(t_hits))/totalacqtime
    np.savetxt(filepath+'FluxperASIC_'+fileprefix+str(ST)+'_THR_'+str(fromThr)+'-'+str(toThr)+'-step'+str(stepThr)+'.csv', F_ASIC, delimiter=',')
    np.savetxt(filepath+'UncertaintyFluxperASIC_'+fileprefix+str(ST)+'_THR_'+str(fromThr)+'-'+str(toThr)+'-step'+str(stepThr)+'.csv', unc_F_ASIC, delimiter=',')
```

</details>
