# VELO Pixel Calibration Analysis
## Heterogeneity of VELO’s pixels response to a monochromatic Fe55 radioactive source
The Vertex Locator (VELO) is a detector forming part of the Large Hadron Collider beauty experiment, at CERN, composed of 256x256 pixel squares, tasked with the particle detection when crossing a pixel. After its recent replacement for another model, the pixels in the VELO need to be calibrated. In the study of the thesis, the heterogeneity of the pixels and the nature of the discrepancy was inspected to determine the calibration needs of the VELO. In what follows below, the research project is not discussed, rather, an overview of the approach to the data analysis performed to draw conclusions.

### Before starting
With the calibration purposes, some data was taken with a single module of 256x256 pixels, exposed to a radiation source and recording the particle hits on each pixel. To record a particle hit, the electric charge produced with its passing should be above a threshold, otherwise no hit is recorded. This is done to avoid recording false hits from electric fluctuations. The data available, then, consisted of a 256x256 csv file per acquisition run and set threshold, containing the amount of hits on each pixel for a fixed period of time. This was done for a large amount of runs per threshold and a large amount of thresholds. Additionally, some small python mathematical model and filtering functions were provided by another member of the team, that I will not showcase here for privacy. The function names are ```fitfunction```, ```check_badscan``` and ```get_mask```, you may see them in the code snippets that follow. 

#### My task
Equipped with this data, it was my responsability to load, transform and fit the data to a mathematical model describing the amount of hits on each pixel for each threshold. More importantly, I should then evaluate the goodness of the fit on each pixel individually and compare it to the fit performed to the pixel grid as a whole. From this I would then draw conclusions on the difference between the pixels, if any, and the reasons for this. 

### 1. Data loading and transforming
To load and transform the data into the required format, a function ```fluxpixelthr``` is written. The function loads all the csv files corresponding to each acquisition of each threshold for each run and computes the flux (particle hits per unit of time) for each pixel and for the grid as a whole (ASIC). The proper error analysis is performed and stored separately. Then, a 256x256 csv file is generated and stored for the flux of particles on each pixel and the flux uncertainties for each threshold. The file loading and saving are done with ```NumPy```. Below you can find the code snippet for this function. Note that some parts have been omitted for data protection.

<details>

<summary>Click to see the code snippet</summary>
  
```python
def fluxpixelthr(filepath, fileprefix, ST, nacq, fromThr, toThr, stepThr, CUT_BAD=True, bad_cut=*):
    '''
    Creates a list of flux per pixel for each threshold, the uncertainty of the flux per pixel for each threshold,
    the total flux on the ASIC per threshold and the uncertainty of the total flux on the ASIC per threshold.
    The analyzed data files are named in the form ***
    (e.g. ***).
    The saved files will be named in the form ***.

    Arguments:
    - filepath (str): Filepath (folder) to the data (e.g. ***).
    - fileprefix (str): Prefix of the data files (e.g. ***).
    - ST (str): shutter time (e.g. ***).
    - nacq (int): Number of acquisitions per threshold (e.g. ***).
    - fromThr (int): Starting threshold of the scan (e.g. ***).
    - toThr (int): Ending threshold of the scan (e.g. ***).
    - stepThr (int): Increment of the threshold between scans (e.g. ***).
    - CUT_BAD (bool): Default True. Skip data acquisitions that contain bad data. 
    - bad_cut (float, int): Default ***. Used to identify bad data acquisitions. 
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

### 2. Fitting the data
Possessing the flux per threshold, a function ```fitter``` is now written to load these, filter out the pixels with insufficient data to perform a fit and fit the mathematical model to the remaining pixels. After the fit, for each fitted parameter (three), two 256x256 csv files are generated: one with the value of the fitted parameter and one with the uncertainties of the parameter for each pixel. Then, another 256x256 csv file is saved containing the arbitrarily defined fit type per pixel (no fit attempt, no fit found, bad fit, good fit). See below for the code snippet.

<details>
<summary>Click to see the code snippet</summary>

```python
def fitter(FPTfolder, FPTfilename, ST, fromThr, toThr, stepThr, p0, nohits_cut=0.8, alpha=0.05):
    '''
    Finds best fit to the flux per pixel and total flux on pixel grid to fitfunction (defined below). Saves eight 256x256 arrays 
    containing the fitted E0, f and s parameter, their uncertainties and the fit type (1 for good fit; 0 for failed chi square test or unphysical 
    values of the fitted parameters, which is the case when E0, f or s is negative; -1 if the fit could not converge; and -2
    if the data is cut before fitting). For the 0, -1 and -2 cases (so, if not a good fit), the value of the parameters is 0.
    Zeros and NaN values are trimmed. Uses files named in the form ***. 
    Saves files named in the form ***, where datatype is E0matrix, fmatrix, smatrix, fittypematrix
    or fixedfittypematrix.

    Arguments:
    - FPTfolder (str): Folder path to the Flux per Pixel per Threshold arrays files
    (e.g. ***).
    - FPTfilename (str): Path to the Flux per Pixel per Threshold arrays made by fluxpixelthr (needs importing)
    (e.g. ***).
    - ST (str): Shutter time (e.g. ***).
    - fromThr (int): Starting threshold of the scan (e.g. ***).
    - toThr (int): Ending threshold of the scan (e.g. ***).
    - stepThr (int): Increment of the threshold between scans (e.g. ***).
    - p0 (list of length 4): initial guess of the parameters to fit [f, A, s, E0].
    - nohits_cut (float between 0 and 1): Default=0.8. Mask the pixel if it has too many zero-valued points 
    (e.g. nohits_cut=0.8 means that the pixel will be masked if the it got no hits for 80% or more of the thresholds).
    - alpha (float): Default=0.05. Level of significance.
    '''

    #Loading FluxperASIC first to find indices of bad thresholds
    fluxASIC=np.loadtxt(FPTfolder+***+str(fromThr)+'-'+str(toThr)+***+str(stepThr)+'.csv', dtype=float, delimiter=',')
    
    #If for some thresholds, all acquisitions were bad (which returns nan) set a index to slice all pertinent data to leave the data after the bad thresholds
    #If all thresholds were good, nan_cut is set to 0, which doesn't slice the data
    if any(np.isnan(fluxASIC)):
        nan_cut=np.where(np.isnan(fluxASIC))[0][-1]+1
    else:
        nan_cut=0
    
    #Loading Fluxperpixel and UncertaintyFluxperpixel files. NaN cut is applied by cutting Thrs
    Thrs=list(range(fromThr, toThr+1, stepThr))[nan_cut:]
    FPT=[]
    unc_FPT=[]
    for thr in Thrs:
        f=np.loadtxt(FPTfolder+FPTfilename+ST+***+str(thr)+'.csv', dtype=float, delimiter=',')
        u=np.loadtxt(FPTfolder+***+ST+***+str(thr)+'.csv', dtype=float, delimiter=',')
        FPT.append(f)
        unc_FPT.append(u)
    

    #Assigning a 256x256 matrix to the fitted parameters. The 0s are to be replaced by the calculated values. If no or bad fits, the element is unchanged
    E0matrix, fmatrix, smatrix= np.zeros((256, 256)), np.zeros((256, 256)), np.zeros((256, 256))
    #Assigning a 256x256 matrix to the uncertainties in the fitted parameters
    uE0matrix, ufmatrix, usmatrix= np.zeros((256, 256)), np.zeros((256, 256)), np.zeros((256, 256))
    #Assigning a 256x256 matrix to fit type. These will express the type (lack of hits (-2), unable to find (-1), bad (0) or good (1) fits) of fit performed on the pixel. One for free fit and one for E0, f and s fixed fit
    fittypematrix, fixedfittypematrix= np.zeros((256, 256)), np.zeros((256, 256))

    #Performing fit for flux on ASIC
    fluxASIC=fluxASIC[nan_cut:]
    unc_fluxASIC=np.loadtxt(FPTfolder+'UncertaintyFluxperASIC_Module0_VP0-1_ECS_data_ST_'+ST+'_THR_'+str(fromThr)+'-'+str(toThr)+'-step'+str(stepThr)+'.csv', dtype=float, delimiter=',')[nan_cut:]
    poptASIC, pcovASIC=curve_fit(fitfunction, Thrs, fluxASIC, sigma=unc_fluxASIC, p0=p0)
    poptASIC[1]=poptASIC[1]/65352 #To rescale to the flux on a single pixel. Used to calculate X2ASIC
    E0ASIC=poptASIC[-1] #Renaming for reading ease

    analysistime=[] #Just to make an estimation of total analysis time
    #Selecting a single pixel
    for row in range(256):
        st=time.time()
        print('\nPerfoming the fit for the pixels on row {}'.format(row))
        for column in range(256):
            fluxpixel=[i[row][column] for i in FPT]
            unc_fluxpixel=[i[row][column] for i in unc_FPT]

            #If fluxpixel contains too many zeros, don't analyze it
            if (len(fluxpixel)-np.count_nonzero(fluxpixel))/len(fluxpixel)>=nohits_cut:
                #Assign -2 for type of fit if cut for lack of hits
                fittypematrix[row][column]=-2
                fixedfittypematrix[row][column]=-2
                continue

            #Cut the hanging zeros on the left and right
            trimcut=np.nonzero(fluxpixel)
            thrs=Thrs[trimcut[0][0]:trimcut[0][-1]+1]
            fluxpixel=np.trim_zeros(fluxpixel)
            unc_fluxpixel=unc_fluxpixel[trimcut[0][0]:trimcut[0][-1]+1]
            


            ###Performing fit
            try:
                popt, pcov=curve_fit(fitfunction, thrs, fluxpixel, sigma=unc_fluxpixel, p0=p0)
            except RuntimeError:
                #If fit could not converge, assign -1 to fit type
                fittypematrix[row][column]=-1
                continue
            
            #Setting expected values and calculating X2
            E=fitfunction(thrs, *popt)
            X2=chisquare(np.array(fluxpixel), E, np.array(unc_fluxpixel))
            
            
            #Degrees of freedom for critical X2 (fitting four parameters, so -4)
            dof=len(fluxpixel)-4
            criticalX2=scipy.stats.chi2.ppf(1-alpha, dof)

            #Mask pixel if X2>criticalX2, X2==0, or E0, f, s are negative (by leaving the zero in the matrix)
            #If calculating X2 was not possible (division by zero, X2=0), then consider the fit bad
            if X2<criticalX2 and X2!=0 and popt[-1]>0 and popt[0]>0 and popt[1]>0:
                #If the fit was good, change the 0 in the 256x256 matrices by the fitted value and its uncertainty
                E0matrix[row][column]=popt[-1]
                uE0matrix[row][column]=np.sqrt(np.diag(pcov))[-1]
                fmatrix[row][column]=popt[0]
                ufmatrix[row][column]=np.sqrt(np.diag(pcov))[0]
                smatrix[row][column]=popt[1]
                usmatrix[row][column]=np.sqrt(np.diag(pcov))[1]
                #Assign 1 to fit type if it is a good fit (passed chi square test and parameters are positive)
                fittypematrix[row][column]=1

            
            ###Performing fit fixing E0, f, s
            #Fixing E0, f, s parameters in fitfunction
            def E0fsfixedfitfunction(x, A):
                return fitfunction(x, poptASIC[0], A, poptASIC[2], E0ASIC)

            try:
                poptfixed, pcovfixed=curve_fit(E0fsfixedfitfunction, thrs, fluxpixel, sigma=unc_fluxpixel, p0=p0[1])
            except RuntimeError:
                #If fit could not converge, assign -1 to fit type
                fixedfittypematrix[row][column]=-1
                continue
            
            #Setting expected values and calculating X2
            Efixed=E0fsfixedfitfunction(thrs, *poptfixed)
            X2fixed=chisquare(np.array(fluxpixel), Efixed, np.array(unc_fluxpixel))

            #Degrees of freedom for critical X2 (fitting one parameter, so -1)
            doffixed=len(fluxpixel)-1
            criticalX2fixed=scipy.stats.chi2.ppf(1-alpha, doffixed)
            
            if X2fixed<criticalX2fixed and X2fixed!=0:
                #Assign 1 to fit type if it is a good fit (passed chi square test)
                fixedfittypematrix[row][column]=1
            
        
        #Printing the time that analyzing the row took and estimated time (based on average time per row) to finish
        et=time.time()
        analysistime.append(et-st)
        print('Analyzing this row took {:.2f} seconds\nEstimated time to finish analyzing the data set is {:.2f} minutes'.format(et-st, sum(analysistime)/len(analysistime)*(255-row)/60))
    
    #Saving the eight matrices
    np.savetxt(FPTfolder+'E0matrix_ST_'+ST+'.csv', E0matrix, delimiter=',')
    np.savetxt(FPTfolder+'uE0matrix_ST_'+ST+'.csv', uE0matrix, delimiter=',')
    np.savetxt(FPTfolder+'fmatrix_ST_'+ST+'.csv', fmatrix, delimiter=',')
    np.savetxt(FPTfolder+'ufmatrix_ST_'+ST+'.csv', ufmatrix, delimiter=',')
    np.savetxt(FPTfolder+'smatrix_ST_'+ST+'.csv', smatrix, delimiter=',')
    np.savetxt(FPTfolder+'usmatrix_ST_'+ST+'.csv', usmatrix, delimiter=',')
    np.savetxt(FPTfolder+'Fittypematrix_ST_'+ST+'.csv', fittypematrix, delimiter=',')
    np.savetxt(FPTfolder+'Fixedfittypematrix_ST_'+ST+'.csv', fixedfittypematrix, delimiter=',')

    #Printing where the matrices have been saved to make it easier to find them
    print('\nE0matrix has been saved in {}'.format(FPTfolder+'E0matrix_ST_'+ST+'.csv'))
    print('\nuE0matrix has been saved in {}'.format(FPTfolder+'uE0matrix_ST_'+ST+'.csv'))
    print('\nfmatrix has been saved in {}'.format(FPTfolder+'fmatrix_ST_'+ST+'.csv'))
    print('\nufmatrix has been saved in {}'.format(FPTfolder+'ufmatrix_ST_'+ST+'.csv'))
    print('\nsmatrix has been saved in {}'.format(FPTfolder+'smatrix_ST_'+ST+'.csv'))
    print('\nusmatrix has been saved in {}'.format(FPTfolder+'usmatrix_ST_'+ST+'.csv'))
    print('\nFittypematrix has been saved in {}'.format(FPTfolder+'Fittypematrix_ST_'+ST+'.csv'))
    print('\nFixedfittypematrix has been saved in {}'.format(FPTfolder+'Fixedfittypematrix_ST_'+ST+'.csv'))
```
  
</details>

### 3. Data visualization
Finally, we wish to visualize the results, finding two types of visualizations needed. One function, ```pixelscanplot```, plots a scatter and line plot for the flux per threshold and the fitted model of a specified pixel in juxtaposition with the flux and model for the ASIC. The other, ```heatmapplot```, plots a colored 256x256 grid for the fitted values of the model parameters for each pixel and the fit types. All the visualizations are performed using ```matplotlib.pyplot```. See below for the code snippet.

<details>
<summary>Click to see the code snippet</summary>

```python
def pixelscanplot(FPTfolder, FPTfilename, ST, fromThr, toThr, stepThr, p0, rows, columns, alpha=0.05):
    '''
    Plots flux per threshold and its fits for the pixels specified in rows and columns.

    Arguments:
    - FPTfolder (str): Folder path to the Flux per Pixel per Threshold arrays files
    (e.g. ***).
    - FPTfilename (str): Path to the Flux per Pixel per Threshold arrays made by fluxpixelthr (needs importing)
    (e.g. ***).
    - ST (str): Shutter time (e.g. ***).
    - fromThr (int): Starting threshold of the scan (e.g. ***).
    - toThr (int): Ending threshold of the scan (e.g. ***).
    - stepThr (int): Increment of the threshold between scans (e.g. ***).
    - p0 (list of length 4): initial guess of the parameters to fit.
    - rows (list): rows of the pixels to be analyzed.
    - columns (list): columns of the pixels to be analyzed.    
    - alpha (float): Default=0.05. Level of significance.
    '''

    #Loading FluxperASIC first to find indices of bad thresholds
    fluxASIC=np.loadtxt(FPTfolder+***+ST+***+str(fromThr)+'-'+str(toThr)+***+str(stepThr)+'.csv', dtype=float, delimiter=',')

    #If for some thresholds, all acquisitions were bad (which returns nan) set a index to slice all pertinent data to leave the data after the bad thresholds
    #If all thresholds were good, nan_cut is set to 0, which doesn't slice the data
    if any(np.isnan(fluxASIC)):
        nan_cut=np.where(np.isnan(fluxASIC))[0][-1]+1
    else:
        nan_cut=0

    Thrs=list(range(fromThr, toThr+1, stepThr))[nan_cut:]
    x=np.linspace(fromThr, toThr, 2000) #For plotting the fits

    #FPT, flux per pixel per threshold. List of 256x256 arrays for each threshold
    FPT=[]
    unc_FPT=[]
    for thr in Thrs:
        f=np.loadtxt(FPTfolder+'Fluxperpixel_'+FPTfilename+ST+***+str(thr)+'.csv', dtype=float, delimiter=',')
        u=np.loadtxt(FPTfolder+'UncertaintyFluxperpixel_'+FPTfilename+ST+***+str(thr)+'.csv', dtype=float, delimiter=',')
        FPT.append(f)
        unc_FPT.append(u)

    fluxASIC=fluxASIC[nan_cut:]
    #Loading the uncertainty of the flux on the ASIC
    unc_fluxASIC=np.loadtxt(FPTfolder+***+FPTfilename+ST+***+str(fromThr)+'-'+str(toThr)+***+str(stepThr)+'.csv', dtype=float, delimiter=',')
    unc_fluxASIC=unc_fluxASIC[nan_cut:]
    thrsASIC=Thrs

    #Fitting the flux on the ASIC and calculating its X2
    poptASIC, pcovASIC=curve_fit(fitfunction, thrsASIC, fluxASIC, p0=p0)
    poptuncASIC=np.sqrt(np.diag(pcovASIC)) #Uncertainty in the fitted parameters
    X2ASIC=chisquare(fluxASIC, fitfunction(np.array(thrsASIC), *poptASIC), unc_fluxASIC)
    E0ASIC=poptASIC[-1] #Redefine for reading ease

    plt.figure() #Resets the figure if pixelscanplot is looped through
    #Picking the flux on the pixels per threshold specified by rows and columns. Trimming the zeros
    for i in range(len(rows)):
        #Creates list with flux on each pixel for each threshold
        fluxpixel=[j[rows[i]-1][columns[i]-1] for j in FPT]
        unc_fluxpixel=[j[rows[i]-1][columns[i]-1] for j in unc_FPT]

        #Trimming the left and right zeros in the flux data
        trimcut=np.nonzero(fluxpixel)
        thrs=Thrs[trimcut[0][0]:trimcut[0][-1]+1]
        fluxpixel=np.trim_zeros(fluxpixel)
        unc_fluxpixel=unc_fluxpixel[trimcut[0][0]:trimcut[0][-1]+1]
        if len(thrs)==0:
            continue

        
        #Predefining the labels. If not, the code will raise error for some pixels and data sets. The reason for the error is unknown
        labelfit=''
        labelfixedfit=''

        try:
            lock=1
            popt, pcov=curve_fit(fitfunction, thrs, fluxpixel, sigma=unc_fluxpixel, p0=p0)
        except RuntimeError:
            labelfit='Unable to find fit'
            lock=0 #If a fit is not found, don't compute X2 and don't plot fit

        #If a fit was found, calculate X2
        if lock==1:
            #Calculating X2 and critical X2. Fitting four parameters so -4 degrees of freedom
            X2=chisquare(np.array(fluxpixel), fitfunction(np.array(thrs), *popt), np.array(unc_fluxpixel))
            dof=len(fluxpixel)-4
            criticalX2=scipy.stats.chi2.ppf(1-alpha, dof)

            #Assigning different labels to the fitted line depending on the type of fit
            if X2>criticalX2:
                # labelfit='Bad fit on pixel {}x{}, fixed E0'.format(rows[i], columns[i])
                labelfit='Bad fit on pixel {}x{}'.format(rows[i], columns[i])
            elif X2<=criticalX2:
                poptunc=np.sqrt(np.diag(pcov)) #Uncertainty in the fitted parameters if a fit was found
                labelfit='Fit on pixel {}x{}\nE0={:.1f}$\pm${:.1f}, f={:.4f}$\pm${:.4f}, s={:.1f}$\pm${:.1f}'.format(rows[i], columns[i], popt[-1], poptunc[-1]+0.1, popt[0], poptunc[0]+0.0001, popt[2], poptunc[2]+0.1) #Additions are to round uncertainties up, hard coded



        ###Fit with E0, f and s fixed
        #Fixing E0, f and s
        def E0fsfixedfitfunction(x, A):
            return fitfunction(x, poptASIC[0], A, poptASIC[2], E0ASIC)
        
        #Fitting
        try:
            fixedlock=1
            poptfixed, pcovfixed=curve_fit(E0fsfixedfitfunction, thrs, fluxpixel, sigma=unc_fluxpixel, p0=p0[1])
        except RuntimeError:
            labelfixedfit='Unable to find fit fixing E0, f and s'
            fixedlock=0 #If a fit is not found, E cannot be calculated, so X2 will give error. Also locks the plotting of the fit

        #If a fit was found, calculate X2
        if fixedlock==1:
            #Calculating X2 and critical X2. Fitting one parameter, so -1 degree of freedom
            X2fixed=chisquare(np.array(fluxpixel), E0fsfixedfitfunction(np.array(thrs), *poptfixed), np.array(unc_fluxpixel))
            doffixed=len(fluxpixel)-1
            criticalX2fixed=scipy.stats.chi2.ppf(1-alpha, doffixed)

            #Assigning different labels to the fitted line depending on the type of fit
            if X2fixed>criticalX2fixed:
                labelfixedfit='Bad fit on pixel {}x{}, fixed E0, f and s'.format(rows[i], columns[i])
            elif X2fixed<=criticalX2fixed:
                labelfixedfit='Fit on pixel {}x{}, fixed E0, f and s'.format(rows[i], columns[i])


    ###Plotting
        #Iterating color, useful if len(rows)>1. Ranges between 0 and 1 to make a color spectrum for all the lines
        ci=float(i)/float(len(rows))
        #Plotting the flux on the pixel and its uncertainty
        plt.errorbar(thrs, fluxpixel, yerr=unc_fluxpixel, fmt='o', color=(0.8-0.8*ci, 0.4, 0.4-0.4*ci), label='Flux on pixel {}x{}'.format(rows[i], columns[i]))
        #Plotting the free fit, if found
        if lock==1:
            plt.plot(x, fitfunction(x, *popt), linestyle='-', linewidth=1.7, color=(1-ci, 0.4, 0.6-0.6*ci), label=labelfit)

        #Plotting the E0, f and s fixed fit, if found
        if fixedlock==1:
            plt.plot(x, E0fsfixedfitfunction(x, *poptfixed), linestyle='-', linewidth=1.7, color=(1-ci, 0.8, 1-ci), label=labelfixedfit)

    #Plotting the rescaled fluxes on the ASIC
    plt.errorbar(thrsASIC, fluxASIC/65352, yerr=unc_fluxASIC/65352, fmt='o', color=[0.1, 0.4, 0.1], label='Flux on average pixel')
    #Plotting the rescaled fit of the flux on the ASIC
    plt.plot(x, fitfunction(x, *poptASIC)/65352, color=[0.3, 0.4, 0.3], label='Fit on average pixel. \nE0={:.1f}$\pm${:.1f}, f={:.5f}$\pm${:.5f}, s={:.1f}$\pm${:.1f}'.format(E0ASIC, poptuncASIC[3]+0.1, poptASIC[0], poptuncASIC[0]+0.00001, poptASIC[2], poptuncASIC[2]+0.1)) #Additions are to round up uncertainties. Hard coded
    
    #Plot esthetics 
    plt.xlim(thrs[0]-2, thrs[-1]+2)
    #Setting different ymax limits if there is a bad or no fit present. Bad fits have big fluxes that shadow everything else in the plot
    #This might malfunction if more than one pixel is plotted. Max ylim is that of the highest flux of the last pixel analyzed
    if lock or fixedlock or X2>criticalX2 or X2fixed>criticalX2fixed:
        plt.ylim(0, max(fluxpixel)*1.1)
    else:
        plt.ylim(0, None)   
    plt.xlabel('Threshold (DAC)')
    plt.ylabel('Flux (hits/s)')
    plt.title('ST={} from {}'.format(ST, FPTfolder.split('/')[-2])) #To distinguish data sets

    plt.legend(fontsize=8)
    plt.savefig(***, format='png')
    print('\nThe figure has been saved in {}\n'.format(***))

def heatmapplot(filepath, ST, E0=True, f=True, s=True):
    '''
    Makes 256x256 heatmap plots of the fitted values of E0, f, s, free fit type and fixed fit type for each pixel. 


    Arguments:
    - filepath (str): the path of the folder to the E0 and X2 matrices (e.g. ***).
    - ST (str): shutter time of the data to be analyzed.
    - E0 (bool): Default=True. If True, creates a heatmap with the values of E0 from the free fit.
    - f (bool): Default=True. If True, creates a heatmap with the values of f from the free fit.
    - s (bool): Default=True. If True, creates a heatmap with the values of s from the free fit.
    '''
    
        
    #Loading the matrices obtained from Matricesgenerator.py using fitter
    E0=np.loadtxt(filepath+***+ST+'.csv', dtype=float, delimiter=',')
    uE0=np.loadtxt(filepath+***+ST+'.csv', dtype=float, delimiter=',')
    #Masking the pixels that didn't go through the fitting process, were unable to find a fit or found a bad fit (for all of these cases, the parameters are assigned a value of 0)
    E0=np.ma.masked_where(E0<=0, E0)
    uE0=np.ma.masked_where(E0<=0, uE0)

    f=np.loadtxt(filepath+***+ST+'.csv', dtype=float, delimiter=',')
    uf=np.loadtxt(filepath+***+ST+'.csv', dtype=float, delimiter=',')
    f=np.ma.masked_where(f<=0, f)
    uf=np.ma.masked_where(f<=0, uf)

    s=np.loadtxt(filepath+***+ST+'.csv', dtype=float, delimiter=',')
    us=np.loadtxt(filepath+***+ST+'.csv', dtype=float, delimiter=',')
    s=np.ma.masked_where(s<=0, s)
    us=np.ma.masked_where(s<=0, us)

    fittype=np.loadtxt(filepath+***+ST+'.csv', dtype=float, delimiter=',')

    fixedfittype=np.loadtxt(filepath+***+ST+'.csv', dtype=float, delimiter=',')

    #Calculating weighted average of parameters and the uncertainty of the average
    swE0, swf, sws=0, 0, 0
    swuE0, swuf, swus=0, 0, 0
    for row in range(256):
        for column in range(256):

            #Ignoring the masked elements (otherwise will return -- instead of a number)
            if np.ma.is_masked(E0[row][column]):
                continue
            
            #Calculating weights
            #if statements are made to eliminate uncertainties too close to 0. Otherwise, for the fourth data set,
            #average and uncertainty of the parameters will be NaN or inf. If statements may be removed for first to third data sets
            if uE0[row][column]>10**(-15):
                wuE0=1/uE0[row][column]**2
            else:
                wuE0=0
            if uf[row][column]>10**(-15):
                wuf=1/uf[row][column]**2
            else:
                wuf=0
            if us[row][column]>10**(-15):
                wus=1/us[row][column]**2
            else:
                wus=0

            #Calculating sum of weighted parameter 
            swE0+=E0[row][column]*wuE0
            swf+=f[row][column]*wuf
            sws+=s[row][column]*wus
            
            #Calculating sum of weights 
            swuE0+=wuE0
            swuf+=wuf
            swus+=wus
    
    #Finally, calculating weighted average parameter and its uncertainty
    E0avg, favg, savg=swE0/swuE0, swf/swuf, sws/swus
    uE0avg, ufavg, usavg=swuE0**(-0.5), swuf**(-0.5), swus**(-0.5)


    ###Plotting the E0, f and s heatmaps
    #E0 heatmap
    CMapp=plt.get_cmap('autumn')
    CMapp.set_bad([0.94, 0.94, 0.94])
    plt.figure()
    plt.pcolormesh(E0, cmap=CMapp, vmin=0.99*E0avg, vmax=1.01*E0avg) #Set limits to the colors, to better see the variance in the data

    plt.ylim(0, 256)
    plt.xlim(0, 256)
    ticks=np.linspace(0, 256, 5)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.title('E0 parameter, average={:.4f}$\pm${:.4f}'.format(E0avg, uE0avg), fontsize=20)
    plt.ylabel('Pixel rows', fontsize=15)
    plt.xlabel('Pixel columns', fontsize=15)
    plt.colorbar()

    plt.savefig(***, format='png')
    print('\nThe E0 heatmap has been saved in {}'.format(***))


    #f heatmap
    CMAp=plt.get_cmap('summer')
    CMAp.set_bad([0.94, 0.94, 0.94])    
    plt.figure()
    plt.pcolormesh(f, cmap=CMAp, vmin=0.4*favg, vmax=1.6*favg)

    plt.ylim(0, 256)
    plt.xlim(0, 256)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.title('f parameter, average={:.8f}$\pm${:.8f}'.format(favg, ufavg), fontsize=20)
    plt.ylabel('Pixel rows', fontsize=15)
    plt.xlabel('Pixel columns', fontsize=15)
    plt.colorbar()

    plt.savefig(***, format='png')
    print('\nThe f heatmap has been saved in {}'.format(***))

    #s heatmap
    CMAP=plt.get_cmap('winter')
    CMAP.set_bad([0.94, 0.94, 0.94])
    plt.figure()
    plt.pcolormesh(s, cmap=CMAP, vmin=0.7*savg, vmax=1.3*savg)

    plt.ylim(0, 256)
    plt.xlim(0, 256)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.title('s parameter, average={:.11f}$\pm${:.11f}'.format(savg, usavg), fontsize=20)
    plt.ylabel('Pixel rows', fontsize=15)
    plt.xlabel('Pixel columns', fontsize=15)
    plt.colorbar()

    plt.savefig(***, format='png')
    print('\nThe s heatmap has been saved in {}'.format(***.png'))



    ###Plotting the fit type matrices
    plt.figure()
    fig, (ax4, ax5)=plt.subplots(1, 2, figsize=(28, 10))

    Cmap=mpl.colors.ListedColormap([[0.94, 0.94, 0.94], [0.5, 0.5, 0.5], [0.9, 0.9, 0.7], [0.8, 0.4, 0.4]])

    cf4=ax4.pcolormesh(fittype, cmap=Cmap, vmin=-2, vmax=1)
    cf5=ax5.pcolormesh(fixedfittype, cmap=Cmap, vmin=-2, vmax=1)

    ax4.set_ylim(0, 256)
    ax4.set_xlim(0, 256)
    ax4.set_xticks(ticks)
    ax4.set_yticks(ticks)
    # ax4.set_title('Free fit')
    ax4.set_ylabel('Pixel rows')
    ax4.set_xlabel('Pixel columns')
    #Values for ticks are chosed by trial and error to fit in the middle of the corresponding color in colorbar
    fig.colorbar(cf4, ax=ax4, ticks=(-1.64, -0.88, -0.13, 0.62)).set_ticklabels(['Cut data', 'Fit not found', 'Bad fit', 'Good fit'])

    ax5.set_ylim(0, 256)
    ax5.set_xlim(0, 256)
    ax5.set_xticks(ticks)
    ax5.set_yticks(ticks)
    # ax5.set_title('E0, f and s fixed fit')
    ax5.set_ylabel('Pixel rows')
    ax5.set_xlabel('Pixel columns')
    fig.colorbar(cf5, ax=ax5, ticks=(-1.64, -0.88, -0.13, 0.62)).set_ticklabels(['Cut data', 'Fit not found', 'Bad fit', 'Good fit'])  

    plt.savefig(***, format='png')
    print('\nThe fit type heatmap has been saved in {}'.format(***))
```
  
</details>

#### Note
For the error analysis, the chi square was computed to determine the goodness of fit. You may find the code for this below.
<details>
<summary>Click to see the code snippet</summary>

```python
def chisquare(O, E, Oerr):
    '''
    Returns Chi square. If the uncertainty is close to 0 (raising error) or X2 is NaN, it assigns 0 to X2.

    Arguments:
    - O (array): observed frequencies. 
    - E (array): expected frequencies.
    - Oerr (array): uncertainty of the observed frequencies.
    '''
    O=O.flatten()
    E=E.flatten()
    Oerr=Oerr.flatten()
    X2=0
    for i in range(len(O)):
        try:
            X2+=(O[i]-E[i])**2/Oerr[i]**2
        except ZeroDivisionError:
            X2=0
            continue
    if np.isnan(X2):
        X2=0

    return X2
```
  
</details>

### 4. Results
Using the above functions, the analysis is performed and the results saved. Below you may find a sample of the visualizations, with the results censored for privacy.

<details>
<summary>Click to see the sample results</summary>

Scatter and line plot.
<img src="images/Sample result 1.png" />

Heatmaps. Fitted parameter values and fit type, respectively.
<img src="images/Sample result 2.png" />
<img src="images/Sample result 3.png" />

</details>
