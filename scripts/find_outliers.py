""" Python script to find outliers

Run as:

    python3 scripts/find_outliers.py data
"""

import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import stats

def find_outliers(data_directory):
    """ Print filenames and outlier indices for images in `data_directory`.

    Print filenames and detected outlier indices to the terminal.

    Parameters
    ----------
    data_directory : str
        Directory containing containing images.

    Returns
    -------
    None
    """


    # Identify image files in data_directory using names listed in
    # 'hash_list.txt' (executed similarly to validate_data.py)
    hashlist = str(data_directory + '/' + 'hash_list.txt')

    #iterating through to add to an array which contains the values for each volume for each files
    k = 0
    vol_means = {}
    volumes = {}
    imgfiles = {}

    for line in open(hashlist, 'rt'):
        i = line.split()
        imgfile = str(data_directory + '/' + i[1]) #making this filename a string with full path
        # Load the image file
        img = nib.load(imgfile, mmap=False)
        # Retrieve data from image array
        data = img.get_data()

        volumes[str(k)] = data
        #take the mean of each volume and put into a column in the vol_means array
        vol_means[str(k)] = np.mean(data, axis = (0,1,2))
        #storing filenames in a dictionary
        imgfiles[str(k)] = i[1]
        k+=1

    # print(volumes[str(1)].shape)  #making sure this gives the shape of 1 run
    #plot with subplots the means for each volume for each run
    """fig, ax = plt.subplots(20,1)
    for i, ax in enumerate(ax):
        ax.plot(vol_means[str(i)])
        ax.set_ylabel('Run ' + str(i))
        ax.set_xlim(0,162)
        print(len(vol_means[str(i)]))
    plt.show()"""

    """iqr_proportion = 1.5
    for line in vol_means:
        q1, q3 = np.percentile(vol_means[line], [25, 75])
        iqr = q3 - q1
        up_thresh = q3 + iqr * iqr_proportion
        down_thresh = q1 - iqr * iqr_proportion
        print(np.logical_or(vol_means[line] > up_thresh, vol_means[line] < down_thresh))"""

    #making a residuals dictionary that contains a more linear version
    # of the mean data. It gets rid of any drift in the data.

    """residuals = {}
    for i in range(len(vol_means)):
        data = vol_means[str(i)]
        shape = data.shape[0]  # number of volumes
        X = np.ones((shape, 2))  # make a 2 column  vector containing ones
        drift = np.linspace(-1, 1, shape) # make an array of numbers stretching from -1 to 1
        X[: , 1] = drift # puts the values of drift into the second column of X
        beta = np.linalg.pinv(X).dot(data) # numpy function to give values of intercept and slope of best fitting line
        fitted = X.dot(beta)  #creating the best fitted line
        residual = data - fitted  #difference between mean array and the fitted array
        residuals[str(i)] = residual + beta[0]  #adds the intercept to show the fitted version against the original mean value version

        #print(np.corrcoef(drift, residual))
        #print(beta)
        #plt.plot(fitted)
        #plt.show()

        #plt.plot(vol_means[str(6)]) #looking at one of the mean plots
        #plt.plot(residuals[str(6)]) #looking at one of the residual plots
        #plt.show()"""


    #adding a section to select values from ONLY outside of the brain
    #Did this by looking at the values from a bunch of the volumes for different runs,
    #and finding that almost all values inside of (volume_mean + 0.3*mean) were inside the brain
    #we can then run dvars on these values so we are not getting the variablity of voxels inside the brain
    volumes_outside_brain = {}

    for i in range(len(volumes)):
        #selecting a given run of volumes
        run = volumes[str(i)]
        num_vols = volumes[str(i)].shape[-1]
        outside_brain = np.zeros(volumes[str(i)].shape)
        #now looping through each volume in the given run
        for k in range(num_vols):
            vol = run[...,k]
            #lowvals = vol < (np.mean(vol) + 0.3*np.mean(vol))
            #selected = vol[lowvals]
            #outside_brain.append(selected)
            lowvals = stats.threshold(vol, threshmax = np.mean(vol) + 0.3*np.mean(vol), newval = 0)
            outside_brain[...,k] = lowvals
        volumes_outside_brain[str(i)] = outside_brain
        print('Run ' + str(i) + ':' + str(volumes_outside_brain[str(i)].shape))

    #calculating the Dvars of the volumes for the area outside of the brain
    dvars_outside_brain = {}

    for i in range(len(volumes_outside_brain)):
        nvoxels = volumes_outside_brain[str(i)].shape[0] * volumes_outside_brain[str(i)].shape[1] * volumes_outside_brain[str(i)].shape[2]
        num_vols = volumes_outside_brain[str(i)].shape[-1]
        dvarsFirst = []
        for k in range(num_vols - 1):
            data = volumes_outside_brain[str(i)]
            diffs = data[...,k] - data[...,k + 1]
            diffs = diffs**2 # Square the differences;
            sumdiffs = sum(diffs.ravel())
            avgdiffs = sumdiffs/nvoxels #divide by number items
            sqdiffs = np.sqrt(avgdiffs) # Return the square root of these values.
            dvarsFirst.append(sqdiffs)
        dvars_outside_brain[str(i)] = dvarsFirst
        print('dvars ' + str(i) + ':' + str(len(dvars_outside_brain[str(i)])))

#Select and print outliers from dvars that are 2.5 s.d. or more away from mean
    outliers = {}
    for i in range(len(dvars_outside_brain)):
        num_vols = len(dvars_outside_brain[str(i)])
        dvars_list = dvars_outside_brain[str(i)]
        vol_outliers = []
        for k in range(num_vols):
            if dvars_list[k] > (np.mean(dvars_list) + 2.5*np.std(dvars_list)):
                vol_outliers.append(k)
            elif dvars_list[k] < (np.mean(dvars_list) - 2.5*np.std(dvars_list)):
                vol_outliers.append(k)
        outliers[str(i)] = vol_outliers

#MLN 10-9-16: the following commented code attempted to make detection more specific,
# since it is already quite sensitive, but failed to use the right parameters for
# criteria, so it does not ultimately improve final list of outliers
        #dvars_array = np.array(dvars_list)
        #returns = dvars_array[1:num_vols]/dvars_array[0:(num_vols - 1)] - 1.0
        #np.insert(returns, 0, 0.0)
        #(returns_outliers,) = np.where(np.abs(returns) >= 0.25)
        #outlier_inds = np.intersect1d(vol_outliers, returns_outliers)
        #outliers[str(i)] = outlier_inds

# Print each file name and then list each of the volumes that were determined
# to be outliers
    for i in range(len(outliers)):
        print(str(imgfiles[str(i)]) + ' outliers:' + str(outliers[str(i)]))
    # Plot
    fig, ax = plt.subplots(20,1)
    for i, ax in enumerate(ax):
        dvars_run = dvars_outside_brain[str(i)]
        ax.plot(dvars_run)
        xs = outliers[str(i)]
        #ys = dvars_run[outliers[str(i)]]
        ys = [dvars_run[o] for o in outliers[str(i)]]
        ax.scatter(xs, ys, edgecolor = None, color = 'r')
        ax.set_ylabel('Run ' + str(i))
        ax.set_xlim(0,162)
    plt.show()

    raise RuntimeError('No code yet')


def main():
    # This function (main) called when this file run as a script.
    #
    # Get the data directory from the command line arguments
    if len(sys.argv) < 2:
        raise RuntimeError("Please give data directory on "
                           "command line")
    data_directory = sys.argv[1]
    # Call function to validate data in data directory
    find_outliers(data_directory)

if __name__ == '__main__':
    # Python is running this file as a script, not importing it.
    main()
