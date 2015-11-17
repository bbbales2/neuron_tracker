import numpy
import matplotlib.pyplot as plt
import math
import os
import scipy
import scipy.ndimage.filters
import scipy.ndimage.morphology
import skimage.io, skimage.util, skimage.filters, skimage.transform, skimage.feature, skimage.exposure
import sys
import networkx

def detect_local_minima(arr):
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = scipy.ndimage.morphology.generate_binary_structure(len(arr.shape),3)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (scipy.ndimage.filters.minimum_filter(arr, footprint = neighborhood)==arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr==0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = scipy.ndimage.morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min - eroded_background
    return numpy.where(detected_minima)

def rescale(signal, minimum, maximum):
    mins = numpy.min(signal.flatten())
    maxs = numpy.max(signal.flatten())

    output = numpy.array(signal).astype('float')

    output -= mins
    output *= (maximum - minimum) / (maxs - mins)
    output += minimum

    #output = (maximum - minimum) * (signal - mins) / (maxs - mins) + minimum

    return output

colors = ['blue', 'red', 'green', 'yellow', 'black']

TS = 0
number = "scn1"

#data0 = skimage.io.imread('/home/bbales2/brains/raw_images/{0}/Before TTX.gif'.format(number))
#data1 = skimage.io.imread('/home/bbales2/brains/raw_images/{0}/During TTX.gif'.format(number))
data2 = skimage.io.imread('After TTX.tif'.format(number))

#data0 = rescale(data0, 0.0, 1.0)
#data1 = rescale(data1, 0.0, 1.0)
data2 = rescale(data2, 0.0, 1.0)
#data = numpy.concatenate((data0, data1, data2))
#%%

# This function goes through all the images and finds neurons in each of them
def process_data(data):
    blobLists = []
    trees = []
    ims = []
    ids = {}

    T = data.shape[0]

    for t in range(0, T):
        im = data[t]

        ims.append(im)

        im = rescale(im, 0.0, 1.0)

        # This plots the rescaled image
        #plt.imshow(im, cmap = plt.cm.gray)
        #fig = plt.gcf()
        #fig.set_size_inches(7.5, 7.5)
        #plt.show()

        fltrd = scipy.ndimage.filters.gaussian_filter(-im, 2.0) - scipy.ndimage.filters.gaussian_filter(-im, 20.0)
        fltrd = rescale(fltrd, 0.0, 1.0)

        # This plots the thing we're searching for mins on
        #plt.imshow(1.0 - fltrd, cmap = plt.cm.gray)
        #fig = plt.gcf()
        #fig.set_size_inches(7.5, 7.5)
        #plt.show()

        locations = zip(*detect_local_minima(fltrd))
        mags = []

        blurredIm = scipy.ndimage.filters.gaussian_filter(im, 2.0)

        for i, j in locations:
            mags.append(blurredIm[i, j])

        indices = numpy.argsort(mags)

        xys = []
        for i in indices[-200:]:
            xys.append(locations[i])

        if len(xys) == 0:
            xys.append((-500, -500))

        for x, y in xys:
            ids[(t, (x, y))] = len(ids)

        trees.append(scipy.spatial.KDTree(xys))

        blobLists.append(xys)

        # This plots the image with the dots on it
        #plt.imshow(im, cmap = plt.cm.gray)
        #for i, j in xys:
        #    y, x = i, j
        #    c = plt.Circle((x, y), 2, color = 'yellow', linewidth = 0, fill = True)
        #    ax = plt.gca()
        #    ax.add_patch(c)
        #fig = plt.gcf()
        #fig.set_size_inches(7.5, 7.5)
        #plt.show()

        if t % (T / 20) == 0:
            print len(xys)
            print t / float(T) * 100.0

    return ims, ids, trees, blobLists

# This function connects the neurons between images in a timeseries
# Each connection through a series of images is represented by a graph
def build_subgraphs(blobLists, trees, ids):
    T = len(blobLists)
    g = networkx.Graph()

    for t in range(1, T):
        blobsList = blobLists[t]

        if t % (T / 20) == 0:
            print 100.0 * t / float(T)

        for blob in blobsList:
            for bt in range(t - 1, max(-1, t - 50), -1):
                distance, neighbor = trees[bt].query(blob, 1)

                if distance < 1.5 + 0.125 * (t - bt):
                    g.add_edge(ids[(bt, tuple(trees[bt].data[neighbor]))], ids[(t, blob)])
                    break

    subgraphs = list(networkx.connected_components(g))

    return g, subgraphs

# The way we connected neurons above, it's possible that the extracted graphs
# have more than one unique path from beginning to end. Here, we enforce
# that there can only be one
def prune_subgraphs(g, subgraphs, ids):
    reverseIds = {}
    for t, c in ids:
        reverseIds[ids[(t, c)]] = (t, c)

    subgraphs2 = []
    for i, nodes in enumerate(subgraphs):
        #print len(subgraph)

        subgraph = g.subgraph(nodes)

        stuff = []
        for node in nodes:
            stuff.append((reverseIds[node], node))

        stuff2 = {}
        for thing in stuff:
            t = thing[0][0]

            if t not in stuff2:
                stuff2[t] = []

            stuff2[t].append(thing)

        times = sorted(list(set(stuff2)))

        path = networkx.shortest_path(subgraph, stuff2[times[0]][0][1], stuff2[times[-1]][0][1])

        if len(path) > 0:
            subgraphs2.append(path)

    return subgraphs2, reverseIds

# This simply takes the trajectories that we created,
#   Checks if they're long enough to be interesting
#   Extrapolates the neuron sampling to t = 0 and t = tfinal
#   Interpolates neuron positions on frames where we don't know the exact location
#   And samples the actual data
def interpolate_stuff(ims, subgraphs2, ids, reverseIds):
    T = len(ims)
    Bn = 15
    blur = numpy.zeros((Bn, Bn))

    blur[Bn / 2, Bn / 2] = 1

    blur = scipy.ndimage.filters.gaussian_filter(blur, 2.0)
    blur /= sum(blur.flatten())

    reverseIds = {}
    for t, c in ids:
        reverseIds[ids[(t, c)]] = (t, c)

    #plt.imshow(ims[10], cmap = plt.cm.gray)
    #plt.colorbar()
    #fig = plt.gcf()
    #fig.set_size_inches(15,12)

    nodes = {}
    lines = []
    com = []
    traj = []
    for subgraph in subgraphs2:
        xs = []
        ys = []

        # If we have T images, we need to have identified neurons in at least
        # 8% of those images to believe that they are real
        if len(subgraph) > 0.08 * T:
            interpolated = {}
            interpolatedV = {}

            cancel = False
            for node in subgraph:
                t, c = reverseIds[node]
                xs.append(c[0])
                ys.append(c[1])
                interpolated[t] = c
                try:
                    interpolatedV[t] = sum((ims[t][c[0] - Bn / 2 : c[0] + Bn / 2 + 1, c[1] - Bn / 2 : c[1] + Bn / 2 + 1] * blur).flatten())
                except Exception as e:
                    # This exception gets thrown for out of bounds stuff I think -- if the trajectory is at the edge of the image we ignore it
                    cancel = True

            if cancel:
                continue

            interpolatedTimes = sorted(interpolated.keys())

            originalInterpolatedTimes = numpy.array(sorted(interpolated.keys()))

            for t in range(0, interpolatedTimes[0]):
                interpolated[t] = interpolated[interpolatedTimes[0]]
                c = interpolated[t]
                interpolatedV[t] = sum((ims[t][c[0] - Bn / 2 : c[0] + Bn / 2 + 1, c[1] - Bn / 2 : c[1] + Bn / 2 + 1] * blur).flatten())

            for t in range(interpolatedTimes[-1] + 1, T):
                interpolated[t] = interpolated[interpolatedTimes[-1]]
                c = interpolated[t]
                interpolatedV[t] = sum((ims[t][c[0] - Bn / 2 : c[0] + Bn / 2 + 1, c[1] - Bn / 2 : c[1] + Bn / 2 + 1] * blur).flatten())

            for i in range(len(interpolatedTimes) - 1):
                lt = interpolatedTimes[i]
                rt = interpolatedTimes[i + 1]

                if rt == lt + 1:
                    continue

                lc = numpy.array(interpolated[lt])
                rc = numpy.array(interpolated[rt])

                lv = numpy.array(interpolatedV[lt])
                rv = numpy.array(interpolatedV[rt])
                for t in range(lt + 1, rt):
                    a = float(t - lt) / (rt - lt)

                    interpolated[t] = tuple(lc * (1.0 - a) + rc * a)
                    c = numpy.round(numpy.array(interpolated[t])).astype('int')
                    interpolatedV[t] = sum((ims[int(numpy.round((rt - lt) * a + lt))][c[0] - Bn / 2 : c[0] + Bn / 2 + 1, c[1] - Bn / 2 : c[1] + Bn / 2 + 1] * blur).flatten())

            for t, c in interpolated.items():
                if t not in nodes:
                    nodes[t] = []

                nodes[t].append(c)

            interpolated = numpy.array([indice for time, indice in sorted(interpolated.items(), key = lambda x : x[0])])

            tra = numpy.array(interpolated)
            for t in range(1, len(tra)):
                distance = numpy.sqrt((tra[t][0] - tra[t - 1][0])**2 + (tra[t][1] - tra[t - 1][1])**2)

                # This looks like a safety check to make sure our interpolation didn't do anything silly
                if distance > 5.0:
                    print originalInterpolatedTimes
                    print tra[t], tra[t - 1]#, tra[t:]
                    print i, t, t - 1
                    print 'ERROR'
                    1/0
                    continue

            com.append(numpy.mean(interpolated, axis = 0))
            traj.append(numpy.array(interpolated))
            interpolated = numpy.array(sorted(interpolatedV.items(), key = lambda x : x[0]))

            lines.append(interpolated)

    return numpy.array(com), numpy.array(traj), numpy.array(lines)
#%%
ims, ids, trees, blobLists = process_data(data2)
graph, subgraphs = build_subgraphs(blobLists, trees, ids)
subgraphs2, reverseIds = prune_subgraphs(graph, subgraphs, ids)
com, traj, lines = interpolate_stuff(ims, subgraphs2, ids, reverseIds)

for line in lines:
    xs, ys = zip(*line)
    ys = numpy.array(ys)
    plt.plot(xs, ys)
fig = plt.gcf()
fig.set_size_inches(15,15)
plt.show()
print len(lines)
com = numpy.array(com)
plt.imshow(ims[5])
plt.plot(com[:, 1], com[:, 0], '*')
plt.show()
