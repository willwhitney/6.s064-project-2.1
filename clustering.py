import numpy as np

def distanceSquared(a, b):
    return sum((b - a) * (b-a))

def distance(a, b):
    return np.linalg.norm((b-a))

def kMeans(X, init):
    numPoints, numFeatures = X.shape
    k = init.shape[0]
    means = init
    assignments = []
    while True:
        newAssignments = []
        for point in X:
            best = -1
            bestDistance = None
            for i, mean in enumerate(means):
                if best < 0 or distanceSquared(point, mean) < bestDistance:
                    best = i
                    bestDistance = distanceSquared(point, mean)
            newAssignments.append(best)

        # print newAssignments

        newMeans = np.array([[0.] * numFeatures] * k)
        clusterLengths = [0] * k
        for i, point in enumerate(X):
            newMeans[newAssignments[i]] += point
            clusterLengths[newAssignments[i]] += 1

        for i, mean in enumerate(newMeans):
            newMeans[i] = newMeans[i] / float(clusterLengths[i])

        if newAssignments == assignments:
            break

        means = newMeans
        assignments = newAssignments

    return (means, assignments)

def kMeansInit(X, k, p0):
    centroids = [p0]

    while len(centroids) < k:
        options = []
        for point in X:
            distSum = 0
            for centroid in centroids:
                distSum += distance(point, centroid)
            options += [(distSum, list(point))]
        # print options
        options.sort()
        centroids.append(np.array(options[-1][1]))

    return np.array(centroids)

def pca(X):
    numPoints, numFeatures = X.shape
    dim = min(numPoints, numFeatures)

    mean = np.array([0.0] * numFeatures)
    for p in X:
        mean += p
    mean = mean / float(numPoints)

    covar = np.array([[0.0] * dim] * dim)
    for p in X:
        covar += np.transpose(np.asmatrix(p - mean)) * (np.asmatrix(p - mean))

    covar = covar / float(numPoints)
    l, v = np.linalg.eig(covar)
    v = v.transpose()

    pairs = []
    for i in range(len(l)):
        pairs.append( (l[i], list(v[i])) )
    pairs.sort()
    pairs.reverse()

    filteredV = []
    i = 0
    while i < len(pairs) and pairs[i][0] > 1.0e-10:
        filteredV.append(pairs[i][1])
        i += 1
    return np.array(filteredV).transpose()

def pcahd(X):
    numPoints, numFeatures = X.shape
    dim = min(numPoints, numFeatures)

    mean = np.array([0.0] * numFeatures)
    for p in X:
        mean += p
    mean = mean / float(numPoints)

    D = np.zeros(X.shape)
    for i in range(len(X)):
        D[i] = X[i] - mean

    Dt = np.transpose(D)
    DDt = np.asmatrix(D) * np.asmatrix(Dt)
    l, v = np.linalg.eig(np.array(DDt))
    # print l
    # print v

    v = np.transpose(v)
    pairs = []
    for i in range(len(l)):
        pairs.append( (l[i], list(v[i])) )
    pairs.sort()
    pairs.reverse()

    result = []
    for pair in pairs:
        # print pair[1]
        result.append(pair[1])

    return np.transpose(np.array(result[:2]))

X2 = np.array([[ -2.,  -2., 0., 1.], [ -1.,  -1., 1., 2.], [ -0.01,  -0.01, 2., 3.]])
print pcahd(X2)
# init1 = np.array([[1, 1, 2], [1, -1, 2]])

# print kMeans(X1, init1)













