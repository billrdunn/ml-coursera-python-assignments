from matplotlib import pyplot

def plotData(x,y):
    # open a new figure
    # fig = pyplot.figure()
    # ro = red circles, ms = marker size, mec = marker edge colour
    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.xlabel('Profit in $10,000')
    pyplot.ylabel('Population of city in 10,000s')
