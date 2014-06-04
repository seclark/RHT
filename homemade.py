
#Lowell Schudel 6/3/14

def rht(xy, smr, wlen):
	'''
	Returns a 3D array of accumulated weights in x-y-theta space
	'''

	#Parse variables
	xlen, ylen = xy.shape

	#Perform checks
	wlen = int(wlen)
	if (smr > xlen) or (smr > ylen) or (wlen > xlen) or (wlen > ylen):
		print 'Invalid Parameters'
		exit()

	#Import functions
	from numpy import zeros
	from math import hypot, sqrt
	
	#Begin unmask
	xy_blur = [[(0,0)]*ylen]*xlen
	for i in range(xlen):
		for j in range(ylen):
			left = max(0, i-smr)
			right = min(xlen-1, i+smr)
			for x in range(left, right):
				up = max(0, j-sqrt(smr**2 - (x-i)**2))
				down = min(ylen-1, j+sqrt(smr**2 - (x-i)**2))
				for y in range(up, down):
					xy_blur[i][j][0] += xy[x][y]
					xy_blur[i][j][1] += 1

	#Scale accumulators
	def blur(pair):
		return pair[0]/pair[1]
	for i in range(xlen):
		for j in range(ylen):
			xy_blur[i][j] = blur(xy_blur[i][j])

	#Subtract blur
	xy_sharp = subtract(xy, xy_blur)
	def contrast(val):
		if val > 0:
			return 1
		else:
			return 0
	for i in range(xlen):
		for j in range(ylen):
			xy_sharp[i][j] = contrast(xy_sharp[i][j])

	print xy_sharp

