
#Lowell Schudel 6/3/14

def rht(xy, wlen, smr, frac=0.70):

	'''
	xy = 2D Array Image
	wlen = 101.0 #Window diameter
	smr = 11.0 #Smoothing radius
    frac = 0.70 #Theta-power threshold to store

	Returns a 3D array of accumulated weights in x-y-theta space 
	This is not optimal, but represents what I think is going on
	#TODO
	'''
	#Parse variables
	xlen, ylen = xy.shape

	#Perform checks
	wlen = int(wlen)
	if (smr > xlen) or (smr > ylen) or (wlen > xlen) or (wlen > ylen):
		print 'Invalid Parameters'
		exit()

	#Import functions
	from numpy import zeros, pi, arange, inf, isinf
	from math import hypot, sqrt
	from __future__ import division
	
	#Begin unmask
	xy_blur = [[(0,0)]*ylen]*xlen
	for i in range(xlen):
		for j in range(ylen):
			left = max(0, i-smr)
			right = min(xlen-1, i+smr)
			for x in range(left, right):
				#TODO integerize this
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
	#xy_blur = map(blur, xy_blur) TODO

	#Subtract blur
	xy_sharp = subtract(xy, xy_blur)
	def sharpen(val):
		if val > 0:
			return 1
		else:
			return 0
	for i in range(xlen):
		for j in range(ylen):
			xy_sharp[i][j] = sharpen(xy_sharp[i][j])
	#xy_sharp = map(sharpen, xy_sharp) TODO

	#Count theta bins
	slope_set = []
	for i in range(1, xlen):
		for j in range(1, ylen):
			slope_set.append(j/i)
	ntheta = 1 + 2*len(set(slope_set)) #see all_slopes below
	dtheta = pi/ntheta
    theta = arange(0, pi, dtheta)

    #Completes all_slopes w/o using trig function calls
    slope_list = sorted(slope_set)
    all_slopes = slope_list
    all_slopes.append(inf)
    slope_list.reverse()
    all_slopes += slope_list
    #all_slopes is now equivalent to tan(theta) from [0, pi), and has same length
	if __debug__:
		if len(all_slopes) != len(theta):
			print 'Ntheta not correct.'
		from numpy import tan, fabs
		diff= fabs(subtract(tan(all_slopes)-theta))
		print diff
		print 'Worst:', max(diff) 

	#Define local coordinates
	def x_prime(l):
		#l is an integer x-step in pixel space
		if isinf(l):
			return 0
		else:
			return int(l)
	def y_prime(l, s):
		#l is an integer x-step in pixel space
		#s is the slope desired
		if isinf(l):
			return l
		else:
			return s*x_prime(l)
	def r_prime(l, s):
		#l is an integer x-step in pixel space
		#s is the slope desired
		return hypot(x_prime(l) , y_prime(l, s))

    #Prepares accumulator array and a helper function
    xyt = [[zeros(ntheta)]*ylen]*xlen #3D array of theta power
    '''
    def accumulate(i, j, slope):
    	#Accepts accumulation from integer-valued coordinates within the image only
    	#Returns number of bins counted
		
		return
	'''
	
    #Accumulate hits into xyt array
    for i in range(xlen):
		for j in range(ylen):
			for slope in all_slopes:
				
				n=1
				while (r_prime(n, slope) <= wlen/2.0):
					#Searches in +x_prime direction within the image
					#Accepts accumulation from integer-valued coordinates only
					y = j + y_prime(n, slope)
					if (y.is_integer()):
						x = i + x_prime(n)
						if (0 <= x and x < xlen) and (0 <= y and y < ylen):
							xyt[i][j][all_slopes.index(slope)] += xy_sharp[x][y]
						else:
							#Stops at image edge, recording that no more bins were counted
							n -= 1
							break
					n += 1	

				m=1
				while (r_prime(-m, slope) <= wlen/2.0):
					#Searches in -x_prime direction within the image
					#Accepts accumulation from integer-valued coordinates only
					y = j + y_prime(-m, slope)
					if (y.is_integer()):
						x = i + x_prime(-m)
						if (0 <= x and x < xlen) and (0 <= y and y < ylen):
							xyt[i][j][all_slopes.index(slope)] += xy_sharp[x][y]
						else:
							#Stops at image edge, recording that no more bins were counted
							m -= 1
							break
					m += 1


				xyt[i][j][all_slopes.index(slope)] /= (n+m) #Accounts for number of bins counted

					


	#TODO I have to keep working here
	return xyt
