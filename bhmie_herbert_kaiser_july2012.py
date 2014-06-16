from numpy import *

def bhmie(x,refrel,nang):
# This file is converted from mie.m, see http://atol.ucsd.edu/scatlib/index.htm
# Bohren and Huffman originally published the code in their book on light scattering

# Calculation based on Mie scattering theory  
# input:
#      x      - size parameter = k*radius = 2pi/lambda * radius   
#                   (lambda is the wavelength in the medium around the scatterers)
#      refrel - refraction index (n in complex form for example:  1.5+0.02*i;
#      nang   - number of angles for S1 and S2 function in range from 0 to pi/2
# output:
#        S1, S2 - funtion which correspond to the (complex) phase functions
#        Qext   - extinction efficiency
#        Qsca   - scattering efficiency 
#        Qback  - backscatter efficiency
#        gsca   - asymmetry parameter


    nmxx=150000

    s1_1=zeros(nang,dtype=complex128)
    s1_2=zeros(nang,dtype=complex128)
    s2_1=zeros(nang,dtype=complex128)
    s2_2=zeros(nang,dtype=complex128)
    pi=zeros(nang,dtype=complex128)
    tau=zeros(nang,dtype=complex128)

    if (nang > 1000):
        print ('error: nang > mxnang=1000 in bhmie')
        return

    # Require NANG>1 in order to calculate scattering intensities
    if (nang < 2):
        nang = 2

    pii = 4.*arctan(1.)
    dx = x

    drefrl = refrel
    y = x*drefrl
    ymod = abs(y)


    #    Series expansion terminated after NSTOP terms
    #    Logarithmic derivatives calculated from NMX on down

    xstop = x + 4.*x**0.3333 + 2.0
    #xstop = x + 4.*x**0.3333 + 10.0
    nmx = max(xstop,ymod) + 15.0
    nmx=fix(nmx)

    # BTD experiment 91/1/15: add one more term to series and compare resu<s
    #      NMX=AMAX1(XSTOP,YMOD)+16
    # test: compute 7001 wavelen>hs between .0001 and 1000 micron
    # for a=1.0micron SiC grain.  When NMX increased by 1, only a single
    # computed number changed (out of 4*7001) and it only changed by 1/8387
    # conclusion: we are indeed retaining enough terms in series!

    nstop = int(xstop)

    if (nmx > nmxx):
        print ( "error: nmx > nmxx=%f for |m|x=%f" % ( nmxx, ymod) )
        return

    dang = .5*pii/ (nang-1)


    amu=arange(0.0,nang,1)
    amu=cos(amu*dang)

    pi0=zeros(nang,dtype=complex128)
    pi1=ones(nang,dtype=complex128)

    # Logarithmic derivative D(J) calculated by downward recurrence
    # beginning with initial value (0.,0.) at J=NMX

    nn = int(nmx)-1
    d=zeros(nn+1,dtype=complex128)
    for n in range(0,nn):
        en = nmx - n
        d[nn-n-1] = (en/y) - (1./ (d[nn-n]+en/y))


    #*** Riccati-Bessel functions with real argument X
    #    calculated by upward recurrence

    psi0 = cos(dx)
    psi1 = sin(dx)
    chi0 = -sin(dx)
    chi1 = cos(dx)
    xi1 = psi1-chi1*1j
    qsca = 0.
    gsca = 0.
    p = -1

    for n in range(0,nstop):
        en = n+1.0
        fn = (2.*en+1.)/(en* (en+1.))

    # for given N, PSI  = psi_n        CHI  = chi_n
    #              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
    #              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
    # Calculate psi_n and chi_n
        psi = (2.*en-1.)*psi1/dx - psi0
        chi = (2.*en-1.)*chi1/dx - chi0
        xi = psi-chi*1j

    #*** Store previous values of AN and BN for use
    #    in computation of g=<cos(theta)>
        if (n > 0):
            an1 = an
            bn1 = bn

    #*** Compute AN and BN:
        an = (d[n]/drefrl+en/dx)*psi - psi1
        an = an/ ((d[n]/drefrl+en/dx)*xi-xi1)
        bn = (drefrl*d[n]+en/dx)*psi - psi1
        bn = bn/ ((drefrl*d[n]+en/dx)*xi-xi1)

    #*** Augment sums for Qsca and g=<cos(theta)>
        qsca += (2.*en+1.)* (abs(an)**2+abs(bn)**2)
        gsca += ((2.*en+1.)/ (en* (en+1.)))*( real(an)* real(bn)+imag(an)*imag(bn))

        if (n > 0):
            gsca += ((en-1.)* (en+1.)/en)*( real(an1)* real(an)+imag(an1)*imag(an)+real(bn1)* real(bn)+imag(bn1)*imag(bn))


    #*** Now calculate scattering intensity pattern
    #    First do angles from 0 to 90
        pi=0+pi1    # 0+pi1 because we want a hard copy of the values
        tau=en*amu*pi-(en+1.)*pi0
        s1_1 += fn* (an*pi+bn*tau)
        s2_1 += fn* (an*tau+bn*pi)

    #*** Now do angles greater than 90 using PI and TAU from
    #    angles less than 90.
    #    P=1 for N=1,3,...% P=-1 for N=2,4,...
    #   remember that we have to reverse the order of the elements
    #   of the second part of s1 and s2 after the calculation
        p = -p
        s1_2+= fn*p* (an*pi-bn*tau)
        s2_2+= fn*p* (bn*pi-an*tau)

        psi0 = psi1
        psi1 = psi
        chi0 = chi1
        chi1 = chi
        xi1 = psi1-chi1*1j

    #*** Compute pi_n for next value of n
    #    For each angle J, compute pi_n+1
    #    from PI = pi_n , PI0 = pi_n-1
        pi1 = ((2.*en+1.)*amu*pi- (en+1.)*pi0)/ en
        pi0 = 0+pi   # 0+pi because we want a hard copy of the values

    #*** Have summed sufficient terms.
    #    Now compute QSCA,QEXT,QBACK,and GSCA

    #   we have to reverse the order of the elements of the second part of s1 and s2
    s1=concatenate((s1_1,s1_2[-2::-1]))
    s2=concatenate((s2_1,s2_2[-2::-1]))
    gsca = 2.*gsca/qsca
    qsca = (2./ (dx*dx))*qsca
    qext = (4./ (dx*dx))* real(s1[0])

    # more common definition of the backscattering efficiency,
    # so that the backscattering cross section really
    # has dimension of length squared
    qback = 4*(abs(s1[2*nang-2])/dx)**2
    #qback = ((abs(s1[2*nang-2])/dx)**2 )/pii  #old form

    return s1,s2,qext,qsca,qback,gsca

