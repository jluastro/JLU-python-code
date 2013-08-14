# Makefile for slalib

CC = gcc
FC = gfortran

refco:
	f2py --f77exec=$(FC) -c refco.pyf refco.f refro.f atms.f atmt.f drange.f

refro:
	f2py --f77exec=$(FC) -c refro.pyf refro.f atms.f atmt.f drange.f


