from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import astropy
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import Angle, ICRS

import time

def extract_data():
	"""Extract relevant columns from the pandas catalog"""
	
	t = np.loadtxt("../data/pandas.txt", usecols=(0,1,2,3,4,5,11,14,23,25,26,27,28,29,30,31,32))
	
	table = Table(t, names=('rah', 'ram', 'ras', 'decd', 'decm', 'decs', 'gflag', 'iflag', 'ebv', 'g', 'gerr', 'gchi', 'gsharp', 'i', 'ierr', 'ichi', 'isharp'))
	table.write('../data/extract.txt', format='ascii.commented_header')

def reformat_catalog(coordinates=True, chi=True, sharp=True, bright=True):
	"""Format RA, Dec as floats in the final catalog"""
	
	t = Table.read('../data/extract.txt', format='ascii.commented_header')
	
	if coordinates:
		ra = Angle((t['rah'], t['ram'], t['ras']), unit=u.hour)
		dec = Angle((t['decd'], t['decm'], t['decs']), unit=u.deg)
		
		t.remove_columns(['rah', 'ram', 'ras', 'decd', 'decm', 'decs'])
		add_npcolumn(t, 'ra', ra.to(u.deg), index=0)
		add_npcolumn(t, 'dec', dec, index=1)
	
	if bright:
		ind = ((t['g']<18) | (t['i']<18)) & (t['ichi']<90)
		t_bright = t[ind]
		t = t[~ind]
		
	if chi:
		#ind = (t['gchi']<1.4) & (t['ichi']<1.4)
		ind = (t['gchi']<1.5) & (t['ichi']<1.5)
		t = t[ind]
		
	if sharp:
		#ind = (np.abs(t['gsharp'])<0.2) & (np.abs(t['isharp'])<0.2)
		ind = (np.abs(t['gsharp'])<0.3) & (np.abs(t['isharp'])<0.3)
		t = t[ind]
	
	if bright:
		t = astropy.table.vstack((t, t_bright))
	
	t.pprint()
	t.write('../data/ext_catalog.txt', format='ascii.commented_header')

def correct_extinction(gcoeff=-3.793, icoeff=-2.086):
	"""Correct for extinction using ebv from sfd, and sdss transform from martin 2013, eqns 1,3"""
	
	t = Table.read('../data/ext_catalog.txt', format='ascii.commented_header')
	
	t['g'] += gcoeff * t['ebv']
	t['i'] += icoeff * t['ebv']
	
	t.pprint()
	t.write('../data/catalog.txt', format='ascii.commented_header')


def get_bright(coordinates=True, chi=True, sharp=True, flag=False):
	"""Format RA, Dec as floats in the final catalog"""
	
	t = Table.read('../data/extract.txt', format='ascii.commented_header')
	ind = (t['g']<18) | (t['i']<18)
	t = t[ind]
	
	plt.close()
	plt.figure()
	
	plt.subplot(211)
	plt.plot(t['g'], t['gflag'], 'ko')
	plt.xlim(15,20)
	
	plt.subplot(212)
	plt.plot(t['i'], t['iflag'], 'ko')
	plt.xlim(15,20)
	
	plt.tight_layout()

def plot_cmd():
	"""Plot CMD of the pandas catalog, and overplot triangulum stream isochrone"""
	
	# read in the catalog
	t = Table.read('../data/catalog.txt', format='ascii.commented_header')
	
	print(len(t[t['g']<24]))
	
	# read in the isochrone
	dist = 32.
	age = 10.
	feh = -1.25
	g_, i_ = get_isochrone(age, feh, dist)
	
	# select on isochrone
	gil = g_ - i_ - 0.15*(i_/24)**3
	gir = g_ - i_ + 0.15*(i_/24)**3
	
	plt.close()
	plt.figure()
	
	plt.subplot(121, aspect='equal')
	plt.plot(t['g'] - t['i'], t['i'], 'ko', ms=2, alpha=0.1)
	
	plt.xlim(0, 2)
	plt.ylim(24, 18)
	
	plt.xlabel("g - i")
	plt.ylabel("i")
	
	plt.subplot(122, aspect='equal')
	plt.plot(g_ - i_, i_, 'r-', label="[Fe/H]={0:2.1f}\n{1:3.1f} Gyr\n{2:.0f} kpc".format(feh, age, dist))
	plt.plot(gil, i_, 'r:')
	plt.plot(gir, i_, 'r:')

	plt.plot(t['g'] - t['i'], t['i'], 'ko', ms=2, alpha=0.1)
	
	plt.xlim(0, 2)
	plt.ylim(24, 18)
	plt.legend(frameon=False, fontsize=12)
	
	plt.xlabel("g - i")
	plt.ylabel("i")
	
	plt.tight_layout()
	plt.savefig('../plots/triangulum_cmd.png', bbox_inches='tight')

def plot_chisharp(band='g'):
	""""""
	
	t = Table.read('../data/catalog.txt', format='ascii.commented_header')
	
	if band=='g':
		mag = t['g']
		chi = t['gchi']
		sharp = t['gsharp']
	elif band=='i':
		mag = t['i']
		chi = t['ichi']
		sharp = t['isharp']
	
	plt.close()
	plt.figure()
	
	plt.subplot(211)
	plt.plot(mag, chi, 'ko', ms=2, alpha=0.1)
	
	plt.xlabel("mag")
	plt.ylabel("chi")
	
	plt.ylim(0, 2)
	plt.xlim(16, 27)
	
	plt.subplot(212)
	plt.plot(mag, sharp, 'ko', ms=2, alpha=0.1)
	
	plt.ylim(-1,1)
	plt.xlim(16, 27)
	
	plt.xlabel("mag")
	plt.ylabel("sharp")

def plot_map():
	"""Plot map of all pandas sources, and those close to the triangulum isochrone"""
	
	t = Table.read('../data/catalog.txt', format='ascii.commented_header')
	
	# read in the isochrone
	dist = 32
	age = 10.
	feh = -1.25
	g_, i_ = get_isochrone(age, feh, dist)
	
	# select on isochrone
	gil = g_ - i_ - 0.1*(i_/24)**3
	gir = g_ - i_ + 0.1*(i_/24)**3
	iniso = between_lines(t['g'] - t['i'], t['g'], gil, g_, gir, g_)
	bright = (t['i']<23) & (t['i']>21)
	iniso = iniso & bright
	
	# binned map
	dbin = 0.1
	vmax = 20
	bins = [ int((np.max(x) - np.min(x))/dbin) for x in [t['ra'], t['dec']] ]
	hist, xedges, yedges = np.histogram2d(t['ra'][iniso], t['dec'][iniso], bins=bins)
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
	print(np.max(hist))
	
	# map all
	hist_all, xedges, yedges = np.histogram2d(t['ra'][bright], t['dec'][bright], bins=bins)
	
	# map ratio
	hist_all[hist_all==0] = 1.
	hist_ratio = hist / hist_all
	
	# expected stream position
	ra = np.linspace(21,24,100)
	dec = -4.4*ra + 129.
	
	# mask positions
	dx = 16.7*u.arcmin.to(u.deg)
	dy = 5*u.arcmin.to(u.deg)
	
	#mra = [21.577, 22.2436, 22.791]
	#mdec = [34.0743, 31.1892, 28.822]
	
	#mra = [21.577, 22.2436, 22.791, 1.4275*15, 21.7849]
	#mdec = [34.0743, 31.1892, 28.822, 32.9036, 33.231]
	
	mra = [21.7867, 22.1269, 22.9109, 23.0144, 23.2215]
	mdec = [33.3037, 31.6026, 28.7033, 27.9046, 27.3129]
	
	plt.close()
	plt.figure(figsize=(18,8))
	
	plt.subplot(151, aspect='equal')
	plt.plot(t['ra'][bright], t['dec'][bright], 'ko', ms=2, alpha=0.1)
	plt.plot(ra, dec, 'r-', alpha=0.3)
	
	plt.xlim(extent[1], extent[0])
	plt.ylim(extent[2], extent[3])
	
	plt.xlabel("RA")
	plt.ylabel("Dec")
	plt.title("All sources")
	
	plt.subplot(152, aspect='equal')
	plt.imshow(hist_all.T, extent=extent, interpolation='none', origin='lower', cmap='binary', vmax=100)
	plt.plot(ra, dec, 'r-', alpha=0.3)
	
	plt.xlim(extent[1], extent[0])
	plt.ylim(extent[2], extent[3])
	
	plt.xlabel("RA")
	plt.ylabel("Dec")
	plt.title("All sources")
	
	plt.subplot(153, aspect='equal')
	#plt.plot(t['ra'][iniso], t['dec'][iniso], 'ko', ms=2, alpha=0.2)
	plt.imshow(hist.T, extent=extent, interpolation='none', origin='lower', cmap='binary', vmax=vmax)
	plt.plot(ra, dec, 'r-', alpha=0.3)
	
	plt.xlim(extent[1], extent[0])
	plt.ylim(extent[2], extent[3])
	
	plt.xlabel("RA")
	plt.ylabel("Dec")
	plt.title("Close to isochrone")
	
	plt.subplot(154, aspect='equal')
	plt.imshow(hist_ratio.T, extent=extent, interpolation='none', origin='lower', cmap='binary', vmax=0.4)
	plt.plot(ra, dec, 'r-', alpha=0.3)
	
	plt.xlim(extent[1], extent[0])
	plt.ylim(extent[2], extent[3])
	
	plt.subplot(155, aspect='equal')
	plt.imshow(hist_ratio.T, extent=extent, interpolation='none', origin='lower', cmap='binary', vmax=0.4)
	plt.plot(ra, dec, 'r-', alpha=0.3)
	
	# rectangles
	for mx, my in zip(mra, mdec):
		plt.gca().add_patch(Rectangle((mx-0.5*dy, my-0.5*dx), dy, dx, alpha=1, facecolor='none', edgecolor='b'))
	
	plt.xlim(extent[1], extent[0])
	plt.ylim(extent[2], extent[3])
	
	plt.xlabel("RA")
	plt.ylabel("Dec")
	plt.title("Ratio")
	
	plt.tight_layout()
	plt.savefig('../plots/triangulum_map.png', bbox_inches='tight')
	
def mask_cmds(scale=0.5):
	""""""
	# read in the catalog
	t = Table.read('../data/catalog.txt', format='ascii.commented_header')
	
	# read in the isochrone
	dist = 32.
	age = 10.
	feh = -1.25
	g_, i_ = get_isochrone(age, feh, dist)
	
	# mask positions
	dx = 16.7*u.arcmin.to(u.deg)
	dy = 5*u.arcmin.to(u.deg)

	#mra = [21.577, 22.2436, 22.791]
	#mdec = [34.0743, 31.1892, 28.822]
	
	#mra = [21.577, 22.2436, 22.791, 21.4125, 21.7849]
	#mdec = [34.0743, 31.1892, 28.822, 32.9036, 33.231]
	
	mra = [21.7867, 22.1269, 22.9109, 23.0144, 23.2215]
	mdec = [33.3037, 31.6026, 28.7033, 27.9046, 27.3129]
	
	Np = len(mra)
	ms = 3
	
	plt.close()
	fig, axes = plt.subplots(2, Np, figsize=(10,8), sharey='all', sharex='all')
	
	for i in range(Np):
		# stream fields
		#ax = axes[0, i]
		plt.sca(axes[0, i])
		
		ind = (np.abs(t['ra']-mra[i]) < scale*dy) & (np.abs(t['dec']-mdec[i]) < scale*dx)
		t_ = t[ind]
		
		plt.plot(t_['g']-t_['i'], t_['g'], 'ko', ms=ms)
		plt.plot(g_ - i_, g_, 'r-')
		
		if i==int(Np/2):
			plt.title("Stream fields")
		
		if i==0:
			plt.ylabel("g")
		
		# off-stream fields
		plt.sca(axes[1, i])
		
		ind = (np.abs(t['ra']-mra[i]-0.5) < scale*dy) & (np.abs(t['dec']-mdec[i]-0.5) < scale*dx)
		t_ = t[ind]
		
		plt.plot(t_['g']-t_['i'], t_['g'], 'ko', ms=ms)
		plt.plot(g_ - i_, g_, 'r-')
		
		if i==int(Np/2):
			plt.title("Off-stream fields")
		
		plt.xlim(0, 2)
		plt.ylim(24, 18)
		
		if i==0:
			plt.ylabel("g")
		plt.xlabel("g - i")
		
	plt.tight_layout()
	plt.savefig("../plots/mask_cmds.{0:.1f}.png".format(scale), bbox_inches='tight')

def plot_mask():
	"""Plot mask info"""
	
	fname = '../data/TriS-1.out'
	ra_s, dec_s, i, priority, inmask = np.loadtxt(fname, usecols=(1,2,4,6,8), unpack=True, dtype='|S11, |S11, <f4, <f4, <i4', skiprows=4)
	
	N = np.size(i)
	ra = np.zeros(N)*u.deg
	dec = np.zeros(N)*u.deg
	
	for j in range(N):
		ra[j] = Angle(ra_s[j] + ' hours')
		dec[j] = Angle(dec_s[j] + 'degrees')
	
	t = Table.read('../data/catalog.txt', format='ascii.commented_header')
	
	# read in the isochrone
	dist = 32
	age = 10.
	feh = -1.25
	g_, i_ = get_isochrone(age, feh, dist)
	
	# select on isochrone
	gil = g_ - i_ - 0.1*(i_/24)**3
	gir = g_ - i_ + 0.1*(i_/24)**3
	iniso = between_lines(t['g'] - t['i'], t['g'], gil, g_, gir, g_)
	iniso = iniso & (t['g']<24) & (t['g']>21)
	
	# binned map
	dbin = 0.2
	vmax = 20
	bins = [ int((np.max(x) - np.min(x))/dbin) for x in [t['ra'], t['dec']] ]
	hist, xedges, yedges = np.histogram2d(t['ra'][iniso], t['dec'][iniso], bins=bins)
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
	
	plt.close()
	plt.figure()
	
	# map
	plt.subplot(121)
	plt.imshow(hist.T, extent=extent, interpolation='gaussian', origin='lower', cmap='binary', vmax=vmax)
	
	ind = inmask==1
	ind2 = ind & (priority>500)
	plt.plot(ra[ind], dec[ind], 'ro', mec='none', alpha=0.1)
	plt.plot(ra[ind2], dec[ind2], 'bo', mec='none', alpha=0.5)
	
	plt.xlim(extent[1], extent[0])
	plt.ylim(extent[2], extent[3])

def get_isochrone(age, feh, dist):
	"""Get dartmouth isochrone of given age, feh, shift to given distance"""
	
	g_, i_ = np.loadtxt("../data/dart_{0:.1f}_{1:.2f}.txt".format(age, -feh), usecols=(6,8), unpack=True)
	g_ += 5*np.log10(dist*100)
	i_ += 5*np.log10(dist*100)
	
	return (g_, i_)

def add_priority(verbose=False):
	""""""
	t = Table.read('../data/catalog.txt', format='ascii.commented_header', include_names=('ra', 'dec', 'g', 'i'))
	t = t[t['i']<23]
	priority = np.ones(len(t))
	
	# bright targets
	# guide stars -1, alignment -2 (toggle in dsim)
	guide = t['i']<18.5
	priority[guide] = -2
	
	# science targets
	# read in the isochrone
	dist = 32
	age = 10.
	feh = -1.25
	g_, i_ = get_isochrone(age, feh, dist)
	
	# select on isochrone
	gil = g_ - i_ - 0.15*(i_/24)**3
	gir = g_ - i_ + 0.15*(i_/24)**3
	iniso = between_lines(t['g'] - t['i'], t['g'], gil, g_, gir, g_)
	bright = (t['i']<22.5) & (t['i']>21)
	
	# assign
	priority[iniso] = 10
	priority[iniso & bright] = 500
	
	add_npcolumn(t, name='priority', vec=priority)
	
	# star id
	add_npcolumn(t, name='sid', vec=np.arange(len(t), dtype=np.int64), index=0, dtype=np.int64)
	
	if verbose: t.pprint()
	
	t.write("../data/catalog_priorities.txt", format='ascii.commented_header')

#import astropy.coordinates as ac

def create_infile(graph=False):
	"""Create a dsim infile"""
	
	t = Table.read("../data/catalog_priorities.txt", format='ascii.commented_header')
	
	high_label = ""
	high = True
	if high:
		ind = (t['priority']==500) | (t['priority']<0)
		t = t[ind]
		high_label = "_high"
	
	
	# mask setup
	field = 0
	mra = [21.7584, 21.7867, 22.1269, 22.9109, 23.0144, 23.2215]
	mdec = [33.2741, 33.3037, 31.6026, 28.7033, 27.9046, 27.3129]
	print(mra[field]/15, mdec[field])
	
	dx = 16.7*u.arcmin.to(u.deg)
	dy = 5*u.arcmin.to(u.deg)
	
	inmask = (np.abs(t['ra'] - mra[field])<1.5*dx) & (np.abs(t['dec'] - mdec[field])<1.5*dx)
	t = t[inmask]

	guide = t['priority']<0

	
	# infile fields
	nstar = len(t)
	band = 'i'
	
	sid = t['sid']
	ra = Angle(t['ra']*u.deg).to_string(unit=u.hour, sep=':')
	dec = Angle(t['dec']*u.deg).to_string(unit=u.deg, sep=':',  alwayssign=True)
	epoch = np.ones(nstar) * 2000.0
	mag = t[band]
	passband = np.repeat([band.upper(),], nstar)
	priority = t['priority']
	sample = np.ones(nstar, dtype=np.int64)
	preselect = np.zeros(nstar, dtype=np.int64)
	
	t_infile = Table([sid, ra, dec, epoch, mag, passband, priority, sample, preselect], names=('sid', 'ra', 'dec', 'epoch', 'mag', 'band', 'priority', 'sample', 'preselect'))
	t_infile.pprint()
	
	t_infile.write("../data/infiles/TriS_{0:d}_infile{1:s}.dat".format(field+2, high_label), format='ascii.no_header')
	
	if graph:
		plt.close()
		plt.figure()
		
		plt.subplot(221)
		plt.plot(t['g']-t['i'], t['i'], 'ko', ms=2)
		
		plt.gca().invert_yaxis()
		
		plt.subplot(222)
		plt.plot(t['ra'][guide], t['dec'][guide], 'ro')
		plt.plot(t['ra'][~guide], t['dec'][~guide], 'ko', ms=4)
		
		# expected stream position
		ras = np.linspace(21,24,100)
		decs = -4.4*ras + 129.
		plt.plot(ras, decs, 'r-')
		
		mx = mra[field]
		my = mdec[field]
		plt.gca().add_patch(Rectangle((mx-0.5*dy, my-0.5*dx), dy, dx, angle=0, alpha=1, facecolor='none', edgecolor='b'))
		
		plt.xlim(np.max(t['ra']), np.min(t['ra']))
		plt.ylim(np.min(t['dec']), np.max(t['dec']))

# utilities

def add_npcolumn(t, name="", vec=None, dtype='float', index=None):
	"""Add numpy array as a table column"""
	
	if index==None:
		index = len(t.columns)
	
	if vec==None:
		vec = np.array(np.size(t))
		
	tvec = astropy.table.Column(vec, name=name, dtype=dtype)
	t.add_column(tvec, index=index)
	
	return vec

def between_lines(x, y, x1, y1, x2, y2):
	"""check if points x,y are between lines defined with x1,y1 and x2,y2"""
	
	if y1[0]>y1[-1]:
		y1 = y1[::-1]
		x1 = x1[::-1]
		
	if y2[0]>y2[-1]:
		y2 = y2[::-1]
		x2 = x2[::-1]
	
	xin1 = np.interp(y,y1,x1)
	xin2 = np.interp(y,y2,x2)
	
	indin = (x>=xin1) & (x<=xin2)
	
	return indin


##### Data #####

def v_hist():
	""""""
	t = Table.read("../data/deimos.fits")
	print(t.keys())
	
	N = len(t)
	ra = np.zeros(N)*u.deg
	dec = np.zeros(N)*u.deg
	
	for j in range(N):
		ra[j] = Angle(t['RA'][j] + ' hours')
		dec[j] = Angle(t['DEC'][j] + 'degrees')
	
	# 
	fname = '../data/TriS-1.out'
	ra_s, dec_s, i, priority, inmask = np.loadtxt(fname, usecols=(1,2,4,6,8), unpack=True, dtype='|S11, |S11, <f4, <f4, <i4', skiprows=4)
	
	Nm = np.size(ra_s)
	ra_m = np.zeros(Nm)*u.deg
	dec_m = np.zeros(Nm)*u.deg
	
	for j in range(Nm):
		ra_m[j] = Angle(ra_s[j] + ' hours')
		dec_m[j] = Angle(dec_s[j] + 'degrees')
	
	# match catalogs
	ref = ICRS(ra_m, dec_m, unit=(u.degree, u.degree))
	cat = ICRS(ra, dec, unit=(u.degree, u.degree))
	id1, d21, d31 = cat.match_to_catalog_sky(ref)
	
	matches = d21<0.5*u.arcsec
	
	t_priority = priority[id1]
	i_m = i[id1]
	indices = (t['ZQUALITY']>-1)
	
	t['Z'] *= 300000
	
	add_npcolumn(t, name="priority", vec=t_priority)
	#print(t_priority, np.size(t_priority))
	
	plt.close()
	plt.figure()
	
	plt.subplot(221)
	plt.hist(t['Z'], bins=np.arange(-400,200,10), histtype='step', color='k')
	plt.hist(t['Z'][indices], bins=np.arange(-400,200,10), histtype='step', color='b')
	plt.hist(t['Z'][t_priority>=500], bins=np.arange(-400,200,10), histtype='step', color='r')
	
	plt.minorticks_on()
	
	plt.subplot(222)
	plt.plot(i_m[indices], t['Z'][indices], 'ko')
	print(t['Z'][t_priority>=500], i_m[t_priority>=500])
	print(np.max(i_m[indices]))

def v_map():
	""""""
	
	t = Table.read("../data/deimos.fits")
	t['Z'] *= 300000
	
	N = len(t)
	ra = np.zeros(N)*u.deg
	dec = np.zeros(N)*u.deg
	
	for j in range(N):
		ra[j] = Angle(t['RA'][j] + ' hours')
		dec[j] = Angle(t['DEC'][j] + 'degrees')
		
	
	plt.close()
	plt.figure()
	
	plt.subplot(121)
	plt.plot(ra, t['Z'], 'ko')
	
	plt.xlim(21.8, 21.4)
	plt.ylim(-400, 200)
	
	plt.subplot(122)
	plt.plot(dec, t['Z'], 'ko')
	
	plt.xlim(34., 34.2)
	plt.ylim(-400, 200)
	
	plt.tight_layout()

# rescale exposure times based on the preliminary data
def scale_exptime(mlim, t, mtarget):
	"""Calculate exposure time necessary to get down to mtarget, given mlim at time t"""
	
	return t * 10**(-0.8*(mlim - mtarget))

def scale_mlim(mlim, t, ttarget):
	"""Calculate limiting magnitude at exposure time ttarget, given mlim at time t"""
	
	return mlim + 1.25*np.log10(ttarget/t)

def get_times():
	""""""
	
	mlim = 21.4
	t = 1.
	mtarget = np.arange(21.5, 23.1, 0.5)
	
	print(scale_exptime(mlim, t, mtarget), mtarget)

def get_mlims():
	""""""
	
	mlim = 21.4
	t = 1.
	ttarget = np.arange(1.5, 3.1, 0.5)
	print(scale_mlim(mlim, t, ttarget), ttarget)
	




