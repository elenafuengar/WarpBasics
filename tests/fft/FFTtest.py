# test for CST WP to Impedance FFT
import sys
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt
import wakis

wk = wakis.Wakis()

#bunch parameters
q = 1e-9
sigmaz = 18.737*1e-3

# read charge dist in distance lambda(z)
d1 = wk.read_cst_1d('lambda.txt')
chargedist = d1['Y'] 
z = d1['X']*1e-3

# read wake potential WPz(s)
d2 = wk.read_cst_1d('WP.txt')
WP = d2['Y']
s = d2['X']*1e-3

# interpolate charge dist to s
wk.calc_lambdas(z=z, s=s, chargedist=chargedist, q=q, sigmaz=sigmaz)
wk.calc_long_Z(WP=WP)

# Plot impedance
Z = wk.Z
f = wk.f
fig = plt.figure(2, figsize=(8,5), dpi=150, tight_layout=True)
ax=fig.gca()
ax.plot(f*1e-9, abs(Z), lw=1.2, c='b', label = 'abs Z(w)')
ax.plot(f*1e-9, np.real(Z), lw=1.2, c='r', ls='--', label = 'Re Z(w)')
ax.plot(f*1e-9, np.imag(Z), lw=1.2, c='g', ls='--', label = 'Im Z(w)')

# Compare with CST
d = wk.read_cst_1d('Z.txt')
ax.plot(d['X'], d['Y'], lw=1, c='k', ls='--', label='abs Z(w) CST')

#add Z computed with analytical lambda
wk.calc_lambdas_analytic(s=s, sigmaz=sigmaz)
wk.calc_long_Z(WP=WP)
Z_gauss = wk.Z
f_gauss = wk.f
ax.plot(f_gauss*1e-9, abs(Z_gauss), lw=1.2, c='c', label = 'abs Z(w) - analytical')


ax.set( title='Impedance Z(w)',
        xlabel='f [GHz]',
        ylabel='Z(w) [$\Omega$]',   
        xlim=(0.,np.max(f)*1e-9)    )
ax.legend(loc='upper left')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#fig.savefig(path+'Zz.png', bbox_inches='tight')