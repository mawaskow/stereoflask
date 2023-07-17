from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os.path
import dateutil.parser
from tqdm import tqdm
import cv2 as cv
import glob
from datetime import datetime


class NetCDFAnalysis():
    def __init__(self, ncfile):
        self.NCFILE = ncfile
        _, self.ncfilef = os.path.split(self.NCFILE)

        self.root = Dataset( self.NCFILE, "r", format="NETCDF4" )

        self.scriptpath, _ = os.path.split( __file__ )

        self.timestring = "unknown"
        self.datadatetime = None
        
        self.uploadpath = os.path.join(self.scriptpath, "uploads\\analysis\\")

        try:
            timestring = self.root["/meta"].getncattr("timestring")
            self.timestring = int(timestring[1])
            dt_object = datetime.fromtimestamp(self.timestring)
            self.datadatetime = dt_object
        except Exception as error: # AttributeError, IndexError, ParserError
            print("Unable to read /meta/timestring from nc file.")
            print(repr(error))

    def get_ncfile_info( self ):
        S = ""
        S += "Dataset: %s\n"%self.ncfilef
        S += "------------------------------------------------------\n"
        S += "/ Variables:\n"
        for v in self.root.variables:
            S += " - %s\n"%v
        S += "Attributes:\n"
        for a in self.root.ncattrs():
            S += " - %s = %s\n"%(a,self.root.getncattr(a))
        S += "Groups: \n"
        for g in self.root.groups:
            S += " %s\n"%self.root[g].path
            S += "  Variables:\n"
            for v in self.root[g].variables:
                S += "    - %s\n"%v
            S += "  ncattrs:\n"
            for a in self.root[g].ncattrs():
                S += " - %s = %s\n"%(a,self.root[g].getncattr(a))
        return S

    def filetobase64( filename ):
        import base64
        with open(filename,'rb') as f:
            fb64 = base64.b64encode( f.read() )
            return fb64

    def analyse( self ):
        plt.rcParams.update({'font.size': 18})

        nsamples = self.root["/Z"].shape[0]
        gridsize = self.root["/Z"].shape[1:3]
        halfgridsize_i = int( gridsize[0]/2)
        halfgridsize_j = int( gridsize[1]/2)
        valid_samples_i = range( halfgridsize_i-5, halfgridsize_i+6 )
        valid_samples_j = range( halfgridsize_j-5, halfgridsize_j+6 )
        try:
            I0 = cv.imdecode( self.root["/cam0images"][1], cv.IMREAD_GRAYSCALE )
            I0 = cv.pyrDown(I0)
            cv.imwrite(os.path.join(self.uploadpath+"frame.png"),I0)
            I0 = None
        except:
            print("   NetCDF file contains no frame data")
            # We write a null image instead
            cv.imwrite(os.path.join(self.uploadpath+"frame.png"), np.ones( (2,3), dtype=np.uint8 )*255)

        # -------------------------------------------------------------------------- TIMESERIE

        #Computing statistics on the grid center timeseries
        timeserie = self.root["/Z"][:, halfgridsize_i, halfgridsize_j] * 1E-3
        timeserie = timeserie - np.nanmean(timeserie)

        t = self.root["/time"]
        dt = t[2] - t[1]
        
        def crossings_nonzero_pos2neg(data):
            pos = data > 0
            return (pos[:-1] & ~pos[1:]).nonzero()[0]

        def crossings_nonzero_all(data):
            pos = data > 0
            npos = ~pos
            return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]

        crossings = crossings_nonzero_pos2neg(timeserie)

        dmins = []
        dmaxs = []
        for ii in range( np.size(crossings)-1 ):
            datarange = np.arange(crossings[ii], crossings[ii+1])
            data = timeserie[ datarange ]
            dmax = np.argmax(data)
            dmin = np.argmin(data)
            dmins.append( datarange[dmin] )
            dmaxs.append( datarange[dmax] )
            
        waveheights = np.array(timeserie[dmaxs]) - np.array(timeserie[dmins])

        waveheights = waveheights[np.logical_not(np.isnan(waveheights))]
        q = np.quantile( waveheights, 2.0/3.0)
        #timeserie quantile: q

        highestthirdwaves = waveheights[ waveheights>q ]
        H13 = np.mean(highestthirdwaves)
        #H1/3

        plt.figure( figsize=(20,10))
        plt.plot(t, timeserie )
        plt.scatter( t[crossings], np.zeros_like(crossings), c="r")
        plt.scatter( t[dmins], timeserie[dmins], c="b")
        plt.scatter( t[dmaxs], timeserie[dmaxs], c="g")
        plt.grid()
        plt.title("Timeserie at grid center. %d waves"%np.size(waveheights))
        plt.xlabel("Time (secs.)")
        plt.ylabel("Height (m)")
        plt.savefig(self.uploadpath+"timeserie.png")
        plt.close()

        Zcube = np.array( self.root["/Z"][:,valid_samples_i,valid_samples_j] * 1E-3 )
        Hs = 4.0*np.nanstd( Zcube-np.nanmean(Zcube) )
        Zcube = None

        # -------------------------------------------------------------------------- 1D SPECTRUM
        #Analyzing 1D spectrum

        import scipy.signal

        f, S = scipy.signal.csd(timeserie, timeserie, 1.0/dt, nperseg=512 )

        for ii in tqdm(valid_samples_i):
            for jj in valid_samples_j:
                timeserie_neigh = self.root["/Z"][:,ii,jj] * 1E-3
                timeserie_neigh = timeserie_neigh - np.mean(timeserie_neigh)
                _, S_neig = scipy.signal.csd(timeserie_neigh, timeserie_neigh, 1.0/dt, nperseg=512 )
                S = np.nan_to_num(S, copy=False, nan=0)
                S_neig = np.nan_to_num(S_neig, copy=False, nan=0)
                S += S_neig

        S = S / float( np.size(valid_samples_i)*np.size(valid_samples_j) + 1)

        plt.figure( figsize=(10,10) )
        plt.loglog( f, S)
        plt.xticks([1E-2,1E-1,1E0,1E1])
        plt.grid(which='minor')
        plt.ylabel("S (m^2s)")
        plt.xlabel("f_a (1/s)")
        plt.title("Spectrum (Welch method) averaged in central grid region")
        plt.savefig(self.uploadpath+"spectrum.png")
        plt.close()

        # Compute Hs
        dFreq = np.gradient( f )
        m0 = np.sum( S*dFreq )
        m1 = np.sum( f*S*dFreq )
        Hm0 = 4.0 * np.sqrt( m0 )

        # Peak frequency
        pp = f[np.argmax( S )]

        # Average Period Tm01
        Tm01 = m0/m1

        # -------------------------------------------------------------------------- 3D SPECTRUM
        #Analyzing space-time 3D spectrum
        Z = self.root["/Z"][3,:,:]
        N = Z.shape[0]
        Nm = int( N/2 )
        dy = (self.root["/Y_grid"][2,0] - self.root["/Y_grid"][1,0])/1000.0
        dx = (self.root["/X_grid"][0,2] - self.root["/X_grid"][0,1])/1000.0
        #grid dx,dy

        if np.abs( dx-dy ) < 1E-3:
            dy = dx  # force dx = dy if very close to avoid numerical errors

        # Extract a central part of the Zcube
        N = 140
        min_freq = 0.25
        max_freq = 0.7
        num_plots = 10
        segments = 5 

        sequence_length = np.size(timeserie)
        Nt = int(sequence_length / segments)
        if Nt%2 > 0:
            Nt+=1
        seg_shift = int(Nt/2)

        Zcube_mr = int( self.root["/Z"].shape[1] / 2 )
        Zcube_mc = int( self.root["/Z"].shape[2] / 2 )
        r_start, r_end = Zcube_mr-int(N/2)-20, Zcube_mr+int(N/2)-20+1
        c_start, c_end = Zcube_mc-int(N/2), Zcube_mc+int(N/2)+1 

        Nx = r_end - r_start
        Ny = c_end - c_start

        kx_max=(2.0*np.pi/dx)/2.0
        ky_max=(2.0*np.pi/dy)/2.0
        f_max= (1.0/dt)/2.0
        dkx=2.0*np.pi/(dx*np.floor(Nx/2.0)*2.0)
        dky=2.0*np.pi/(dy*np.floor(Ny/2.0)*2.0)
        df =1.0/(dt*np.floor(Nt/2.0)*2.0)

        assert( Nx%2 != 0)
        assert( Ny%2 != 0)
        assert( Nt%2 == 0)

        kx=np.arange(-kx_max,kx_max+dkx,dkx)
        ky=np.arange(-ky_max,ky_max+dky,dky)

        if Nt%2==0:
            f=np.arange(-f_max, f_max, df)
        else:
            f=np.arange(-f_max, f_max+df, df)

        KX, KY = np.meshgrid( kx, ky )
        dkx=kx[3]-kx[2]
        dky=ky[3]-ky[2]
        KXY=np.sqrt(KX**2+KY**2)

        hanningx = scipy.signal.windows.hann(KX.shape[0])
        hanningy = scipy.signal.windows.hann(KX.shape[1])
        hanningt = scipy.signal.windows.hann(Nt)

        Win3Dhann = np.tile( np.expand_dims( hanningx, axis=-1) * hanningy, (Nt,1,1) ) *  np.tile( np.expand_dims( np.expand_dims( hanningt, axis=-1 ), axis=-1 ), (1, KX.shape[0], KX.shape[1]) )
        assert( KX.shape == Win3Dhann.shape[1:] )

        #  window correction factors
        wc2x = 1.0/np.mean(hanningx**2)
        wc2y = 1.0/np.mean(hanningy**2)
        wc2t = 1.0/np.mean(hanningt**2)
        wc2xy  = wc2x *wc2y
        wc2xyt = wc2xy*wc2t

        # Fix for rounding errors
        r_end = r_start + Win3Dhann.shape[1]
        c_end = c_start + Win3Dhann.shape[2]

        S_welch = np.zeros_like( Win3Dhann )
        n_samples = 0

        #Computing 3D FFT via Welch's method
        for ii in tqdm(range(segments*2)):
            #print("Welch sample %d/%d"%(ii+1,segments*2))
            Zcube_small = np.array( self.root["/Z"][(ii*seg_shift):(ii*seg_shift+Nt), r_start:r_end, c_start:c_end ] )
            Zcube_small = np.nan_to_num(Zcube_small, copy=False, nan=0)
            if Zcube_small.shape[0] != Nt:
                break
            
            Zcube_w = (Zcube_small - np.mean(Zcube_small) ) * Win3Dhann
            
            S = np.fft.fftshift( np.fft.fftn( Zcube_w, norm="ortho" ) )
            S /= (S.shape[0]*S.shape[1]*S.shape[2])
            S = np.abs(S)**2 / (dkx*dky*df)
            #-----------------------------
            #%%%%% corrects for window
            #----------------------------
            #%% FABIEN
            S *= wc2xyt
            
            # Store
            S_welch += S    
            n_samples += 1
            
        S_welch /= n_samples    

        start_freq_ii = np.argmin( np.abs(f-min_freq) )
        end_freq_ii = np.argmin( np.abs(f-max_freq) )
        indices = np.round( np.linspace(start_freq_ii, end_freq_ii, num_plots ) ).astype(np.uint32)

        #Generating plots

        kk=0
        for ii in tqdm(indices):

            plt.figure( figsize=(11,10))    

            #dummy = np.flipud( 2* np.mean(S_welch[ mdt+ii-1:mdt+ii+2,:,:], axis=0) )    
            dummy = 2* np.mean(S_welch[ ii-1:ii+2,:,:], axis=0) 
            
            dummy_cen = np.copy(dummy)
            dummy_cen[ int(dummy_cen.shape[0]/2)-1:int(dummy_cen.shape[0]/2)+1, int(dummy_cen.shape[1]/2)-1:int(dummy_cen.shape[1]/2)+1 ] = 0
            maxidx = np.unravel_index( np.argmax(dummy_cen), dummy_cen.shape )
            
            qp=( np.arctan2( KY[ maxidx[0],maxidx[1] ], KX[ maxidx[0],maxidx[1] ]) )/np.pi*180.0
            if qp<0:
                qp=qp+360

            kp=np.sqrt( KX[ maxidx[0],maxidx[1] ]**2 + KY[ maxidx[0],maxidx[1] ]**2 )

            plt.pcolor(KX,KY, 10*np.log10(dummy), shading = "auto")
            plt.clim( 10*np.array([-4.0 + np.amax(np.log10(dummy)), -0+np.amax(np.log10(dummy))]) )
            plt.colorbar()

            plt.scatter( [KX[ maxidx[0],maxidx[1] ]], [KY[ maxidx[0],maxidx[1] ]], marker="x", s=100, c="k")

            plt.ylim([-3.0,3.0])
            plt.xlim([-3.0,3.0])

            plt.xlabel("Kx (rad/m)")
            plt.ylabel("Ky (rad/m)")
            plt.title("S_kx_ky, fa=%3.2f (Hz).\n Peak angle: %3.0f°, mag: %2.3f (rad/m)\n"%( f[ii],qp,kp ) )
            plt.savefig(os.path.join(self.uploadpath,"spectrum_dir_%03d.png"%kk))
            plt.close()
            kk+=1
        # -------------------------------------------------------------------------- REPORT 

        #Generating report
        datafile = ""
        try:
            datafile = self.root["/meta"].getncattr("datafile") 
        except:
            pass

        location = "unknown"
        try:
            location = self.root["/meta"].getncattr("location") 
        except:
            pass
        
        title="%s wave analysis"%self.ncfilef
        framedata=NetCDFAnalysis.filetobase64(os.path.join(self.uploadpath,"frame.png")).decode("ascii")
        timeseriedata=NetCDFAnalysis.filetobase64(os.path.join(self.uploadpath,"timeserie.png")).decode("ascii")
        spectrumdata=NetCDFAnalysis.filetobase64(os.path.join(self.uploadpath,"spectrum.png")).decode("ascii")
        dirspectrumdata1=NetCDFAnalysis.filetobase64(os.path.join(self.uploadpath,"spectrum_dir_000.png")).decode("ascii")
        dirspectrumdata2=NetCDFAnalysis.filetobase64(os.path.join(self.uploadpath,"spectrum_dir_001.png")).decode("ascii")
        dirspectrumdata3=NetCDFAnalysis.filetobase64(os.path.join(self.uploadpath,"spectrum_dir_002.png")).decode("ascii")
        dirspectrumdata4=NetCDFAnalysis.filetobase64(os.path.join(self.uploadpath,"spectrum_dir_003.png")).decode("ascii")
        dirspectrumdata5=NetCDFAnalysis.filetobase64(os.path.join(self.uploadpath,"spectrum_dir_004.png")).decode("ascii")
        dirspectrumdata6=NetCDFAnalysis.filetobase64(os.path.join(self.uploadpath,"spectrum_dir_005.png")).decode("ascii")
        dirspectrumdata7=NetCDFAnalysis.filetobase64(os.path.join(self.uploadpath,"spectrum_dir_006.png")).decode("ascii")
        dirspectrumdata8=NetCDFAnalysis.filetobase64(os.path.join(self.uploadpath,"spectrum_dir_007.png")).decode("ascii")
        dirspectrumdata9=NetCDFAnalysis.filetobase64(os.path.join(self.uploadpath,"spectrum_dir_008.png")).decode("ascii")
        dirspectrumdata10=NetCDFAnalysis.filetobase64(os.path.join(self.uploadpath,"spectrum_dir_009.png")).decode("ascii")
        hs="%2.3f"%Hs
        hm0="%2.3f"%Hm0
        pp="%2.3f"%pp
        Tm01="%3.3f"%Tm01
        location=location
        date=self.datadatetime.date() if self.datadatetime else "unknown"
        time=self.datadatetime.time() if self.datadatetime else "unknown"
        ncfile="%s (%s)"%(self.ncfilef,datafile )
        duration="%d secs."%(t[-1]-t[0])
        fps="%3.1f"%(1.0/dt)
        meta=self.get_ncfile_info()
        outfile =  self.NCFILE.replace(".nc",".html")
        dirspectrumdata = [dirspectrumdata1, dirspectrumdata2, dirspectrumdata3, dirspectrumdata4, dirspectrumdata5, dirspectrumdata6, dirspectrumdata7, dirspectrumdata8, dirspectrumdata9, dirspectrumdata10]

        return title, framedata, timeseriedata, spectrumdata, dirspectrumdata, hs, hm0, pp, Tm01, location, date, time, ncfile, duration, fps, meta, outfile