from __future__ import division
import sys
sys.path.append('/home/sciortino/ML')
import profile_unc_estimation
import numpy as np
import collections
import profiletools
import MDSplus
import gptools
import os
import scipy
import Tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.gridspec as mplgs
import matplotlib.widgets as mplw
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import profiletools
import profiletools.gui
import re

# Regex used to split lists up. This will let the list be delimted by any
# non-numeric characters, where the decimal point and minus sign are NOT
# considered numeric.
LIST_REGEX = r'([0-9]+)[^0-9]*'

# Region of interest for Ca lines on XEUS (in nanometers):
XEUS_ROI = (1.8, 2.2)

# Locations of Ca 17+ lines (in nanometers, taken from pue#ca17.dat):
CA_17_LINES = (
    1.8683, 30.2400, 1.9775, 1.8727, 34.4828, 1.9632, 2.0289, 1.4080, 2.0122,
    1.4091, 1.4736, 1.2642, 1.9790, 1.4652, 1.4852, 1.2647, 1.3181, 1.4763,
    5.4271, 1.3112, 1.3228, 5.7753, 1.4739, 5.4428, 3.7731, 5.7653, 5.8442,
    5.6682, 1.3157, 3.9510, 5.6332, 3.7770, 5.8015, 3.9457, 3.8990, 1.3182,
    3.9404, 5.8320, 3.8812, 3.9210, 5.8370, 11.8177, 12.4743, 5.6734, 3.9644,
    12.6866, 12.4557, 11.8555, 5.7780, 12.2669, 3.9627, 3.9002, 12.6020, 12.6091,
    3.9517, 12.2001, 5.8190, 12.6265, 12.4970, 12.4883, 3.9585, 12.2793, 12.4807,
    12.5836, 12.5252, 12.5256, 12.5007, 107.5003, 12.5127, 124.3039, 260.0374,
    301.5136, 229.1056, 512.7942, 286.7219, 595.2381, 321.8228, 545.2265,
    682.1748, 1070.0909, 1338.6881, 766.2248, 1505.1174, 3374.9577, 4644.6817,
    6583.2783, 9090.9089, 7380.0736, 14430.0141
)

# Locations of Ca 16+ lines (in nanometers):
CA_16_LINES = (19.2858,)

# Combined Ca 16+, 17+ lines (in nanometers):
CA_LINES = scipy.asarray(CA_17_LINES + CA_16_LINES)

# POS vector for XEUS:
XEUS_POS = [2.561, 0.2158, 0.196, 0.1136]

class VUVData(object):
    """Helper object to load and process the VUV data.
    
    Execution proceeds as follows:
    
    * Loads the XEUS data.
    * Allows user to select lines, background subtraction intervals.
    * Loads the LoWEUS data.
    * Allows the user to select lines, background subtraction intervals.
    * Computes the normalization factors.
    * Loads the data into a :py:class:`Signal` instance. This is stored in the
      attribute :py:attr:`signal` for later use.
    """
    def __init__(self, shot, injections, debug_plots=False):
        self.shot = shot
        
        self.vuv_lines = collections.OrderedDict()
        self.vuv_signal = {}
        self.vuv_time = {}
        self.vuv_lam = {}
        self.vuv_uncertainty = {}
        
        self.load_vuv('XEUS')
        
        t = []
        y = []
        std_y = []
        y_norm = []
        std_y_norm = []
        
        for k, i in enumerate(injections):
            vuv_signals = []
            vuv_uncertainties = []
            vuv_times = []
            
            for s in self.vuv_lines.keys():
                i_start, i_stop = profiletools.get_nearest_idx(
                    [i.t_start, i.t_stop],
                    self.vuv_time[s]
                )
                for l in self.vuv_lines[s]:
                    if l.diagnostic_lines is not None:
                        vuv_signals.append(
                            l.signal[i_start:i_stop + 1]
                        )
                        vuv_uncertainties.append(
                            l.uncertainty[i_start:i_stop + 1]
                        )
                        vuv_times.append(
                            self.vuv_time[s][i_start:i_stop + 1] - i.t_inj
                        )
            vuv_signals = scipy.asarray(vuv_signals)
            vuv_uncertainties = scipy.asarray(vuv_uncertainties)
            vuv_times = scipy.asarray(vuv_times)
            
            # We don't have a brightness cal for XEUS or LoWEUS, so normalize to
            # the peak:
            vuv_signals_norm = scipy.nan * scipy.zeros_like(vuv_signals)
            vuv_uncertainties_norm = scipy.nan * scipy.zeros_like(vuv_uncertainties)
            
            # import pdb
            # pdb.set_trace()
            for j in xrange(0, vuv_signals.shape[0]):
                m, s = interp_max(
                    vuv_times[j, :],
                    vuv_signals[j, :],
                    err_y=vuv_uncertainties[j, :],
                    debug_plots=debug_plots,
                    s_max=100.0,
                    method='GP' #added by FS
                )
                vuv_signals_norm[j, :] = vuv_signals[j, :] / m
                vuv_uncertainties_norm[j, :] = (
                    scipy.sqrt(
                        (vuv_uncertainties[j, :] / m)**2.0 + ((vuv_signals[j, :] / m)*(s / m))**2.0
                    )
                )
            
            # Assume all are on the same timebase:
            t.append(vuv_times[0])
            y.append(vuv_signals)
            std_y.append(vuv_uncertainties)
            y_norm.append(vuv_signals_norm)
            std_y_norm.append(vuv_uncertainties_norm)
        
        blocks = []
        names = []
        pos = []
        i = 0
        for s in self.vuv_lines.keys():
            for l in self.vuv_lines[s]:
                if l.diagnostic_lines is not None:
                    blocks.append(i)
                    i += 1
                    names.append(s)
                    pos.append(XEUS_POS if s == 'XEUS' else LOWEUS_POS)
        
        self.signal = Signal(
            scipy.hstack(y).T,
            scipy.hstack(std_y).T,
            scipy.hstack(y_norm).T,
            scipy.hstack(std_y_norm).T,
            scipy.hstack(t),
            names,
            scipy.asarray(blocks, dtype=int) + 1,
            pos=pos,
            blocks=blocks
        )
    
    def load_vuv(self, system):
        """Load the data from a VUV instrument.
        
        Parameters
        ----------
        system : {'XEUS', 'LoWEUS'}
            The VUV instrument to load the data from.
        """
        print("Loading {system} data...".format(system=system))
        t = MDSplus.Tree('spectroscopy', self.shot)
        N = t.getNode(system + '.spec')
        self.vuv_lines[system] = []
        self.vuv_signal[system] = scipy.asarray(N.data(), dtype=float)
        self.vuv_time[system] = scipy.asarray(N.dim_of(idx=1).data(), dtype=float)
        self.vuv_lam[system] = scipy.asarray(N.dim_of(idx=0).data(), dtype=float) / 10.0
        
        # Get the raw count data to compute the uncertainty:
        self.vuv_uncertainty[system] = (
            self.vuv_signal[system] /
            scipy.sqrt(t.getNode(system + '.raw:data').data())
        )
        
        print("Processing {system} data...".format(system=system))
        self.select_vuv(system)
        
    def select_vuv(self, system):
        """Select the lines to use from the given VUV spectrometer.
        """
        root = VuvWindow(self, system)
        # print("Type 'continue' after closing window.")
        # import pdb
        # pdb.set_trace()
        root.mainloop()

class VuvWindow(tk.Tk):
    def __init__(self, data, system):
        tk.Tk.__init__(self)
        self.protocol("WM_DELETE_WINDOW", self.exit)
        
        self.data = data
        self.system = system
        
        self.wm_title(system + " inspector")
        
        self.plot_frame = VuvPlotFrame(self)
        self.plot_frame.grid(row=0, column=0, sticky='NESW')
        
        self.slider_frame = VuvSliderFrame(self)
        self.slider_frame.grid(row=1, column=0, sticky='NESW')
        
        self.line_frame = VuvLineFrame(self)
        self.line_frame.grid(row=0, column=1, rowspan=2, sticky='NESW')
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.bind("<Left>", self.on_arrow)
        self.bind("<Right>", self.on_arrow)
        self.bind("<Up>", self.on_arrow)
        self.bind("<Down>", self.on_arrow)
    
    def exit(self):
        self.destroy()
        self.quit()

    def on_arrow(self, evt):
        """Handle arrow keys to move slider.
        """
        if evt.keysym == 'Right':
            self.slider_frame.lam_slider.set(
                min(
                    self.slider_frame.lam_slider.get() + 1,
                    len(self.data.vuv_lam[self.system]) - 1
                )
            )
        elif evt.keysym == 'Left':
            self.slider_frame.lam_slider.set(
                max(self.slider_frame.lam_slider.get() - 1, 0)
            )
        elif evt.keysym == 'Up':
            self.slider_frame.t_slider.set(
                min(
                    self.slider_frame.t_slider.get() + 1,
                    len(self.data.vuv_time[self.system]) - 1
                )
            )
        elif evt.keysym == 'Down':
            self.slider_frame.t_slider.set(
                max(self.slider_frame.t_slider.get() - 1, 0)
            )
    
    def update_t(self, t_idx):
        """Update the time slice plotted.
        """
        # Cast to int, because Tkinter is inexplicably giving me str (!?)
        t_idx = int(t_idx)
        # Need to check this, since apparently getting cute with setting the
        # label creates an infinite recursion...
        if t_idx != self.slider_frame.t_idx:
            self.slider_frame.t_idx = t_idx
            self.slider_frame.t_slider.config(
                label="t = %.3fs" % (self.data.vuv_time[self.system][t_idx],)
            )
            remove_all(self.plot_frame.l_time)
            self.plot_frame.l_time = []
            self.plot_frame.l_time.append(
                self.plot_frame.a_spec.plot(
                    self.data.vuv_lam[self.system],
                    self.data.vuv_signal[self.system][t_idx, :],
                    'k'
                )
            )
            self.plot_frame.l_time.append(
                self.plot_frame.a_time.axvline(
                    self.data.vuv_time[self.system][t_idx],
                    color='b'
                )
            )
            self.plot_frame.l_time.append(
                self.plot_frame.a_im.axhline(
                    self.data.vuv_time[self.system][t_idx],
                    color='b'
                )
            )
            self.plot_frame.a_spec.relim()
            self.plot_frame.a_spec.autoscale_view(scalex=False)
            self.plot_frame.canvas.draw()
    
    def update_lam(self, lam_idx):
        """Update the wavelength slice plotted.
        """
        lam_idx = int(lam_idx)
        if lam_idx != self.slider_frame.lam_idx:
            self.slider_frame.lam_idx = lam_idx
            self.slider_frame.lam_slider.config(
                label=u"\u03bb = %.3fnm" % (self.data.vuv_lam[self.system][lam_idx],)
            )
            remove_all(self.plot_frame.l_lam)
            self.plot_frame.l_lam = []
            self.plot_frame.l_lam.append(
                self.plot_frame.a_time.plot(
                    self.data.vuv_time[self.system],
                    self.data.vuv_signal[self.system][:, lam_idx],
                    'k'
                )
            )
            self.plot_frame.l_lam.append(
                self.plot_frame.a_spec.axvline(
                    self.data.vuv_lam[self.system][lam_idx],
                    color='g'
                )
            )
            self.plot_frame.l_lam.append(
                self.plot_frame.a_im.axvline(
                    self.data.vuv_lam[self.system][lam_idx],
                    color='g'
                )
            )
            self.plot_frame.a_time.relim()
            self.plot_frame.a_time.autoscale_view(scalex=False)
            self.plot_frame.canvas.draw()
    
    def update_max_val(self, max_val):
        """Update the maximum value on the image plot.
        """
        max_val = float(max_val)
        if max_val != self.slider_frame.max_val:
            self.slider_frame.max_val = max_val
            self.plot_frame.im.set_clim(vmax=max_val)
            self.plot_frame.canvas.draw()

class VuvLineFrame(tk.Frame):
    """Frame that holds the controls to setup VUV line information.
    """
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
        # Keep track of the selected idx separately, since tkinter is stupid
        # about it (loses state when using tab to move between text boxes):
        self.idx = None
        
        self.listbox_label = tk.Label(self, text="defined lines:", anchor=tk.SW)
        self.listbox_label.grid(row=0, column=0, columnspan=2, sticky='NESW')
        
        self.listbox = tk.Listbox(self)
        self.listbox.grid(row=1, column=0, columnspan=2, sticky='NESW')
        self.listbox.bind('<<ListboxSelect>>', self.on_select)
        
        self.add_button = tk.Button(self, text="+", command=self.add_line)
        self.add_button.grid(row=2, column=0, sticky='NESW')
        
        self.remove_button = tk.Button(self, text="-", command=self.remove_line)
        self.remove_button.grid(row=2, column=1, sticky='NESW')
        
        self.included_lines_label = tk.Label(self, text="included lines:", anchor=tk.SW)
        self.included_lines_label.grid(row=3, column=0, columnspan=2, sticky='NESW')
        
        self.included_lines_box = tk.Entry(self)
        self.included_lines_box.grid(row=4, column=0, columnspan=2, sticky='NESW')
        
        self.lam_lb_label = tk.Label(self, text=u"\u03bb min (nm):", anchor=tk.SW)
        self.lam_lb_label.grid(row=5, column=0, sticky='NESW')
        
        self.lam_lb_box = tk.Entry(self)
        self.lam_lb_box.grid(row=6, column=0, sticky='NESW')
        
        self.lam_ub_label = tk.Label(self, text=u"\u03bb max (nm):", anchor=tk.SW)
        self.lam_ub_label.grid(row=5, column=1, sticky='NESW')
        
        self.lam_ub_box = tk.Entry(self)
        self.lam_ub_box.grid(row=6, column=1, sticky='NESW')
        
        self.t_lb_label = tk.Label(self, text="baseline start (s):", anchor=tk.SW)
        self.t_lb_label.grid(row=7, column=0, sticky='NESW')
        
        self.t_lb_box = tk.Entry(self)
        self.t_lb_box.grid(row=8, column=0, sticky='NESW')
        
        self.t_ub_label = tk.Label(self, text="baseline end (s):", anchor=tk.SW)
        self.t_ub_label.grid(row=7, column=1, sticky='NESW')
        
        self.t_ub_box = tk.Entry(self)
        self.t_ub_box.grid(row=8, column=1, sticky='NESW')
        
        self.apply_button = tk.Button(self, text="apply", command=self.on_apply)
        self.apply_button.grid(row=9, column=0, columnspan=2, sticky='NESW')
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Add the existing VuvLine instances to the GUI:
        if self.master.data.vuv_lines[self.master.system]:
            for l in self.master.data.vuv_lines[self.master.system]:
                self.listbox.insert(tk.END, ', '.join(map(str, l.diagnostic_lines)))
        else:
            self.add_line()
    
    def on_select(self, event):
        """Handle selection of a new line.
        """
        # TODO: This should save the current state into the selected line
        
        try:
            self.idx = int(self.listbox.curselection()[0])
        except IndexError:
            self.idx = None
        
        if self.idx is not None:
            self.included_lines_box.delete(0, tk.END)
            if self.master.data.vuv_lines[self.master.system][self.idx].diagnostic_lines is not None:
                self.included_lines_box.insert(
                    0,
                    ', '.join(
                        map(
                            str,
                            self.master.data.vuv_lines[self.master.system][self.idx].diagnostic_lines
                        )
                    )
                )
            
            self.lam_lb_box.delete(0, tk.END)
            if self.master.data.vuv_lines[self.master.system][self.idx].lam_lb is not None:
                self.lam_lb_box.insert(
                    0,
                    self.master.data.vuv_lines[self.master.system][self.idx].lam_lb
                )
            
            self.lam_ub_box.delete(0, tk.END)
            if self.master.data.vuv_lines[self.master.system][self.idx].lam_ub is not None:
                self.lam_ub_box.insert(
                    0,
                    self.master.data.vuv_lines[self.master.system][self.idx].lam_ub
                )
            
            if self.master.data.vuv_lines[self.master.system][self.idx].t_lb is not None:
                self.t_lb_box.delete(0, tk.END)
                self.t_lb_box.insert(
                    0,
                    self.master.data.vuv_lines[self.master.system][self.idx].t_lb
                )
            
            if self.master.data.vuv_lines[self.master.system][self.idx].t_ub is not None:
                self.t_ub_box.delete(0, tk.END)
                self.t_ub_box.insert(
                    0,
                    self.master.data.vuv_lines[self.master.system][self.idx].t_ub
                )
            
            remove_all(self.master.plot_frame.l_final)
            self.master.plot_frame.l_final = []
            if self.master.data.vuv_lines[self.master.system][self.idx].signal is not None:
                self.master.plot_frame.l_final.append(
                    self.master.plot_frame.a_final.plot(
                        self.master.data.vuv_time[self.master.system],
                        self.master.data.vuv_lines[self.master.system][self.idx].signal,
                        'k'
                    )
                )
            self.master.plot_frame.canvas.draw()
    
    def add_line(self):
        """Add a new (empty) line to the listbox.
        """
        self.master.data.vuv_lines[self.master.system].append(VuvLine(self.master.system))
        self.listbox.insert(tk.END, "unassigned")
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(tk.END)
        self.on_select(None)
    
    def remove_line(self):
        """Remove the currently-selected line from the listbox.
        """
        if self.idx is not None:
            self.master.data.vuv_lines[self.master.system].pop(self.idx)
            self.listbox.delete(self.idx)
            self.included_lines_box.delete(0, tk.END)
            self.lam_lb_box.delete(0, tk.END)
            self.lam_ub_box.delete(0, tk.END)
            # Don't clear the time boxes, since we will usually want the same
            # time window for baseline subtraction.
            # self.t_lb_box.delete(0, tk.END)
            # self.t_ub_box.delete(0, tk.END)
            self.idx = None
    
    def on_apply(self):
        """Apply the current settings and update the plot.
        """
        if self.idx is None:
            print("Please select a line to apply!")
            self.bell()
            return
        
        included_lines = re.findall(LIST_REGEX, self.included_lines_box.get())
        if len(included_lines) == 0:
            print("No lines to include!")
            self.bell()
            return
        try:
            included_lines = [int(l) for l in included_lines]
        except ValueError:
            print("Invalid entry in included lines!")
            self.bell()
            return
        
        try:
            lam_lb = float(self.lam_lb_box.get())
        except ValueError:
            print("Invalid lower bound for wavelength!")
            self.bell()
            return
        
        try:
            lam_ub = float(self.lam_ub_box.get())
        except ValueError:
            print("Invalid upper bound for wavelength!")
            self.bell()
            return
        
        try:
            t_lb = float(self.t_lb_box.get())
        except ValueError:
            print("Invalid baseline start!")
            self.bell()
            return
        
        try:
            t_ub = float(self.t_ub_box.get())
        except ValueError:
            print("Invalid baseline end!")
            self.bell()
            return
        
        xl = self.master.data.vuv_lines[self.master.system][self.idx]
        xl.diagnostic_lines = included_lines
        xl.lam_lb = lam_lb
        xl.lam_ub = lam_ub
        xl.t_lb = t_lb
        xl.t_ub = t_ub
        
        xl.process_bounds(self.master.data)
        
        self.listbox.delete(self.idx)
        self.listbox.insert(self.idx, ', '.join(map(str, included_lines)))
        
        remove_all(self.master.plot_frame.l_final)
        self.master.plot_frame.l_final = []
        self.master.plot_frame.l_final.append(
            self.master.plot_frame.a_final.plot(
                self.master.data.vuv_time[self.master.system],
                xl.signal,
                'k'
            )
        )
        self.master.plot_frame.canvas.draw()

class VuvLine(object):
    """Class to store information on a single VUV diagnostic line.
    
    The line may encapsulate more than one "diagnostic line" from the STRAHL
    output in case these lines overlap too much.
    
    Assumes you set the relevant attributes externally, then call
    :py:meth:`process_bounds`.
    
    Attributes
    ----------
    diagnostic_lines : list of int
        List of the indices of the lines included in the spectral region of the
        line.
    lam_lb : float
        Lower bound of wavelength to include (nm).
    lam_ub : float
        Upper bound of wavelength to include (nm).
    t_lb : float
        Lower bound of time to use for baseline subtraction.
    t_ub : float
        Upper bound of time to use for baseline subtraction.
    signal : array, (`N`,)
        The `N` timepoints of the combined, baseline-subtracted signal.
    """
    def __init__(self, system, diagnostic_lines=None, lam_lb=None, lam_ub=None, t_lb=None, t_ub=None):
        self.system = system
        self.diagnostic_lines = diagnostic_lines
        self.lam_lb = lam_lb
        self.lam_ub = lam_ub
        self.t_lb = t_lb
        self.t_ub = t_ub
        
        self.signal = None
        self.uncertainty = None
    
    def process_bounds(self, data):
        # Find the indices in the data:
        lam_lb_idx, lam_ub_idx = profiletools.get_nearest_idx(
            [self.lam_lb, self.lam_ub],
            data.vuv_lam[self.system]
        )
        t_lb_idx, t_ub_idx = profiletools.get_nearest_idx(
            [self.t_lb, self.t_ub],
            data.vuv_time[self.system]
        )
        
        # Form combined spectrum:
        # The indices are reversed for lambda vs. index:
        self.signal = data.vuv_signal[self.system][:, lam_ub_idx:lam_lb_idx + 1].sum(axis=1)
        
        # Perform the baseline subtraction:
        self.signal -= self.signal[t_lb_idx:t_ub_idx + 1].mean()
        
        # Compute the propagated uncertainty:
        self.uncertainty = (data.vuv_uncertainty[self.system][:, lam_ub_idx:lam_lb_idx + 1]**2).sum(axis=1)
        self.uncertainty += (self.uncertainty[t_lb_idx:t_ub_idx + 1]**2).sum() / (t_ub_idx - t_lb_idx + 1)**2
        self.uncertainty = scipy.sqrt(self.uncertainty)

class VuvPlotFrame(tk.Frame):
    """Frame to hold the plots with the XEUS data.
    """
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
        # Store the lines that change when updating the time:
        self.l_time = []
        
        # Store the lines that change when updating the wavelength:
        self.l_lam = []
        
        # Store the lines that change when updating the XEUS line:
        self.l_final = []
        
        self.f = Figure()
        self.suptitle = self.f.suptitle('')
        gs = mplgs.GridSpec(2, 2)
        self.a_im = self.f.add_subplot(gs[0, 0])
        self.a_spec = self.f.add_subplot(gs[1, 0])
        self.a_time = self.f.add_subplot(gs[0, 1])
        self.a_final = self.f.add_subplot(gs[1, 1])
        self.canvas = FigureCanvasTkAgg(self.f, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='NESW')
        
        # Need to put the toolbar in its own frame, since it automatically calls
        # pack on itself, but I am using grid.
        self.toolbar_frame = tk.Frame(self)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar_frame.grid(row=1, column=0, sticky='EW')
        
        self.canvas.mpl_connect(
            'button_press_event',
            lambda event: self.canvas._tkcanvas.focus_set()
        )
        self.canvas.mpl_connect('key_press_event', self.on_key_event)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Just plot the image now since it doesn't change:
        LAM, T = scipy.meshgrid(
            self.master.data.vuv_lam[self.master.system],
            self.master.data.vuv_time[self.master.system]
        )
        self.im = self.a_im.pcolormesh(
            LAM,
            T,
            self.master.data.vuv_signal[self.master.system],
            cmap='gray'
        )
        xlim = self.a_im.get_xlim()
        for x, i, c in zip(
                CA_17_LINES + CA_16_LINES,
                range(0, len(CA_17_LINES) + len(CA_16_LINES)),
                ['r'] * len(CA_17_LINES) + ['c'] * len(CA_16_LINES)
            ):
            self.a_im.axvline(x, linestyle='--', color=c)
            self.a_spec.axvline(x, linestyle='--', color=c)
            self.a_im.text(
                x,
                self.master.data.vuv_time[self.master.system].min(),
                str(i)
            )
            self.a_spec.text(x, 0, str(i))
        
        self.a_im.set_xlim(xlim)
        self.a_spec.set_xlim(xlim)
        
        self.a_im.set_xlabel(r'$\lambda$ [nm]')
        self.a_im.set_ylabel('$t$ [s]')
        
        self.a_spec.set_xlabel(r'$\lambda$ [nm]')
        self.a_spec.set_ylabel('raw signal [AU]')
        
        self.a_time.set_xlabel('$t$ [s]')
        self.a_time.set_ylabel('raw signal [AU]')
        
        self.a_final.set_xlabel('$t$ [s]')
        self.a_final.set_ylabel('processed signal [AU]')
    
    def on_key_event(self, evt):
        """Tie keys to the toolbar.
        """
        key_press_handler(evt, self.canvas, self.toolbar)
    
    def on_click(self, evt):
        """Move the cursors with a click in any given axis.
        
        Only does so if the widgetlock is not locked.
        """
        if not self.canvas.widgetlock.locked():
            if evt.inaxes == self.a_im:
                # Update both lam and t:
                lam_idx = profiletools.get_nearest_idx(
                    evt.xdata,
                    self.master.data.vuv_lam[self.master.system]
                )
                self.master.slider_frame.lam_slider.set(lam_idx)
                
                t_idx = profiletools.get_nearest_idx(
                    evt.ydata,
                    self.master.data.vuv_time[self.master.system]
                )
                self.master.slider_frame.t_slider.set(t_idx)
            elif evt.inaxes == self.a_spec:
                # Only update lam:
                lam_idx = profiletools.get_nearest_idx(
                    evt.xdata,
                    self.master.data.vuv_lam[self.master.system]
                )
                self.master.slider_frame.lam_slider.set(lam_idx)
            elif evt.inaxes == self.a_time:
                # Only update t:
                t_idx = profiletools.get_nearest_idx(
                    evt.xdata,
                    self.master.data.vuv_time[self.master.system]
                )
                self.master.slider_frame.t_slider.set(t_idx)

class VuvSliderFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
        self.t_idx = None
        self.lam_idx = None
        self.max_val = None
        
        self.t_slider = tk.Scale(
            master=self,
            from_=0,
            to=len(self.master.data.vuv_time[self.master.system]) - 1,
            command=self.master.update_t,
            orient=tk.HORIZONTAL,
            label='t'
        )
        self.t_slider.grid(row=0, column=0)
        
        self.lam_slider = tk.Scale(
            master=self,
            from_=0,
            to=len(self.master.data.vuv_lam[self.master.system]) - 1,
            command=self.master.update_lam,
            orient=tk.HORIZONTAL,
            label=u'\u03bb'
        )
        self.lam_slider.grid(row=0, column=1)
        
        self.max_val_slider = tk.Scale(
            master=self,
            from_=0,
            to=self.master.data.vuv_signal[self.master.system].max(),
            command=self.master.update_max_val,
            orient=tk.HORIZONTAL,
            label='max =',
            resolution=0.01
        )
        self.max_val_slider.set(self.master.data.vuv_signal[self.master.system].max()/2)
        self.max_val_slider.grid(row=0, column=2)




class Signal(object):
    def __init__(self, y, std_y, y_norm, std_y_norm, t, name, atomdat_idx, pos=None, sqrtpsinorm=None, weights=None, blocks=0,m=None,s=None ):
        """Class to store the data from a given diagnostic.
        
        In the parameter descriptions, `n` is the number of signals (both
        spatial and temporal) contained in the instance.
        
        Parameters
        ----------
        y : array, (`n_time`, `n`)
            The unnormalized, baseline-subtracted data as a function of time and
            space. If `pos` is not None, "space" refers to the chords. Wherever
            there is a bad point, it should be set to NaN.
        std_y : array, (`n_time`, `n`)
            The uncertainty in the unnormalized, baseline-subtracted data as a
            function of time and space.
        y_norm : array, (`n_time`, `n`)
            The normalized, baseline-subtracted data.
        std_y_norm : array, (`n_time`, `n`)
            The uncertainty in the normalized, baseline-subtracted data.
        t : array, (`n_time`,)
            The time vector of the data.
        name : str
            The name of the signal.
        atomdat_idx : int or array of int, (`n`,)
            The index or indices of the signals in the atomdat file. If a single
            value is given, it is used for all of the signals. If a 1d array is
            provided, these are the indices for each of the signals in `y`. If
            `atomdat_idx` (or one of its entries) is -1, it will be treated as
            an SXR measurement.
        pos : array, (4,) or (`n`, 4), optional
            The POS vector(s) for line-integrated data. If not present, the data
            are assumed to be local measurements at the locations in
            `sqrtpsinorm`. If a 1d array is provided, it is used for all of the
            chords in `y`. Otherwise, there must be one pos vector for each of
            the chords in `y`.
        sqrtpsinorm : array, (`n`,), optional
            The square root of poloidal flux grid the (local) measurements are
            given on. If line-integrated measurements with the standard STRAHL
            grid for their quadrature points are to be used this should be left
            as None.
        weights : array, (`n`, `n_quadrature`), optional
            The quadrature weights to use. This can be left as None for a local
            measurement or can be set later.
        blocks : int or array of int, (`n`), optional
            A set of flags indicating which channels in the :py:class:`Signal`
            should be treated together as a block when normalizing. If a single
            int is given, all of the channels will be taken together. Otherwise,
            any channels sharing the same block number will be taken together.
        m : float
            maximum signal recorded across any chords and any time for this diagnostic.
            This value is used for normalization of the signals. 
        s : float
            uncertainty in m (see above)
        """
        self.y = scipy.asarray(y, dtype=float)
        if self.y.ndim != 2:
            raise ValueError("y must have two dimensions!")
        self.std_y = scipy.asarray(std_y, dtype=float)
        if self.y.shape != self.std_y.shape:
            raise ValueError("The shapes of y and std_y must match!")
        self.y_norm = scipy.asarray(y_norm, dtype=float)
        if self.y.shape != self.y_norm.shape:
            raise ValueError("The shapes of y and y_norm must match!")
        self.std_y_norm = scipy.asarray(std_y_norm, dtype=float)
        if self.std_y_norm.shape != self.y.shape:
            raise ValueError("The shapes of y and std_y_norm must match!")
        self.t = scipy.asarray(t, dtype=float)
        if self.t.ndim != 1:
            raise ValueError("t must have one dimension!")
        if len(self.t) != self.y.shape[0]:
            raise ValueError("The length of t must equal the length of the leading dimension of y!")
        if isinstance(name, str):
            name = [name,] * self.y.shape[1]
        self.name = name
        try:
            iter(atomdat_idx)
        except TypeError:
            self.atomdat_idx = atomdat_idx * scipy.ones(self.y.shape[1], dtype=int)
        else:
            self.atomdat_idx = scipy.asarray(atomdat_idx, dtype=int)
            if self.atomdat_idx.ndim != 1:
                raise ValueError("atomdat_idx must have at most one dimension!")
            if len(self.atomdat_idx) != self.y.shape[1]:
                raise ValueError("1d atomdat_idx must have the same number of elements as the second dimension of y!")
        if pos is not None:
            pos = scipy.asarray(pos, dtype=float)
            if pos.ndim not in (1, 2):
                raise ValueError("pos must have one or two dimensions!")
            if pos.ndim == 1 and len(pos) != 4:
                raise ValueError("pos must have 4 elements!")
            if pos.ndim == 2 and (pos.shape[0] != self.y.shape[1] or pos.shape[1] != 4):
                raise ValueError("pos must have shape (n, 4)!")
        
        self.pos = pos
        self.sqrtpsinorm = sqrtpsinorm
        
        self.weights = weights
        
        try:
            iter(blocks)
        except TypeError:
            self.blocks = blocks * scipy.ones(self.y.shape[1], dtype=int)
        else:
            self.blocks = scipy.asarray(blocks, dtype=int)
            if self.blocks.ndim != 1:
                raise ValueError("blocks must have at most one dimension!")
            if len(self.blocks) != self.y.shape[1]:
                raise ValueError("1d blocks must have the same number of elements as the second dimension of y!")
        
        if isinstance(m,(float)):
            self.m=m
        elif m==None: 
            pass
        else:
            raise ValueError("maximum signal m must be a float!")
        if isinstance(s,(float)):
            self.s=s
        elif s==None:
            pass
        else: 
            raise ValueError("maximum signal m must be a float!")

    def sort_t(self):
        """Sort the time axis.
        """
        srt = self.t.argsort()
        self.t = self.t[srt]
        self.y = self.y[srt, :]
        self.std_y = self.std_y[srt, :]
        self.y_norm = self.y_norm[srt, :]
        self.std_y_norm = self.std_y_norm[srt, :]
    
    def plot_data(self, norm=False, f=None, share_y=False, y_label='$b$ [AU]',
                  max_ticks=None, rot_label=False, fast=False, ncol=6):
        """Make a big plot with all of the data.
        
        Parameters
        ----------
        norm : bool, optional
            If True, plot the normalized data. Default is False (plot
            unnormalized data).
        f : :py:class:`Figure`, optional
            The figure instance to make the subplots in. If not provided, a
            figure will be created.
        share_y : bool, optional
            If True, the y axes of all of the subplots will have the same scale.
            Default is False (each y axis is automatically scaled individually).
        y_label : str, optional
            The label to use for the y axes. Default is '$b$ [AU]'.
        max_ticks : int, optional
            The maximum number of ticks on the x and y axes. Default is no limit.
        rot_label : bool, optional
            If True, the x axis labels will be rotated 90 degrees. Default is
            False (do not rotate).
        fast : bool, optional
            If True, errorbars will not be drawn in order to make the plotting
            faster. Default is False
        ncol : int, optional
            The number of columns to use. Default is 6.
        """
        if norm:
            y = self.y_norm
            std_y = self.std_y_norm
        else:
            y = self.y
            std_y = self.std_y
        
        if f is None:
            f = plt.figure()
        
        ncol = int(min(ncol, self.y.shape[1]))
        nrow = int(scipy.ceil(1.0 * self.y.shape[1] / ncol))
        gs = mplgs.GridSpec(nrow, ncol)
        
        a = []
        i_col = 0
        i_row = 0
        
        for k in xrange(0, self.y.shape[1]):
            a.append(
                f.add_subplot(
                    gs[i_row, i_col],
                    sharex=a[0] if len(a) >= 1 else None,
                    sharey=a[0] if len(a) >= 1 and share_y else None
                )
            )
            if i_col > 0 and share_y:
                plt.setp(a[-1].get_yticklabels(), visible=False)
            else:
                a[-1].set_ylabel(y_label)
            if i_row < nrow - 2 or (i_row == nrow - 2 and i_col < self.y.shape[1] % (nrow - 1)):
                plt.setp(a[-1].get_xticklabels(), visible=False)
            else:
                a[-1].set_xlabel('$t$ [s]')
                if rot_label:
                    plt.setp(a[-1].xaxis.get_majorticklabels(), rotation=90)
            i_col += 1
            if i_col >= ncol:
                i_col = 0
                i_row += 1
            a[-1].set_title('%s, %d' % (self.name[k], k))
            good = ~scipy.isnan(self.y[:, k])
            if fast:
                a[-1].plot(self.t[good], y[good, k], '.')
            else:
                a[-1].errorbar(self.t[good], y[good, k], yerr=std_y[good, k], fmt='.')
            if max_ticks is not None:
                a[-1].xaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks - 1))
                a[-1].yaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks - 1))
        
        if share_y:
            a[0].set_ylim(bottom=0.0)
            a[0].set_xlim(self.t.min(), self.t.max())
        
        f.canvas.draw()
        
        return (f, a)


def remove_all(v):
    """ Recursive remover, because matplotlib is stupid.
    """
    try:
        for vv in v:
            remove_all(vv)
    except TypeError:
        v.remove()


def interp_max(x, y, err_y=None, s_guess=0.2, s_max=10.0, l_guess=0.005, fixed_l=False, debug_plots=False, method='GP'):
    """Compute the maximum value of the smoothed data.
    
    Estimates the uncertainty using Gaussian process regression and returns the
    mean and uncertainty.
    
    Parameters
    ----------
    x : array of float
        Abscissa of data to be interpolated.
    y : array of float
        Data to be interpolated.
    err_y : array of float, optional
        Uncertainty in `y`. If absent, the data are interpolated.
    s_guess : float, optional
        Initial guess for the signal variance. Default is 0.2.
    s_max : float, optional
        Maximum value for the signal variance. Default is 10.0
    l_guess : float, optional
        Initial guess for the covariance length scale. Default is 0.03.
    fixed_l : bool, optional
        Set to True to hold the covariance length scale fixed during the MAP
        estimate. This helps mitigate the effect of bad points. Default is True.
    debug_plots : bool, optional
        Set to True to plot the data, the smoothed curve (with uncertainty) and
        the location of the peak value.
    method : {'GP', 'spline'}, optional
        Method to use when interpolating. Default is 'GP' (Gaussian process
        regression). Can also use a cubic spline.
    """
    grid = scipy.linspace(max(0, x.min()), min(0.08, x.max()), 1000)
    if method == 'GP':
        hp = (
            gptools.UniformJointPrior([(0, s_max),]) *
            gptools.GammaJointPriorAlt([l_guess,], [0.1,])
        )
        k = gptools.SquaredExponentialKernel(
            # param_bounds=[(0, s_max), (0, 2.0)],
            hyperprior=hp,
            initial_params=[s_guess, l_guess],
            fixed_params=[False, False]
        )
        gp = gptools.GaussianProcess(k, X=x, y=y, err_y=err_y)
        gp.optimize_hyperparameters(verbose=True, random_starts=100)
        m_gp, s_gp = gp.predict(grid)
        i = m_gp.argmax()
    elif method == 'spline':
        m_gp = scipy.interpolate.UnivariateSpline(
            x, y, w=1.0 / err_y, s=2*len(x)
        )(grid)
        if scipy.isnan(m_gp).any():
            print(x)
            print(y)
            print(err_y)
        i = m_gp.argmax()
    else:
        raise ValueError("Undefined method %s" % (method,))
    
    if debug_plots:
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        a.errorbar(x, y, yerr=err_y, fmt='.', color='b')
        a.plot(grid, m_gp, color='g')
        if method == 'GP':
            a.fill_between(grid, m_gp - s_gp, m_gp + s_gp, color='g', alpha=0.5)
        a.axvline(grid[i])
    
    if method == 'GP':
        return (m_gp[i], s_gp[i])
    else:
        return m_gp[i]





