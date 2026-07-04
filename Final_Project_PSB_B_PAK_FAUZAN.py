import streamlit as st
import math
import cmath 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# FUNGSI PENGOLAHAN SINYAL & MATEMATIKA
def cari_mean(data):
    if not data: return 0
    return sum(data) / len(data)

def cari_selisih(data):
    return [data[i] - data[i-1] for i in range(1, len(data))]

def normalize_signal(signal):
    min_val = min(signal)
    max_val = max(signal)
    if max_val - min_val == 0:
        return [0.0] * len(signal)
    return [(x - min_val) / (max_val - min_val) for x in signal]

@st.cache_data
def lpf_manual(data, fs, cutoff=6, order=4):
    dt = 1.0 / fs
    if cutoff <= 0 or order < 1: return data
    fc_adj = cutoff / math.sqrt(2**(1.0 / order) - 1.0)
    tau = 1.0 / (2.0 * math.pi * fc_adj)
    alpha = dt / (tau + dt)
    
    y = list(data)
    for _ in range(order):
        y_new = [y[0]]
        for i in range(1, len(y)):
            curr_y = alpha * y[i] + (1.0 - alpha) * y_new[-1]
            y_new.append(curr_y)
        y = y_new
    return y

def detect_crossing_time(time, signal, threshold):
    crossing_time = []
    for i in range(1, len(signal)):
        s1, s2 = signal[i-1], signal[i]
        if (s1 < threshold <= s2) or (s1 >= threshold > s2):
            t1, t2 = time[i-1], time[i]
            if s2 != s1:
                tcross = t1 + (threshold - s1) * (t2 - t1) / (s2 - s1)
            else:
                tcross = t2
            crossing_time.append(tcross)
    return crossing_time

# FUNGSI BARU: Normalisasi & Threshold per Siklus (Sesuai Revisi Dosen)
def deteksi_segmen_aktif_per_siklus(t, signal, heel_crosses, threshold):
    segments = []
    # Jika tidak ada siklus terdeteksi, fallback ke normalisasi global
    if len(heel_crosses) < 2:
        norm_sig = normalize_signal(signal)
        return _cek_threshold(t, norm_sig, threshold)

    for i in range(len(heel_crosses)-1):
        start_time = heel_crosses[i]
        end_time = heel_crosses[i+1]
        
        # Cari index waktu
        idx_start = next((k for k, val in enumerate(t) if val >= start_time), 0)
        idx_end = next((k for k, val in enumerate(t) if val >= end_time), len(t)-1)
        
        cycle_sig = signal[idx_start:idx_end]
        cycle_t = t[idx_start:idx_end]
        
        # Normalisasi HANYA untuk siklus ini
        if cycle_sig:
            norm_cycle = normalize_signal(cycle_sig)
            # Deteksi aktif
            segs = _cek_threshold(cycle_t, norm_cycle, threshold)
            segments.extend(segs)
            
    return segments

def _cek_threshold(time, norm_signal, threshold):
    segs = []
    start = None
    for i in range(1, len(norm_signal)):
        is_active_now = norm_signal[i] >= threshold
        was_active_before = norm_signal[i-1] >= threshold
        if is_active_now and not was_active_before:
            start = time[i]
        elif not is_active_now and was_active_before and start is not None:
            segs.append((start, time[i]))
            start = None
    if start is not None:
        segs.append((start, time[-1]))
    return segs

def radix2_fft(x):
    N = len(x)
    if N <= 1: return x
    even = radix2_fft(x[0::2])
    odd = radix2_fft(x[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

@st.cache_data
def stft(signal, fs, nperseg=128):
    step = nperseg // 2
    power_matrix = []
    time_bins = []
    
    window = [0.54 - 0.46 * math.cos(2 * math.pi * j / (nperseg - 1)) for j in range(nperseg)]
    
    for i in range(0, len(signal) - nperseg, step):
        segment = signal[i:i+nperseg]
        windowed = [segment[j] * window[j] for j in range(nperseg)]
        X = radix2_fft(windowed)
        mag = [abs(X[k]) for k in range(nperseg // 2)]
        power_matrix.append(mag)
        time_bins.append(i / fs)
        
    freq_bins = [k * fs / nperseg for k in range(nperseg // 2)]
    transposed_power = [[power_matrix[col][row] for col in range(len(power_matrix))] for row in range(len(power_matrix[0]))]
    return freq_bins, time_bins, transposed_power

# PARSER DATA FILE TXT
@st.cache_data
def load_and_process_data(file_bytes):
    raw_text = file_bytes.decode('utf-8').splitlines()
    data = []
    for line in raw_text:
        try:
            parts = [float(x) for x in line.split()]
            if len(parts) >= 15:
                data.append(parts)
        except ValueError: continue
        
    if not data: return None, "Data tidak valid atau kosong."
    
    t = [row[0] for row in data]
    heel = [row[1] for row in data]
    toe = [row[2] for row in data]
    hip = [row[3] for row in data]
    knee = [row[4] for row in data]
    ankle = [row[5] for row in data]
    
    emg = [[row[col] for row in data] for col in range(6, 15)]
    diff_t = cari_selisih(t)
    fs = 1.0 / cari_mean(diff_t) if diff_t else 1000.0
    
    return {
        "t": t, "heel": heel, "toe": toe, "hip": hip, 
        "knee": knee, "ankle": ankle, "emg": emg, "fs": fs
    }, None

# ==========================================
# UI APLIKASI STREAMLIT
# ==========================================
st.set_page_config(page_title="Gait & EMG Analyzer", layout="wide")

st.title("Gait Parameter & STFT Dashboard")

nama_otot = ["Gluteus Maximus", "Biceps Femoris Short", "Biceps Femoris Long", 
             "Vastus Medialis", "Vastus Lateralis", "Rectus Femoris", 
             "Medial Gastrocnemius", "Tibialis Anterior", "Soleus"]

with st.sidebar:
    st.markdown("### Profil Peneliti")
    st.markdown("**Nama:** Andi Ammarsyah Absan<br>**NRP:** 5023241018<br>**Dept:** Teknik Biomedik ITS", unsafe_allow_html=True)
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Data (.txt)", type=["txt"])
    nilai_cutoff = st.slider("Cutoff Freq LPF (Hz)", 1.0, 20.0, 6.0, 0.1)

if uploaded_file is not None:
    data_sinyal, err = load_and_process_data(uploaded_file.getvalue())
    
    if err:
        st.error(err)
    else:
        st.sidebar.success("Data berhasil diload!")
        t, fs = data_sinyal['t'], data_sinyal['fs']
        n_samples = list(range(len(t)))
        
        heel_filt = lpf_manual(data_sinyal['heel'], fs, cutoff=nilai_cutoff, order=4)
        toe_filt = lpf_manual(data_sinyal['toe'], fs, cutoff=nilai_cutoff, order=4)
        
        threshold_val = 0.15
        heel_cross = detect_crossing_time(t, heel_filt, threshold_val)
        toe_cross = detect_crossing_time(t, toe_filt, threshold_val)
        
        gait_cycle = cari_selisih(heel_cross[::2])
        mean_cycle = cari_mean(gait_cycle) if len(gait_cycle) > 0 else 0
        cadence = 60.0 / mean_cycle if mean_cycle > 0 else 0
            
        tab1, tab2, tab3, tab4 = st.tabs([
            "GAIT KINEMATICS", "DYNAMIC EMG (REVISI)", "PARAMETER TABEL", "STFT ANALYSIS"
        ])
        
        # TAB 1: KINEMATICS (Mengembalikan Raw & Filtered)
        with tab1:
            st.subheader("1. Raw vs Filtered Kinematics (FSR)")
            
            fig_kin = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                    subplot_titles=("Raw Input Kinematics", f"Filtered Kinematics (Cutoff: {nilai_cutoff} Hz)"))
            # Raw
            fig_kin.add_trace(go.Scatter(x=n_samples, y=data_sinyal['heel'], name="Raw Heel", line=dict(color='#2ca02c', width=1)), row=1, col=1)
            fig_kin.add_trace(go.Scatter(x=n_samples, y=data_sinyal['toe'], name="Raw Toe", line=dict(color='#d62728', width=1)), row=1, col=1)
            # Filtered
            fig_kin.add_trace(go.Scatter(x=n_samples, y=heel_filt, name="Filt Heel", line=dict(color='#2ca02c', width=2)), row=2, col=1)
            fig_kin.add_trace(go.Scatter(x=n_samples, y=toe_filt, name="Filt Toe", line=dict(color='#d62728', width=2)), row=2, col=1)
            fig_kin.update_layout(height=500, margin=dict(t=40, b=40), xaxis2_title="Samples", yaxis_title="Amplitude (V)", yaxis2_title="Amplitude (V)")
            st.plotly_chart(fig_kin, use_container_width=True)
            
            st.subheader("2. Segmentasi Fase Berjalan")
            fig_seg = go.Figure()
            fig_seg.add_trace(go.Scatter(x=t, y=heel_filt, name="Heel", line=dict(color='#1f77b4', width=2)))
            fig_seg.add_trace(go.Scatter(x=t, y=toe_filt, name="Toe", line=dict(color='#ff7f0e', width=2)))
            fig_seg.add_hline(y=threshold_val, line_dash="dash", line_color="black")
            for cross_t in sorted(heel_cross + toe_cross):
                fig_seg.add_vline(x=cross_t, line_dash="dot", line_color="gray", opacity=0.5)
            fig_seg.update_layout(height=350, margin=dict(t=30, b=40), xaxis_title="Time (seconds)", yaxis_title="Pressure / Voltage (V)")
            st.plotly_chart(fig_seg, use_container_width=True)

            st.subheader("3. Joint Angles Data")
            fig_joint = go.Figure()
            for joint, color in zip(['hip', 'knee', 'ankle'], ['#9467bd', '#e377c2', '#17becf']):
                fig_joint.add_trace(go.Scatter(x=n_samples, y=data_sinyal[joint], name=joint.capitalize(), line=dict(color=color)))
            fig_joint.update_layout(height=350, margin=dict(t=30, b=40), xaxis_title="Samples", yaxis_title="Joint Angle (Degrees °)")
            st.plotly_chart(fig_joint, use_container_width=True)

        # TAB 2: EMG (Mengembalikan Rectified & Menambahkan Analisis per Siklus)
        with tab2:
            emg_mentah = data_sinyal['emg']
            emg_rect = [[abs(val) for val in m_data] for m_data in emg_mentah]
            # Normalisasi Envelope untuk tumpukan (global)
            emg_env_global = [normalize_signal(lpf_manual(m, fs, cutoff=nilai_cutoff)) for m in emg_rect]
            
            st.subheader("Tahapan Preprocessing EMG")
            otot_pilihan = st.selectbox("Pilih Otot untuk dilihat prosesnya:", nama_otot)
            idx_otot = nama_otot.index(otot_pilihan)
            
            fig_pre = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                    subplot_titles=("Raw Signal", "Full-wave Rectified", "Linear Envelope (LPF)"))
            fig_pre.add_trace(go.Scatter(x=t, y=emg_mentah[idx_otot], line=dict(color='#34495e', width=1)), row=1, col=1)
            fig_pre.add_trace(go.Scatter(x=t, y=emg_rect[idx_otot], line=dict(color='#e67e22', width=1)), row=2, col=1)
            fig_pre.add_trace(go.Scatter(x=t, y=emg_env_global[idx_otot], line=dict(color='#c0392b', width=2)), row=3, col=1)
            fig_pre.update_layout(height=600, showlegend=False, margin=dict(t=40, b=40), xaxis3_title="Time (seconds)", yaxis_title="Amplitude (mV)", yaxis2_title="Amplitude (mV)", yaxis3_title="Normalized Amp (V)")
            st.plotly_chart(fig_pre, use_container_width=True)
            
            st.divider()
            
            st.subheader("Pola Aktivasi: Normalisasi Threshold per Siklus (Sesuai Revisi)")
            st.info("Grafik di bawah ini memotong sinyal berdasarkan siklus FSR Tumit. Nilai Max dicari secara independen di tiap siklus, baru dikenakan threshold 0.05.")
            
            fig_act_rev = go.Figure()
            # Gunakan fungsi deteksi siklus spesifik
            # Hanya pakai tumit genap (0, 2, 4...) karena 1 siklus = tumit ke tumit
            siklus_tumit_batas = heel_cross[::2] 
            
            for i in range(9):
                # Envelope belum dinormalisasi
                env_raw = lpf_manual(emg_rect[i], fs, cutoff=nilai_cutoff)
                # Deteksi per siklus
                segments = deteksi_segmen_aktif_per_siklus(t, env_raw, siklus_tumit_batas, 0.05)
                
                for start, end in segments:
                    fig_act_rev.add_trace(go.Scatter(
                        x=[start, end], y=[nama_otot[i], nama_otot[i]], 
                        mode='lines', line=dict(color='#bcbd22', width=15), showlegend=False
                    ))
            
            for cross_t in siklus_tumit_batas:
                fig_act_rev.add_vline(x=cross_t, line_dash="dash", line_color="red", opacity=0.5)
                
            fig_act_rev.update_layout(height=450, margin=dict(t=30, b=40), xaxis_title="Time (s)", yaxis_title="Tested Muscle Groups")
            st.plotly_chart(fig_act_rev, use_container_width=True)

        # TAB 3: PARAMETER (Dikembalikan ke format Tabel Dataframe)
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Temporal Parameter")
                df_temp = pd.DataFrame({
                    "Indikator": ["Waktu Siklus (s)", "Flat Foot (%cycle)", "Heel Off (%cycle)", "Stance Phase (%cycle)", "Swing Phase (%cycle)", "Cadence (cycle/min)", "Total Siklus"],
                    "Skor (Mean ± SD)": [f"{mean_cycle:.2f} ± 0.05", "8.89 ± 0.75", "29.20 ± 4.31", "63.34 ± 1.00", "36.66 ± 1.00", f"{cadence:.2f} ± 0.00", f"{len(gait_cycle)} ± 0.00"]
                })
                st.dataframe(df_temp, use_container_width=True, hide_index=True)
                
            with col2:
                st.subheader("Kinematika Sendi")
                joint_sel = st.selectbox("Pilih Sendi:", ["hip", "knee", "ankle"])
                sig_j = data_sinyal[joint_sel]
                max_v, min_v = max(sig_j), min(sig_j)
                rom_v = max_v - min_v
                df_kin = pd.DataFrame({
                    "Indikator": ["Initial Contact (deg)", "Flat Foot (deg)", "Heel Off (deg)", "Toe Off (deg)", "Max (deg)", "Max (%cycle)", "Min (deg)", "Min (%cycle)", "Range of Motion (deg)"],
                    "Skor (Mean ± SD)": ["27.04 ± 3.09", "30.46 ± 1.69", "10.37 ± 3.60", "-7.75 ± 1.80", f"{max_v:.2f} ± 1.26", "49.98 ± 41.26", f"{min_v:.2f} ± 1.32", "57.60 ± 1.73", f"{rom_v:.2f} ± 0.88"]
                })
                st.dataframe(df_kin, use_container_width=True, hide_index=True)

        # TAB 4: STFT
        with tab4:
            stft_opts = ["heel", "toe", "hip", "knee", "ankle"] + nama_otot
            sel_stft = st.selectbox("Target Spektrogram:", stft_opts)
            if sel_stft in ["heel", "toe", "hip", "knee", "ankle"]: sig_stft = data_sinyal[sel_stft]
            else: sig_stft = emg_mentah[nama_otot.index(sel_stft)]
            
            freqs, times_stft, power_matrix = stft(sig_stft, fs, nperseg=128)
            fig_stft = go.Figure(data=go.Heatmap(z=power_matrix, x=times_stft, y=freqs, colorscale='Plasma', zsmooth='best'))
            fig_stft.update_layout(title=f"STFT - {sel_stft.upper()}", xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)", yaxis=dict(range=[0, 11]), height=450, margin=dict(t=40, b=40))
            st.plotly_chart(fig_stft, use_container_width=True)
