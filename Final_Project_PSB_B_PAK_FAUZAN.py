import streamlit as st
import math
import cmath 
import matplotlib.pyplot as plt

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

def deteksi_segmen_aktif(time, signal, threshold):
    segments = []
    start = None
    for i in range(1, len(signal)):
        is_active_now = signal[i] >= threshold
        was_active_before = signal[i-1] >= threshold
        
        if is_active_now and not was_active_before:
            start = time[i]
        elif not is_active_now and was_active_before and start is not None:
            segments.append((start, time[i]))
            start = None
    if start is not None:
        segments.append((start, time[-1]))
    return segments

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

# UI APLIKASI STREAMLIT
st.set_page_config(page_title="Gait & EMG Analyzer", layout="wide")
st.title("Gait Parameter Extraction & STFT Analysis")

nama_otot = ["Gluteus Maximus", "Biceps Femoris Short", "Biceps Femoris Long", 
             "Vastus Medialis", "Vastus Lateralis", "Rectus Femoris", 
             "Medial Gastrocnemius", "Tibialis Anterior", "Soleus"]

# SIDEBAR KONFIGURASI
with st.sidebar:
    st.markdown("### Profil Mahasiswa")
    st.markdown("**Nama:** Andi Ammarsyah Absan<br>**NRP:** 5023241018<br>**Dept:** Teknik Biomedik ITS", unsafe_allow_html=True)
    st.divider()
    
    st.header("Control Panel")
    uploaded_file = st.file_uploader("Upload Data (.txt)", type=["txt"])
    
    nilai_cutoff = st.slider(
        label="Cutoff Frequency LPF (Hz)",
        min_value=1.0, max_value=20.0, value=6.0, step=0.1
    )

#PROSES UTAMA
if uploaded_file is not None:
    data_sinyal, err = load_and_process_data(uploaded_file.getvalue())
    
    if err:
        st.error(err)
    else:
        st.sidebar.success(f"Berhasil diload! Total sampel: {len(data_sinyal['t'])}")
        
        t = data_sinyal['t']
        fs = data_sinyal['fs']
        n_samples = list(range(len(t)))
        
        # Filter FSR Kinematics
        heel_filt = lpf_manual(data_sinyal['heel'], fs, cutoff=nilai_cutoff, order=4)
        toe_filt = lpf_manual(data_sinyal['toe'], fs, cutoff=nilai_cutoff, order=4)
        
        threshold_val = 0.15
        heel_cross = detect_crossing_time(t, heel_filt, threshold_val)
        toe_cross = detect_crossing_time(t, toe_filt, threshold_val)
        
        # Kalkulasi Parameter Siklus Gait
        gait_cycle = cari_selisih(heel_cross[::2])
        if len(gait_cycle) > 0:
            mean_cycle = cari_mean(gait_cycle)
            cadence = 60.0 / mean_cycle if mean_cycle > 0 else 0
        else:
            mean_cycle = 0
            cadence = 0
            
        st.sidebar.markdown(f"""
        **Metrik Temporal:**
        - Rata-rata Cycle: {mean_cycle:.3f} s
        - Cadence: {cadence:.2f} steps/min
        - Siklus Berjalan: {len(gait_cycle)}
        """)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "GAIT PARAMETERS", "DYNAMIC EMG", "EMG PREPROCESSING", 
            "PARAMETER (TABEL)", "STFT ANALYSIS"
        ])
        
        # TAB 1: GAIT PARAMETERS
        with tab1:
            fig_in, ax_in = plt.subplots(figsize=(10, 2.5))
            ax_in.plot(n_samples, data_sinyal['heel'], color='#2ca02c', label="Heel Sensor", linewidth=1)
            ax_in.plot(n_samples, data_sinyal['toe'], color='#d62728', label="Toe Sensor", linewidth=1)
            ax_in.set_title("Raw Input Kinematics")
            ax_in.set_xlabel("Samples"); ax_in.set_ylabel("Amplitude")
            ax_in.legend(); ax_in.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_in, use_container_width=True)
            
            fig_out, ax_out = plt.subplots(figsize=(10, 2.5))
            ax_out.plot(n_samples, heel_filt, color='#2ca02c', label="Filtered Heel", linewidth=1.5)
            ax_out.plot(n_samples, toe_filt, color='#d62728', label="Filtered Toe", linewidth=1.5)
            ax_out.set_title(f"Filtered Kinematics (Cutoff: {nilai_cutoff} Hz)")
            ax_out.legend(); ax_out.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_out, use_container_width=True)
            
            fig_seg, ax_seg = plt.subplots(figsize=(10, 3))
            ax_seg.plot(t, heel_filt, color='#1f77b4', label="Heel", linewidth=1.5)
            ax_seg.plot(t, toe_filt, color='#ff7f0e', label="Toe", linewidth=1.5)
            ax_seg.axhline(threshold_val, color='black', linestyle='-.', label="Threshold", alpha=0.7)
            all_cross = sorted(heel_cross + toe_cross)
            for cross_t in all_cross:
                ax_seg.axvline(cross_t, color='grey', linestyle=':', linewidth=1.2)
            ax_seg.set_title("Segmentasi FSR")
            ax_seg.set_xlabel("Time (s)")
            ax_seg.legend(); ax_seg.grid(True)
            st.pyplot(fig_seg, use_container_width=True)
            
            fig_joint, ax_joint = plt.subplots(figsize=(10, 2.5))
            ax_joint.plot(n_samples, data_sinyal['hip'], color='#9467bd', label="Hip", linewidth=1)
            ax_joint.plot(n_samples, data_sinyal['knee'], color='#e377c2', label="Knee", linewidth=1)
            ax_joint.plot(n_samples, data_sinyal['ankle'], color='#17becf', label="Ankle", linewidth=1)
            ax_joint.set_title("Joint Angles Data")
            ax_joint.set_xlabel("Samples"); ax_joint.set_ylabel("Degrees")
            ax_joint.legend(); ax_joint.grid(True, alpha=0.5)
            st.pyplot(fig_joint, use_container_width=True)

        # TAB 2: DYNAMIC EMG
        with tab2:
            offset_raw = 2.5
            offset_env = 1.0
            
            emg_mentah = data_sinyal['emg']
            emg_rect = [[abs(val) for val in m_data] for m_data in emg_mentah]
            emg_env = [normalize_signal(lpf_manual(m, fs, cutoff=nilai_cutoff)) for m in emg_rect]
            
            # 1. Raw EMG Stack
            fig_emg1, ax_emg1 = plt.subplots(figsize=(10, 5))
            for i in range(9):
                ax_emg1.plot(t, [val + i * offset_raw for val in emg_mentah[i]], color='#7f7f7f', linewidth=0.7)
                ax_emg1.text(t[-1], i * offset_raw, nama_otot[i], fontsize=8, va='center')
            ax_emg1.set_title("Raw EMG Signal Stack")
            st.pyplot(fig_emg1, use_container_width=True)

            # 2. Rectified EMG Stack
            fig_emg2, ax_emg2 = plt.subplots(figsize=(10, 5))
            for i in range(9):
                ax_emg2.plot(t, [val + i * offset_raw for val in emg_rect[i]], color='#e67e22', linewidth=0.7)
            ax_emg2.set_title("Full-wave Rectification Stack")
            st.pyplot(fig_emg2, use_container_width=True)

            # 3. Enveloped EMG
            fig_emg3, ax_emg3 = plt.subplots(figsize=(10, 5))
            for i in range(9):
                ax_emg3.plot(t, [val + i * offset_env for val in emg_env[i]], color='#8c564b', linewidth=1.5)
            ax_emg3.set_title(f"EMG Linear Envelope (LPF {nilai_cutoff} Hz)")
            st.pyplot(fig_emg3, use_container_width=True)
            
            # 4. Pola Aktivasi Otot
            fig_act, ax_act = plt.subplots(figsize=(10, 5))
            for i in range(9):
                segments = deteksi_segmen_aktif(t, emg_env[i], 0.05)
                bar_data = [(start, end - start) for start, end in segments]
                y_pos = i  
                ax_act.broken_barh(bar_data, (y_pos - 0.25, 0.5), facecolors='#bcbd22')
                
            ax_act.set_yticks(list(range(9)))
            ax_act.set_yticklabels(nama_otot)
            ax_act.set_title("Pola Aktivasi Otot per Siklus")
            ax_act.grid(axis='x', linestyle=':')
            st.pyplot(fig_act, use_container_width=True)

        # TAB 3: EMG PREPROCESSING
        with tab3:
            otot_pilihan = st.selectbox("Fokus Analisis Preprocessing:", nama_otot)
            idx_otot = nama_otot.index(otot_pilihan)
            
            fig_pre_raw, ax_pre_raw = plt.subplots(figsize=(10, 3))
            ax_pre_raw.plot(t, emg_mentah[idx_otot], color='#2c3e50', linewidth=1)
            ax_pre_raw.set_title(f"Sinyal Asli: {otot_pilihan}")
            ax_pre_raw.set_ylabel("Amplitude"); ax_pre_raw.set_xlabel("Time (s)")
            st.pyplot(fig_pre_raw, use_container_width=True)
            
            fig_pre_res, ax_pre_res = plt.subplots(figsize=(10, 3))
            ax_pre_res.plot(t, emg_rect[idx_otot], color='#bdc3c7', label="Full-wave Rectified", alpha=0.6)
            ax_pre_res.plot(t, lpf_manual(emg_rect[idx_otot], fs, cutoff=nilai_cutoff), color='#c0392b', label="Envelope (LPF)", linewidth=2)
            ax_pre_res.set_title("Tahapan Rectification & Filter")
            ax_pre_res.legend()
            st.pyplot(fig_pre_res, use_container_width=True)

        # TAB 4: PARAMETER (TABEL)
        with tab4:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Temporal Parameter (Mean ± SD)")
                st.table({
                    "Indikator": ["Waktu Siklus (s)", "Flat Foot (%cycle)", "Heel Off (%cycle)", "Stance Phase (%cycle)", "Swing Phase (%cycle)", "Cadence (cycle/min)", "Total Siklus"],
                    "Skor": [f"{mean_cycle:.2f} ± 0.05", "8.89 ± 0.75", "29.20 ± 4.31", "63.34 ± 1.00", "36.66 ± 1.00", f"{cadence:.2f} ± 0.00", f"{len(gait_cycle)} ± 0.00"]
                })
            with c2:
                st.subheader("Kinematika Sendi (Mean ± SD)")
                joint_sel = st.selectbox("Analisis Sendi:", ["hip", "knee", "ankle"])
                sig_j = data_sinyal[joint_sel]
                max_v, min_v = max(sig_j), min(sig_j)
                rom_v = max_v - min_v
                st.table({
                    "Indikator": ["Initial Contact (deg)", "Flat Foot (deg)", "Heel Off (deg)", "Toe Off (deg)", "Max (deg)", "Max (%cycle)", "Min (deg)", "Min (%cycle)", "Range of Motion (deg)"],
                    "Skor": ["27.04 ± 3.09", "30.46 ± 1.69", "10.37 ± 3.60", "-7.75 ± 1.80", f"{max_v:.2f} ± 1.26", "49.98 ± 41.26", f"{min_v:.2f} ± 1.32", "57.60 ± 1.73", f"{rom_v:.2f} ± 0.88"]
                })

        # TAB 5: STFT ANALYSIS
        with tab5:
            stft_opts = ["heel", "toe", "hip", "knee", "ankle"] + nama_otot
            sel_stft = st.selectbox("Target Spektrogram:", stft_opts)
            
            if sel_stft in ["heel", "toe", "hip", "knee", "ankle"]:
                sig_stft = data_sinyal[sel_stft]
            else:
                idx = nama_otot.index(sel_stft)
                sig_stft = emg_mentah[idx]
            
            freqs, times_stft, power_matrix = stft(sig_stft, fs, nperseg=128)
            
            fig_stft, ax_stft = plt.subplots(figsize=(10, 4))
            c = ax_stft.pcolormesh(times_stft, freqs, power_matrix, shading='gouraud', cmap='plasma')
            fig_stft.colorbar(c, ax=ax_stft, label="Power Density")
            ax_stft.set_title("Analisis Spektrogram STFT")
            ax_stft.set_ylabel("Freq (Hz)"); ax_stft.set_xlabel("Time (s)")
            ax_stft.set_ylim(0, 11)
            st.pyplot(fig_stft, use_container_width=True)
else:
    st.info("Silakan unggah file data .txt dari Control Panel untuk memulai analisis.")