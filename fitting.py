import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fitter import Fitter
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# Kamus untuk nama distribusi yang ramah pengguna
distribution_names = {
    'norm': 'Distribusi Normal',
    'lognorm': 'Distribusi Lognormal',
    'gamma': 'Distribusi Gamma',
    'weibull_min': 'Distribusi Weibull',
    'pareto': 'Distribusi Pareto',
    'expon': 'Distribusi Eksponensial',
    'beta': 'Distribusi Beta',
    'cauchy': 'Distribusi Cauchy',
    'logistic': 'Distribusi Logistik',
    'gumbel_r': 'Distribusi Gumbel Kanan',
    'poisson': 'Distribusi Poisson',
    'nbinom': 'Distribusi Binomial Negatif',
    'geom': 'Distribusi Geometrik',
    'binom': 'Distribusi Binomial',
    'hypergeom': 'Distribusi Hipergeometrik',
    'logser': 'Distribusi Log-Series',
    'planck': 'Distribusi Planck',
    'dlaplace': 'Distribusi Laplace Diskrit',
    'bernoulli': 'Distribusi Bernoulli',
    'skellam': 'Distribusi Skellam'
}

# Daftar semua distribusi (20 distribusi)
distributions = [
    'norm', 'lognorm', 'gamma', 'weibull_min', 'pareto',
    'expon', 'beta', 'cauchy', 'logistic', 'gumbel_r',
    'poisson', 'nbinom', 'geom', 'binom', 'hypergeom',
    'logser', 'planck', 'dlaplace', 'bernoulli', 'skellam'
]

# Fungsi untuk menghitung metrik
def calculate_metrics(data, dist_name, params):
    dist = getattr(stats, dist_name)
    
    # RMSE
    fitted_data = dist.rvs(*params, size=len(data))
    rmse = np.sqrt(np.mean((data - fitted_data) ** 2))
    
    # Log-Likelihood
    log_likelihood = np.sum(dist.logpdf(data, *params))
    
    # AIC dan BIC
    k = len(params)  # Jumlah parameter
    n = len(data)    # Jumlah data
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    
    # Kolmogorov-Smirnov (KS) Statistic
    ks_stat, _ = stats.ks_2samp(data, fitted_data)
    
    return {
        'RMSE': rmse,
        'Log-Likelihood': log_likelihood,
        'AIC': aic,
        'BIC': bic,
        'KS': ks_stat
    }

# Fungsi untuk membaca file dengan caching
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

# Fungsi untuk fitting distribusi dengan caching
@st.cache_data
def fit_distributions(data, distributions, _timeout=60):
    f = Fitter(data, distributions=distributions, timeout=_timeout)
    f.fit()
    return f

# Judul aplikasi
st.title("Fitting Distribusi dengan Metrik Evaluasi")
st.write("Unggah file CSV atau Excel dan pilih kolom untuk fitting distribusi.")

# Upload file
uploaded_file = st.file_uploader("Unggah file data (CSV atau Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Membaca file dengan caching
    try:
        df = load_data(uploaded_file)
        
        st.write("Pratinjau Data:")
        st.dataframe(df.head())

        # Pilih kolom
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            selected_column = st.selectbox("Pilih kolom untuk fitting distribusi", numeric_columns)
            
            # Ambil data dari kolom yang dipilih
            data = df[selected_column].dropna().values

            if len(data) > 0:
                # Menampilkan mean dan standard deviation
                st.subheader("Statistik Data")
                st.write(f"**Rata-rata (Mean):** {np.mean(data):.4f}")
                st.write(f"**Standar Deviasi (Std Dev):** {np.std(data):.4f}")

                # Visualisasi histogram data
                st.subheader("Histogram Data")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data, kde=True, stat="density", bins=150, ax=ax)
                ax.set_title(f'Histogram {selected_column}')
                ax.set_xlabel('Nilai')
                ax.set_ylabel('Densitas')
                st.pyplot(fig)

                # Fitting distribusi dengan caching
                st.subheader("Proses Fitting Distribusi")
                with st.spinner("Sedang melakukan fitting distribusi..."):
                    f = fit_distributions(data, distributions, _timeout=60)

                # Ringkasan 5 distribusi terbaik berdasarkan RMSE
                st.subheader("Ringkasan 5 Distribusi Terbaik (Berdasarkan RMSE)")
                # Menghitung metrik untuk setiap distribusi
                metrics_scores = {}
                for dist_name in distributions:
                    if dist_name in f.fitted_param:
                        params = f.fitted_param[dist_name]
                        metrics = calculate_metrics(data, dist_name, params)
                        metrics_scores[dist_name] = metrics
                
                # Urutkan berdasarkan RMSE
                sorted_distributions = sorted(metrics_scores.items(), key=lambda x: x[1]['RMSE'])[:5]
                summary_df = pd.DataFrame({
                    'Distribusi': [distribution_names.get(dist, dist) for dist, _ in sorted_distributions],
                    'RMSE': [metrics['RMSE'] for _, metrics in sorted_distributions],
                    'Log-Likelihood': [metrics['Log-Likelihood'] for _, metrics in sorted_distributions],
                    'AIC': [metrics['AIC'] for _, metrics in sorted_distributions],
                    'BIC': [metrics['BIC'] for _, metrics in sorted_distributions],
                    'KS': [metrics['KS'] for _, metrics in sorted_distributions],
                    'Parameter': [f.fitted_param.get(dist, {}) for dist, _ in sorted_distributions]
                })
                st.dataframe(summary_df)

                # Visualisasi 3 distribusi terbaik
                st.subheader("Plot Distribusi Terbaik")
                plt.figure(figsize=(10, 6))
                f.plot_pdf(Nbest=3)
                plt.title('Fitting Distribusi Terbaik')
                plt.xlabel('Nilai')
                plt.ylabel('Densitas')
                st.pyplot(plt.gcf())

                # Distribusi terbaik berdasarkan RMSE
                st.subheader("Distribusi Terbaik (Berdasarkan RMSE)")
                best_dist_name, best_metrics = sorted_distributions[0]
                best_params = f.fitted_param.get(best_dist_name, {})
                friendly_name = distribution_names.get(best_dist_name, best_dist_name)
                st.write(f"**Distribusi:** {friendly_name}")
                st.write(f"**RMSE:** {best_metrics['RMSE']:.4f}")
                st.write(f"**Log-Likelihood:** {best_metrics['Log-Likelihood']:.4f}")
                st.write(f"**AIC:** {best_metrics['AIC']:.4f}")
                st.write(f"**BIC:** {best_metrics['BIC']:.4f}")
                st.write(f"**KS Statistic:** {best_metrics['KS']:.4f}")
                st.write(f"**Parameter:** {best_params}")

            else:
                st.error("Kolom yang dipilih tidak memiliki data yang valid.")
        else:
            st.error("Tidak ada kolom numerik dalam file yang diunggah.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {str(e)}")
else:
    st.info("Silakan unggah file CSV atau Excel untuk memulai.")
