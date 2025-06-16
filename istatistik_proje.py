
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Veriyi okuma
df = pd.read_csv("C:\\Users\\PC\\Desktop\\project\\df_arabica_clean.csv")
# İncelenecek sütun: Toplam kupa puanları
total_points = df["Total Cup Points"]

# Temel istatistikler
mean_val = total_points.mean()
median_val = total_points.median()
variance_val = total_points.var()
std_dev_val = total_points.std()
standard_error = std_dev_val / np.sqrt(len(total_points))

# İstatistiksel özet çıktısı
print("Descriptive Statistics for 'Total Cup Points':")
print(f"Mean: {mean_val:.4f}")
print(f"Median: {median_val:.4f}")
print(f"Variance: {variance_val:.4f}")
print(f"Standard Deviation: {std_dev_val:.4f}")
print(f"Standard Error: {standard_error:.4f}")

# Grafiklerle görselleştirme
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(total_points, bins=15, kde=True, color="skyblue", ax=axes[0])
axes[0].set_title("Histogram of Total Cup Points")
axes[0].set_xlabel("Total Cup Points")
axes[0].set_ylabel("Frequency")
axes[0].grid(True)
# Boxplot (kutu grafiği)
sns.boxplot(x=total_points, color="lightgreen", ax=axes[1])
axes[1].set_title("Boxplot of Total Cup Points")
axes[1].set_xlabel("Total Cup Points")
axes[1].grid(True)
# Grafiklerin hizasını ayarla ve göster
plt.tight_layout()
plt.show()

# %95 güven aralığı hesaplamaları (ortalama için)
confidence = 0.95
n = len(total_points)
dfreedom = n - 1
t_crit = stats.t.ppf((1 + confidence) / 2, dfreedom)
margin_of_error = t_crit * standard_error
ci_mean_lower = mean_val - margin_of_error
ci_mean_upper = mean_val + margin_of_error
# %95 güven aralığı hesaplamaları (varyans için)
alpha = 1 - confidence
chi2_lower = stats.chi2.ppf(alpha / 2, dfreedom)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, dfreedom)
ci_var_lower = (dfreedom * variance_val) / chi2_upper
ci_var_upper = (dfreedom * variance_val) / chi2_lower
# Güven aralığı sonuçlarını yazdır
print("\n95% Confidence Interval for the Mean:")
print(f"Mean CI: ({ci_mean_lower:.4f}, {ci_mean_upper:.4f})")
print("\n95% Confidence Interval for the Variance:")
print(f"Variance CI: ({ci_var_lower:.4f}, {ci_var_upper:.4f})")

# Hedef ±0.1 hata ve %90 güvenle örneklem büyüklüğü belirleme
z_score = stats.norm.ppf(0.95)
E = 0.1
required_sample_size = (z_score * std_dev_val / E) ** 2
required_sample_size = np.ceil(required_sample_size)

print("\nRequired sample size (90% confidence, ±0.1 error):", int(required_sample_size))

# Hipotez testi: Ortalama gerçekten 85 mi?
mu_0 = 85 
t_statistic, p_value = stats.ttest_1samp(total_points, mu_0)
# Sonuçları yazdır
print("\nHypothesis Test:")
print("H0: Mean = 85")
print("H1: Mean ≠ 85")
print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")
# Karar: H0 reddedilir mi?
if p_value < 0.05:
    print("Result: Reject the null hypothesis (statistically significant difference from 85).")
else:
    print("Result: Fail to reject the null hypothesis (no statistically significant difference from 85).")
    
