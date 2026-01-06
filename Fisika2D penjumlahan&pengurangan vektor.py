import numpy as np
import matplotlib.pyplot as plt

# ==============================
# DEFINISI VEKTOR
# ==============================
A = np.array([4, 2])   # vektor A
B = np.array([1, 3])   # vektor B

# ==============================
# OPERASI VEKTOR
# ==============================
R_add = A + B        # penjumlahan
R_sub = A - B        # pengurangan
dot_prod = np.dot(A, B)  # dot product

print("Vektor A:", A)
print("Vektor B:", B)
print("Penjumlahan (A+B):", R_add)
print("Pengurangan (A-B):", R_sub)
print("Dot Product A·B:", dot_prod)

# ==============================
# VISUALISASI
# ==============================
plt.figure(figsize=(8,8))
plt.axhline(0, color='black', linewidth=2)
plt.axvline(0, color='black', linewidth=2)

# fungsi menggambar vektor panah
def draw_vector(v, color, label):
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy',
               scale=1, color=color, label=label)

# gambar semua vektor
draw_vector(A, 'blue', 'A')
draw_vector(B, 'green', 'B')
draw_vector(R_add, 'red', 'A + B')
draw_vector(R_sub, 'purple', 'A - B')

plt.xlim(-2, 7)
plt.ylim(-2, 7)
plt.grid(True)
plt.legend()
plt.title("Simulasi Operasi Vektor 2D")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()
