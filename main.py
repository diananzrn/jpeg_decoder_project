import numpy as np
import matplotlib.pyplot as plt
import math

# ---------------------------------------------------------
# GIVEN DATA
# ---------------------------------------------------------
rlc_sequence = [
    -26, (0, -3), (1, 3), (0, 2), (0, -6), (0, 2), (0, -4), 
    (0, 1), (0, -3), (0, 1), (0, 1), (0, 5), (0, 1), (0, 2), 
    (0, 1), (0, 2), (5, -1), (0, 1), (0, 0)
]

# >>> ACTION REQUIRED HERE <<<
# Replace this standard table with the specific 8x8 Quantization Table 
# shown in your lecture notes or the pictures you uploaded.
quantization_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

original_image = np.array([
    [52, 55, 61, 66, 70, 61, 64, 73],
    [63, 59, 55, 90, 109, 85, 69, 72],
    [62, 59, 68, 113, 144, 104, 66, 73],
    [63, 58, 71, 122, 154, 106, 70, 69],
    [67, 61, 68, 104, 126, 88, 68, 70],
    [79, 65, 60, 70, 77, 68, 58, 75],
    [85, 71, 64, 59, 55, 61, 65, 83],
    [187, 79, 69, 68, 65, 76, 78, 94]
])

# ---------------------------------------------------------
# STEP 1: Decode the 64-vector
# ---------------------------------------------------------
vector_64 = np.zeros(64)
vector_64[0] = rlc_sequence[0] # DC Value

current_idx = 1
for item in rlc_sequence[1:]:
    runlength, value = item
    if runlength == 0 and value == 0:  # End of Block
        break
    current_idx += runlength           # Skip zeros based on runlength
    vector_64[current_idx] = value
    current_idx += 1

# ---------------------------------------------------------
# STEP 2: Reconstruct using zig-zag scan
# ---------------------------------------------------------
zigzag_indices = [
    (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
    (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
    (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
    (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
    (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
    (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
    (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
    (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
]

dct_matrix = np.zeros((8, 8))
for idx, (r, c) in enumerate(zigzag_indices):
    dct_matrix[r, c] = vector_64[idx]

# ---------------------------------------------------------
# STEP 3: Multiply by Quantization Table
# ---------------------------------------------------------
dequantized_matrix = dct_matrix * quantization_table

# ---------------------------------------------------------
# STEP 4: Perform 2D IDCT
# ---------------------------------------------------------
def C(xi):
    return math.sqrt(2)/2 if xi == 0 else 1

idct_matrix = np.zeros((8, 8))

for i in range(8):      
    for j in range(8):  
        sum_val = 0
        for u in range(8):      
            for v in range(8):  
                cos_u = math.cos(((2 * i + 1) * u * math.pi) / 16)
                cos_v = math.cos(((2 * j + 1) * v * math.pi) / 16)
                F_uv = dequantized_matrix[u, v]
                
                term = (C(u) * C(v) / 4) * cos_u * cos_v * F_uv
                sum_val += term
                
        idct_matrix[i, j] = sum_val

idct_rounded = np.round(idct_matrix).astype(int)
print("--- Step 4 Checkpoint Matrix ---")
print(idct_rounded)

# ---------------------------------------------------------
# STEP 5: Add 128 (Level Shift)
# ---------------------------------------------------------
reconstructed_image = idct_rounded + 128
reconstructed_image = np.clip(reconstructed_image, 0, 255)

# ---------------------------------------------------------
# BONUS: Display and Compare
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(reconstructed_image, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Decoded Image')
axes[0].axis('off')

axes[1].imshow(original_image, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Original Image')
axes[1].axis('off')

plt.show()