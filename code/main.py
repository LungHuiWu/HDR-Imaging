from HDR import gsolve
import argparse
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--imgdir", help="Directory of HDR Images")
parser.add_argument("--savedir", help="Directory of Results")
parser.add_argument("--sample", type=int, default=500, help="Number of sample points")
parser.add_argument("--lbda", default=100, type=int, help="Constant for smoothness")
args = parser.parse_args()

### Read Image
imageset = np.array([])
num_image = 0
for i in sorted(os.listdir(args.imgdir)):
    if i[-4:] in ['.jpg', '.JPG', '.png']:
        # print(i, B[-1])
        num_image += 1
        img = cv2.imread(os.path.join(args.imgdir, i))
        img = np.expand_dims(img, axis=0)
        imageset = np.concatenate((imageset, img), axis=0) if imageset.size else img
imageset = imageset.astype(np.uint8)
print("Image Reading: Done")

### Sample Pixels
image_spl = np.array([])
x, y = np.random.randint(imageset.shape[1], size=args.sample), np.random.randint(imageset.shape[2], size=args.sample)

for k in range(args.sample):
    samp = np.expand_dims(imageset[:,x[k],y[k],:], axis=0)
    image_spl = np.concatenate((image_spl, samp), axis=0) if k else samp

imgB, imgG, imgR = image_spl[:,:,0], image_spl[:,:,1], image_spl[:,:,2]
print("Pixel Sampling: Done")

### Read Exposures
# B = [np.log(2**i) for i in range(-12, 5, 1)]
B = [np.log(2**i) for i in range(-12, 4, 1)]

### Generate Weights & Read Exposures
gaussian = lambda x, mu, s: 1 / (s * (2 * np.pi) ** (1/2)) * np.exp(-(x - mu) ** 2 / (2 * s ** 2))
# w = [gaussian(i,128,128) for i in range(256)]
w = [(i+1 if i<128 else 256-i) for i in range(256)]
# w = [(i+1 if i<128 else 256-i) for i in range(256)]
# print(w)s

color = ['blue', 'green', 'red']
g_B, lnE_B = gsolve(imgB, B, args.lbda, w)
g_G, lnE_G = gsolve(imgG, B, args.lbda, w)
g_R, lnE_R = gsolve(imgR, B, args.lbda, w)
lnG = [g_B, g_G, g_R]
print("Solving radiance: Done")

### Plot
fig, ax = plt.subplots(3,1)
for i in range(3):
    ax[i].plot(np.arange(256), lnG[i], color=color[i])
    ax[i].set_title(color[i])
    ax[i].set_ylabel('log exposure')
    ax[i].set_xlabel('pixel values')
fig.savefig(os.path.join(args.savedir, 'response_curve.png'), bbox_inches='tight', dpi=256)
print("Saving Response Curve: Done")

### Compute Radiance
image_shape = imageset[0].shape
ln_radiance_bgr = np.zeros(image_shape).astype(np.float32)
height, width, channels = image_shape

for c in range(channels): # BGR channels
    W_sum = np.zeros([height, width], dtype=np.float32) + 1e-8
    ln_radiance_sum = np.zeros([height, width], dtype=np.float32)

    for p in tqdm(range(num_image)): # different shutter times
        im_1D = imageset[p, :, :, c].flatten()
        ln_radiance = (lnG[c][im_1D] - B[p]).reshape(height, width)
        weights = np.array([w[j] for j in im_1D]).reshape(height, width)
        w_ln_radiance = (ln_radiance * weights)
        ln_radiance_sum += w_ln_radiance
        W_sum += weights

    weighted_ln_radiance = ln_radiance_sum / W_sum
    ln_radiance_bgr[:, :, c] = weighted_ln_radiance

hdr = np.exp(ln_radiance_bgr).astype(np.float32)
cv2.imwrite(os.path.join(args.savedir, 'recovered.hdr'), hdr)
print('Save Radiance: Done')

### Tone mapping
delta, a = 1e-8, 0.5
Lw_ave = np.mean(np.exp(np.log(delta + hdr)))
L = (a / Lw_ave) * hdr
Lm_white = np.max(L)
Ld = (L*(1+(L/(Lm_white**2)))) / (1+L)
ldr = np.clip(np.array(Ld * 255), 0, 255).astype(np.uint8)

cv2.imwrite(os.path.join(args.savedir, 'tm_ldr.png'), ldr)
print('Tone mapping: Done')

