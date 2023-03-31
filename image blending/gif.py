import cv2
import imageio
#url = "https://media.giphy.com/media/gH2bKIakvLuW4/giphy.gif"
#url="https://gfycat.com/milkycoldindianhare"
url="https://24.media.tumblr.com/tumblr_mchibyGJyL1qlu8tno1_400.gif"
frames = imageio.mimread(imageio.core.urlopen(url).read(), '.gif')
fg = cv2.imread('hi.jpg')
mask = cv2.imread('mask.png', cv2.IMREAD_UNCHANGED)
fg = cv2.resize(fg, (225, 300))
mask = cv2.resize(mask, (225, 300))


fg_h, fg_w, fg_c = fg.shape
bg_h, bg_w, bg_c = frames[0].shape
top = int((bg_h-fg_h)/2)
left = int((bg_w-fg_w)/2)
bgs = [frame[top: top + fg_h, left:left + fg_w, 0:3] for frame in frames]

results = []
alpha = 0.4
for i in range(len(bgs)):
    result = fg.copy()
    result[mask[:,:,3] != 0] = alpha * result[mask[:,:,3] != 0]
    bgs[i][mask[:,:,3] == 0] = 0
    bgs[i][mask[:,:,3] != 0] = (1-alpha)*bgs[i][mask[:,:,3] != 0]
    result = result + bgs[i]
    results.append(result)

imageio.mimsave('result.gif', results)