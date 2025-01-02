import cv2
import numpy as np

class Extractor:
  def __init__(self):
    self.orb = cv2.ORB().create()
    print(self.orb)
    self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    self.last = None

  def extract_feature(self, img):
    # Detect
    grey_img = np.mean(img, axis=2).astype(np.uint8) # goodFeaturesToTrack required greyscale image
    pts = cv2.goodFeaturesToTrack(grey_img, 3000, qualityLevel=0.01, minDistance=7)

    # Extract
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]

    # Compute
    kps, des = self.orb.compute(img, kps)

    # Matches
    if self.last is not None:
      matches = self.bf.knnMatch(des, self.last['des'], k=2)

      # Lowe's ratio test
      pts1, pts2 = [], []
      for m,n in matches:
        if m.distance < 0.75*n.distance:
          pts1.append(kps[m.queryIdx].pt)
          pts2.append(self.last['kps'][m.trainIdx].pt)

      # Filters
      pts1 = np.int32(pts1)
      pts2 = np.int32(pts2)
      F, mask = cv2.findEssentialMat(pts1, pts2)
      pts1 = pts1[mask.ravel()==1]
      pts2 = pts2[mask.ravel()==1]
      ret = zip(pts1, pts2)
    else:
      ret = None

    # Save last frame stuff
    self.last = {"img": img, "kps": kps, "des": des}
    return ret
