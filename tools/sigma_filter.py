import numpy as np
import pdb

"""
Implementation of Sigma filter (Lee's filter)
Reference:
    1. https://imagej.nih.gov/ij/plugins/sigma-filter.html
    2. http://kaiminghe.com/eccv10/ (for fast implementation of boxfilter)
"""

def boxfilter(imSrc, r):
    """
    Definition (MATLAB): imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
    Reference: http://kaiminghe.com/eccv10/ (for fast implementation of boxfilter)
    """

    hei, wid = imSrc.shape
    imDst = np.zeros(imSrc.shape)
    
    # cumulative sum over Y axis
    imCum = np.cumsum(imSrc, axis=0)
    # difference over Y axis
    imDst[:r+1, :] = imCum[r:2*r+1, :]
    imDst[r+1:hei-r, :] = imCum[2*r+1:hei, :] - imCum[:hei-2*r-1, :]
    imDst[hei-r:hei, :] = imCum[hei-1, :].reshape(1,-1) - imCum[hei-2*r-1:hei-r-1, :]

    # cumulative sum over X axis
    imCum = np.cumsum(imDst, axis=1)
    # difference over Y axis
    imDst[:, :r+1] = imCum[:, r:2*r+1]
    imDst[:, r+1:wid-r] = imCum[:, 2*r+1:wid] - imCum[:, :wid-2*r-1]
    imDst[:, wid-r:wid] = imCum[:, wid-1].reshape(-1,1) - imCum[:, wid-2*r-1:wid-r-1]

    return imDst


class SigmaFilter(object):
    def __init__(self, radius=2, sigma=2., min_frac=0.2):
        """
        Parameters:
        radius   --   half window size for filtering
        sigma    --   define the range of including pixels (see Sigma filter)
        min_frac --   minimum pixel fraction, average if lower than this fraction
        """

        self.radius = radius
        self.sigma = sigma
        self.min_frac = min_frac
        self.min_num = np.floor(self.min_frac * (2*radius + 1)**2)

    def filter_block(self, i, j):
        """
        Perform sigma filtering on specific pixel with position [i, j]
        """

        patch = self.I[i-self.radius:i+self.radius+1, j-self.radius:j+self.radius+1]
        valid = np.logical_and(patch>self.sigmaBottom[i, j], patch<self.sigmaTop[i, j])

        if np.sum(valid) > self.min_num:
            return np.sum(patch[valid]) / np.sum(valid)
        else:
            return np.mean(patch)


    def __call__(self, I):
        """
        I -- input image, should be [0, 1]
        """

        self.I = I
        self.hei, self.wid = I.shape

        print "Pre-computing ..."
        # calculate the range of each local patch
        self.N = boxfilter(np.ones(I.shape), self.radius)

        self.mean_I = boxfilter(I, self.radius) / self.N
        self.mean_II = boxfilter(I*I, self.radius) / self.N
        self.var_I = self.mean_II - self.mean_I * self.mean_I
        self.var_I = np.clip(self.var_I, 0, None)

        self.sigmaRange = self.sigma * np.sqrt(self.var_I + np.finfo(np.float32).eps)
        self.sigmaBottom = self.I - self.sigmaRange
        self.sigmaTop = self.I + self.sigmaRange

        print "Start filtering ..."
        self.filtered_I = np.copy(self.I)
        for i in range(self.radius, self.hei-self.radius):
            for j in range(self.radius, self.wid-self.radius):
                self.filtered_I[i, j] = self.filter_block(i, j)

        return self.filtered_I

if __name__ == "__main__":

    import PIL
    from PIL import Image

    img = Image.open('/Users/xitongyang/Downloads/coins.jpg').convert('L')
    I = np.array(img, dtype=float)
    I = I / 255.

    sigma_filter = SigmaFilter(sigma=3)
    result = sigma_filter(I)
    result *= 255

    result_img = Image.fromarray(result.astype('uint8'))
    result_img.save('/Users/xitongyang/Downloads/coins_filtered3.jpg')
