from filterpy.kalman import FixedLagSmoother
import numpy as np
import matplotlib.pyplot as plt


class Smoother:
    def __init__(self, r):
        # Setting kalman parameters
        self.fls = FixedLagSmoother(dim_x=2, dim_z=1, N=10)

        self.fls.x = np.array([0., .5])
        self.fls.F = np.array([[1., 1.], [0., 1.]])

        self.fls.H = np.array([[1., 0.]])
        self.fls.P *= 200
        self.fls.R *= r
        self.fls.Q *= 0.001

    def smooth_value(self, value):
        self.fls.smooth(value)
        smooth_values = np.array(self.fls.xSmooth)[:, 0]

        return smooth_values[-1]

    @staticmethod
    def get_smooth_parameters(vals):
        t = np.arange(len(vals))
        m = np.array(vals)
        p = np.poly1d(np.polyfit(t, m, 7))

        p_vals = p(t)
        error = m - p_vals
        error_mean = np.mean(error)
        error_0 = error - error_mean
        variance = np.var(error_0)

        # M(t) AND P(t)
        """plt.figure()
        plt.subplot(211)
        plt.plot(t, m, color='blue', marker='o', label='M(t)')
        plt.plot(t, p_vals, color='red', marker='o', label='P(t)')
        plt.title('M(t) AND P(t)')
        plt.xlabel('Frames')
        plt.ylabel('Distance')
        plt.legend()

        # Error = M(t) - P(t)
        plt.subplot(212)
        plt.plot(t, error, color='blue', marker='o', label='Error')
        plt.plot(t, error_0, color='red', marker='o', label='Error (mean 0)')
        plt.title('M(t) - P(t)')
        plt.xlabel('Frames')
        plt.ylabel('Error')
        plt.legend()

        plt.show()"""

        return variance
