# vim: expandtab:ts=4:sw=4
import torch
import scipy.linalg
from typing import Tuple
"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

@torch.jit.script
class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self, device: str):
        ndim, dt = 4, 1.
        self._device = device

        # Create Kalman filter model matrices.
        self._motion_mat = torch.eye(2 * ndim, 2 * ndim, device=self._device, dtype=torch.float32)

        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # (8, 4)
        self._update_mat = torch.eye(2 * ndim, ndim, device=self._device, dtype=torch.float32)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        #JR: make it faster by creating just once
        self.std_pos = torch.empty([4], device=self._device)
        self.std_vel = torch.empty([4], device=self._device)

        self.std=torch.empty([4], device=self._device)

    def initiate(self, measurement: torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:

        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = torch.zeros_like(mean_pos, device=measurement.device)
        mean = torch.cat([mean_pos, mean_vel], dim=-1).view(1, -1)  # (1, 8)

        std = torch.tensor([[
            2 * self._std_weight_position * measurement[0],  # the center point x
            2 * self._std_weight_position * measurement[1],  # the center point y
            1 * measurement[2],  # the ratio of width/height
            2 * self._std_weight_position * measurement[3],  # the height
            10 * self._std_weight_velocity * measurement[0],
            10 * self._std_weight_velocity * measurement[1],
            0.1 * measurement[2],
            10 * self._std_weight_velocity * measurement[3]]],
            device=measurement.device)

        covariance = torch.diag_embed(torch.pow(std, 2))  # (1, 8, 8)
        return mean, covariance

    def predict(self, mean: torch.Tensor, covariance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """

        #is faster if I create it once and asign it many times
        self.std_pos[0]=self._std_weight_position * mean[0][0]
        self.std_pos[1] = self._std_weight_position * mean[0][1]
        self.std_pos[2] = 1 * mean[0][2]
        self.std_pos[3] = self._std_weight_position * mean[0][3]

        self.std_vel[0] = self._std_weight_velocity * mean[0][0]
        self.std_vel[1] = self._std_weight_velocity * mean[0][1]
        self.std_vel[2] = 0.1 * mean[0][2]
        self.std_vel[3] = self._std_weight_velocity * mean[0][3]

        # std_pos = torch.tensor(  [
        #     self._std_weight_position * mean[0][0],
        #     self._std_weight_position * mean[0][1],
        #     1 * mean[0][2],
        #     self._std_weight_position * mean[0][3]],device=self._device)
        # std_vel = torch.tensor( [
        #     self._std_weight_velocity * mean[0][0],
        #     self._std_weight_velocity * mean[0][1],
        #     0.1 * mean[0][2],
        #     self._std_weight_velocity * mean[0][3]],device=self._device)

        # (*, 8, 8)
        motion_cov = torch.diag_embed(torch.pow(torch.cat([self.std_pos, self.std_vel], dim=-1), 2))
        mean = torch.matmul(mean,self._motion_mat)  # (*, 8)

        # (*, 8, 8)
        # covariance = torch.matmul(torch.matmul(covariance, self._motion_mat), self._motion_mat)
        covariance = torch.matmul(torch.matmul(covariance.permute(0, 2, 1), self._motion_mat).permute(0, 2, 1),
                                  self._motion_mat)

        return mean, covariance + motion_cov

    def project(self, mean: torch.Tensor, covariance:torch.Tensor, confidence:float=.0)->Tuple[torch.Tensor, torch.Tensor]:
        """Project state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        confidence: (dyh) 检测框置信度
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        # (*, 4)

        self.std[0]=self._std_weight_position * mean[0][0]
        self.std[1] = self._std_weight_position * mean[0][1]
        self.std[2] = 0.1 * mean[0][2]
        self.std[3] = self._std_weight_position * mean[0][3]

        # std = torch.tensor([[
        #     self._std_weight_position * mean[0][0],
        #     self._std_weight_position * mean[0][1],
        #     0.1 * mean[0][2],
        #     self._std_weight_position * mean[0][3]]],
        #     device=self._device)


        # std = [(1 - confidence) * x for x in std]

        # (*, 4, 4)
        innovation_cov = torch.diag_embed(torch.pow(self.std, 2))
        # (4, 8) dot (*, 8)
        mean = torch.mm(mean, self._update_mat)  # (*, 4)

        # (*, 4, 4)
        # covariance = torch.matmul(self._update_mat,torch.matmul(covariance, self._update_mat.T))
        covariance = torch.matmul(torch.matmul(covariance.permute(0, 2, 1), self._update_mat).permute(0, 2, 1),
                                  self._update_mat)
        return mean, covariance + innovation_cov

    def update(self, mean: torch.Tensor, covariance: torch.Tensor, measurement: torch.Tensor, confidence:float=.0)->Tuple[torch.Tensor, torch.Tensor]:
        """Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        confidence: (dyh)检测框置信度
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance,confidence)

        chol_factor = torch.linalg.cholesky(projected_cov, upper=False)


        kalman_gain = torch.cholesky_solve(torch.matmul(covariance, self._update_mat).permute(0, 2, 1), chol_factor,
                                           upper=False).permute(0, 2, 1)

        # (*, 4)
        innovation = measurement.view(-1, 4) - projected_mean

        kalman_gain_t = kalman_gain.permute(0, 2, 1)
        new_mean = mean + torch.bmm(innovation.unsqueeze(1), kalman_gain_t).view(-1, 8)  # (*, 8)
        new_covariance = covariance - torch.matmul(torch.matmul(projected_cov.permute(0, 2, 1), kalman_gain_t).permute(0, 2, 1),kalman_gain_t)

        return new_mean, new_covariance

    def gating_distance(self, mean: torch.Tensor, covariance: torch.Tensor, measurements: torch.Tensor,
                        only_position: bool)->torch.Tensor:
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance,0.0)
        if only_position:
            mean, covariance = mean[:, None, :2], covariance[:, :2, :2]
            measurements = measurements[None, :, :2]
        else:
            mean = mean.unsqueeze(1)
            measurements = measurements.unsqueeze(0)

        cholesky_factor = torch.linalg.cholesky(covariance)
        d = measurements - mean   # (n, m, 4)

        z = torch.linalg.solve_triangular(cholesky_factor, d.permute(0, 2, 1), upper=False)
        squared_maha = torch.sum(z ** 2, dim=1)  # (n, m)


        return squared_maha
