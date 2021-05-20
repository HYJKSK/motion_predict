import sys
import os
from pprint import pprint
from os.path import join as pjoin

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

import argparse
import numpy as np
import scipy.ndimage.filters as filters
from load_skeleton import Skel
from Quaternions_old import Quaternions
from Pivots import Pivots
import BVH

def phase_from_ft(foot_contact, is_debug=False):
    """
    foot_contact: [T, 4] -> take joints 0, 2 as standards
    phase = left foot in contact (0) --> right foot in contact (pi) --> left foot in contact (2pi),
            in range [0, 2pi)
    """
    num_circles = 0
    circle_length = 0
    total_length = len(foot_contact)
    ft = foot_contact[:, [0, 2]].astype(np.int)
    ft_start = np.zeros((total_length, 2))
    phases = np.zeros((total_length, 1))

    """
    calculate the average "half-phase length"
    find the first and last "01" pattern
    """
    for j in range(2):
        for i in range(1, total_length):
            ft_start[i, j] = (ft[i - 1, j] == 0 and ft[i, j] == 1)
    if is_debug:
        print('ft_start,', ft_start)

    last, beg_i = -1, -1
    starts = []
    for i in range(total_length):
        if ft_start[i, 0] or ft_start[i, 1]:
            if last != -1:
                num_circles += 1
                circle_length += i - last
            else:
                beg_i = i
            last = i
            starts.append(i)

    avg_circle = 0 if num_circles == 0 else circle_length * 1.0 / num_circles
    if is_debug:
        print("%d circles, total length = %d, avg length = %.3lf" % (num_circles, circle_length, avg_circle))

    if len(starts) == 0:  # phase never changed
        return phases

    """[0, beg_i - 1]: first incomplete circle"""
    prev_pos = min(0, beg_i - avg_circle)
    prev_val = 0 if ft_start[beg_i, 1] == 1 else 1  # 0 if next step is on the right
    cir_i = 0
    next_pos = starts[cir_i]

    for i in range(total_length):
        if i == next_pos:
            prev_pos = next_pos
            prev_val = 1 - prev_val
            cir_i += 1
            if cir_i >= len(starts):
                next_pos = max(total_length + 1, next_pos + avg_circle)
            else:
                next_pos = starts[cir_i]
        phases[i] = prev_val + (i - prev_pos) * 1.0 / (next_pos - prev_pos)

    phases *= np.pi
    if is_debug:
        print('phases:', phases)
    return phases



def forward_rotations(skel, rotations, rtpos=None, trim=True): #$$
    """
    input: rotations [T, J, 4], rtpos [T, 3]
    output: positions [T, J, 3]
    """
    transforms = Quaternions(rotations).transforms()  # [..., J, 3, 3]
    glb = np.zeros(rotations.shape[:-1] + (3,))  # [T, J, 3]
    if rtpos is not None:
        glb[..., 0, :] = rtpos
    for i, pi in enumerate(skel.topology):
        if pi == -1:
            continue
        glb[..., i, :] = np.matmul(transforms[..., pi, :, :],
                                   skel.offset[i])
        glb[..., i, :] += glb[..., pi, :]
        transforms[..., i, :, :] = np.matmul(transforms[..., pi, :, :],
                                             transforms[..., i, :, :])
    if trim:
        glb = glb[..., skel.chosen_joints, :]
    return glb


def foot_contact_from_positions(positions, fid_l=(3, 4), fid_r=(7, 8)): #$$
    """
    positions: [T, J, 3], trimmed (only "chosen_joints")
    fid_l, fid_r: indices of feet joints (in "chosen_joints")
    """
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    velfactor = np.array([0.05, 0.05])
    feet_contact = []
    for fid_index in [fid_l, fid_r]:
        foot_vel = (positions[1:, fid_index] - positions[:-1, fid_index]) ** 2  # [T - 1, 2, 3]
        foot_vel = np.sum(foot_vel, axis=-1)  # [T - 1, 2]
        foot_contact = (foot_vel < velfactor).astype(np.float)
        feet_contact.append(foot_contact)
    feet_contact = np.concatenate(feet_contact, axis=-1)  # [T - 1, 4]
    feet_contact = np.concatenate((feet_contact[0:1].copy(), feet_contact), axis=0)

    return feet_contact  # [T, 4]

def across_from_glb(positions, hips=(2, 6), sdrs=(14, 18)): #$$
    """
    positions: positions [T, J, 3], trimmed (only "chosen_joints")
    hips, sdrs: left/right hip joints, left/right shoulder joints
    output: local x-axis for each frame [T, 3]
    """
    across = positions[..., hips[0], :] - positions[..., hips[1], :] #+ \
             #positions[..., sdrs[0], :] - positions[..., sdrs[1], :]  # [T, 3]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    return across


def y_rotation_from_positions(positions, hips=(2, 6), sdrs=(14, 18)): #$$
    """
    input: positions [T, J, 3]
    output: quaters: [T, 1, 4], quaternions that rotate the character around the y-axis to face [0, 0, 1]
            pivots: [T, 1] in [0, 2pi], the angle from [0, 0, 1] to the current facing direction
    """
    across = across_from_glb(positions, hips=hips, sdrs=sdrs)
    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.tile(np.array([0, 0, 1]), forward.shape[:-1] + (1, ))
    quaters = Quaternions.between(forward, target)[..., np.newaxis, :]  # [T, 4] -> [T, 1, 4]
    pivots = Pivots.from_quaternions(-quaters).ps  # from "target"[0, 0, 1] to current facing direction "forward"
    return quaters, pivots


class AnimationData:
    """
    Canonical Representation:
        Skeleton
        [T, Jo * 4 + 4 global params + 4 foot_contact]
    """
    def __init__(self, full, bodytype=0, skel=None, frametime=1/30): #$$
        # full은 motion
        if skel is None:
            skel = Skel(bodytype=bodytype)
        self.skel = skel
        self.frametime = frametime
        self.len = len(full)
        self.rotations = full[:, :-8].reshape(self.len, -1, 4)  # [T, Jo, 4]
        assert self.rotations.shape[1] == len(self.skel.topology), "Rotations do not match the skeleton."
        self.rotations /= np.sqrt(np.sum(self.rotations ** 2, axis=-1))[..., np.newaxis]
        self.rt_pos = full[:, -8:-5]  # [T, 3]
        self.rt_rot = full[:, -5:-4]  # [T, 1]
        self.foot_contact = full[:, -4:]  # [T, 4]
        self.full = np.concatenate([self.rotations.reshape(self.len, -1), self.rt_pos, self.rt_rot, self.foot_contact], axis=-1)
        self.phases = None  # [T, 1]
        self.local_x = None  # [3]
        self.positions_for_proj = None  # [T, (J - 1) + 1, 3], trimmed and not forward facing
        self.global_positions = None

    def get_original_rotations(self, rt_rot=None):
        if rt_rot is None:
            rt_rot = self.rt_rot
        yaxis_rotations = Quaternions(np.array(Pivots(rt_rot).quaternions()))
        rt_rotations = Quaternions(self.rotations[:, :1])  # [T, 1, 4]
        rt_rotations = np.array(yaxis_rotations * rt_rotations)
        rt_rotations /= np.sqrt((rt_rotations ** 2).sum(axis=-1))[..., np.newaxis]
        return np.concatenate((rt_rotations, self.rotations[:, 1:]), axis=1)  # [T, J, 4]

    def get_BVH(self, forward=True):
        rt_pos = self.rt_pos  # [T, 3]
        rt_rot = self.rt_rot  # [T, 1]
        if forward:  # choose a direction in [z+, x+, z-, x-], which is closest to "forward", as the new z+

            directions = np.array(range(4)) * np.pi * 0.5  # [0, 1, 2, 3] * 0.5pi
            diff = rt_rot[np.newaxis, :] - directions[:, np.newaxis, np.newaxis]  # [1, T, 1] - [4, 1, 1]
            diff = np.minimum(np.abs(diff), 2.0 * np.pi - np.abs(diff))
            diff = np.sum(diff, axis=(-1, -2))  # [4, T, 1] -> [4]

            new_forward = np.argmin(diff)
            rt_rot -= new_forward * np.pi * 0.5

            for d in range(new_forward):
                tmp = rt_pos[..., 0].copy()
                rt_pos[..., 0] = -rt_pos[..., 2].copy()
                rt_pos[..., 2] = tmp

        rotations = self.get_original_rotations(rt_rot=rt_rot)

        rest, names, _ = self.skel.rest_bvh
        anim = rest.copy()
        anim.positions = anim.positions.repeat(self.len, axis=0)
        anim.positions[:, 0, :] = rt_pos
        anim.rotations.qs = rotations

        return (anim, names, self.frametime)

    def get_foot_contact(self, transpose=False): #$$
        if transpose:
            return self.foot_contact.transpose(1, 0)  # [4, T]
        else:
            return self.foot_contact

    def get_content_input(self):
        rotations = self.rotations.reshape(self.len, -1)  # [T, Jo * 4]
        return np.concatenate((rotations, self.rt_pos, self.rt_rot), axis=-1).transpose(1, 0)  # [Jo * 4 + 3 + 1, T]

    def get_content_phase(self):
        rotations = self.rotations.reshape(self.len, -1)  # [T, Jo * 4]
        phase = self.get_phases()
        return np.concatenate((rotations, self.rt_pos, self.rt_rot, phase), axis=-1).transpose(1, 0)

    # for phase
    def get_phases(self):
        if self.phases is None:
            self.phases = phase_from_ft(self.foot_contact)
        return self.phases

    def get_full(self):
        return self.full

    @classmethod
    def from_rotations_and_root_positions(cls, rotations, root_positions, bodytype, skel=None, frametime=1/30): #$$
        """
        rotations: [T, J, 4]
        root_positions: [T, 3]
        """
        if skel is None:
            skel = Skel(bodytype=bodytype)

        rotations /= np.sqrt(np.sum(rotations ** 2, axis=-1))[..., np.newaxis]
        global_positions = forward_rotations(skel, rotations, root_positions, trim=True)
        foot_contact = foot_contact_from_positions(global_positions, fid_l=skel.fid_l, fid_r=skel.fid_r)
        quaters, pivots = y_rotation_from_positions(global_positions, hips=skel.hips, sdrs=skel.sdrs)

        root_rotations = Quaternions(rotations[:, 0:1, :].copy())  # [T, 1, 4]
        root_rotations = quaters * root_rotations  # facing [0, 0, 1]
        root_rotations = np.array(root_rotations).reshape((-1, 1, 4))  # [T, 1, 4]
        rotations[:, 0:1, :] = root_rotations
        full = np.concatenate([rotations.reshape((len(rotations), -1)), root_positions, pivots, foot_contact], axis=-1)
        return cls(full, bodytype, skel, frametime)

    @classmethod
    def from_network_output(cls, input, bodytype=0):
        input = input.transpose(1, 0)
        input = np.concatenate((input, np.zeros((len(input), 4))), axis=-1)
        return cls(input, bodytype=bodytype)

    @classmethod
    def from_BVH(cls, filename, bodytype, downsample=4, skel=None, trim_scale=None): #$$
        anim, names, frametime = BVH.load(filename)
        if trim_scale is not None:
            length = (len(anim) // trim_scale) * trim_scale
            anim = anim[:length]
        rotations = np.array(anim.rotations)  # [T, J, 4]
        root_positions = anim.positions[:, 0, :]
        return cls.from_rotations_and_root_positions(rotations, root_positions, bodytype=bodytype, skel=skel, frametime=frametime * downsample)

