import cv2
import os, sys
import numpy as np
import ctypes
from ctypes.util import find_library
from ctypes import Structure
from os.path import join, isdir, islink

class DBusError(Structure):
    _fields_ = [("name", ctypes.c_char_p),
                ("message", ctypes.c_char_p),
                ("dummy1", ctypes.c_int),
                ("dummy2", ctypes.c_int),
                ("dummy3", ctypes.c_int),
                ("dummy4", ctypes.c_int),
                ("dummy5", ctypes.c_int),
                ("padding1", ctypes.c_void_p),]


class HardwareUuid(object):
    def __init__(self, dbus_error=DBusError):
        self._hal = ctypes.cdll.LoadLibrary(find_library('hal'))
        self._ctx = self._hal.libhal_ctx_new()
        self._dbus_error = dbus_error()
        self._hal.dbus_error_init(ctypes.byref(self._dbus_error))
        self._conn = self._hal.dbus_bus_get(ctypes.c_int(1),
                                            ctypes.byref(self._dbus_error))
        self._hal.libhal_ctx_set_dbus_connection(self._ctx, self._conn)
        self._uuid_ = None

    def __call__(self):
        return self._uuid

    @property
    def _uuid(self):
        if not self._uuid_:
            udi = ctypes.c_char_p("/org/freedesktop/Hal/devices/computer")
            key = ctypes.c_char_p("system.hardware.uuid")
            self._hal.libhal_device_get_property_string.restype = \
                                                            ctypes.c_char_p
            self._uuid_ = self._hal.libhal_device_get_property_string(
                                self._ctx, udi, key, self._dbus_error)
        return self._uuid_


def getMachine_addr():
    # wmic bios get serialnumber#Windows
    # hal-get-property --udi /org/freedesktop/Hal/devices/computer --key system.hardware.uuid#Linux
    # ioreg -l | grep IOPlatformSerialNumber#Mac OS X
    os_type = sys.platform.lower()
    if "win" in os_type:
        command = "wmic bios get serialnumber"
    elif "linux" in os_type:
        command = "hal-get-property --udi /org/freedesktop/Hal/devices/computer --key system.hardware.uuid"
    elif "darwin" in os_type:
        command = "ioreg -l | grep IOPlatformSerialNumber"
    return os.popen(command).read().replace("\n", "").replace("	", "").replace(" ", "")


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    if (cv_img is not None) and len(cv_img.shape) > 2 and cv_img.shape[2] == 4:
        # convert the image from RGBA2RGB
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)
    return cv_img


def cv_imwrite(file_path, img):
    cv2.imencode(file_ext(file_path), img)[1].tofile(file_path)

# file process
file_ext = lambda x: os.path.splitext(x)[-1]


def write_list_to_txt(list, txt_path):
    with open(txt_path, 'w') as f:
        f.write(''.join(list))


def make_dirs(*dir_list):
    for dir_str in dir_list:
        isExists = os.path.exists(dir_str)
        if not isExists:
            os.makedirs(dir_str)

# contour process
def make_mask_by_contours(shape, dtype, cnts ):
    mask = np.zeros(shape, dtype)
    mask = cv2.fillPoly(mask, cnts, [255]*shape[2])
    return mask

def get_largest_contour(cnts):
    area_list = []
    if len(cnts) == 0:
        cnt = []
    else:
        for cnt in cnts:
            area_list.append(cv2.contourArea(cnt))
        cnt = cnts[np.argmax(area_list)]
    return cnt

def del_small_contour(cnts):
    num_cnt = []
    for cnt in cnts:
        if cv2.contourArea(cnt)>10:
            num_cnt.append(cnt)
    return num_cnt

def extract_contour_rect(img_org, cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    img = img_org[y:y+h, x:x+w]
    return img

def extract_all_contour_rect(pr_mask_org,img_org):
    x, y, w, h = cv2.boundingRect(np.where(pr_mask_org>0,pr_mask_org,0))

    img_region = pr_mask_org[y:y+h, x:x+w]
    img = img_org[y:y+h, x:x+w]

    return img_region,img


# get the directories in the path
def get_dirs(path, with_sub_dir=False):
    dir_list = []
    for name in os.listdir(path):
        cur_item = join(path, name)
        if isdir(cur_item):
            dir_list.append(cur_item)
            if with_sub_dir:
                dir_list.append(get_dirs(cur_item))
    return dir_list

if __name__ == "__main__":
    pass
