import numpy as np
from torch import tensor

test_dict = {'nose': tensor([258.5221, 67.3700, 1.8808]),
             'left_eye': tensor([265.3453, 61.5286, 2.1782]),
             'right_eye': tensor([249.7493, 61.5286, 1.8303]),
             'left_ear': tensor([271.1937, 70.2907, 0.8773]),
             'right_ear': tensor([236.1029, 71.2643, 1.5248]),
             'left_shoulder': tensor([284.8402, 117.5089, 0.3634]),
             'right_shoulder': tensor([215.6333, 116.5354, 0.3192]),
             'left_elbow': tensor([326.2668, 161.3196, 0.6710]),
             'right_elbow': tensor([166.8960, 159.8593, 0.8279]),
             'left_wrist': tensor([302.8730, 194.4211, 0.5470]),
             'right_wrist': tensor([205.8858, 199.2889, 0.8823]),
             'left_hip': tensor([2.7558e+02, 2.0659e+02, 1.6079e-01]),
             'right_hip': tensor([2.3757e+02, 2.1146e+02, 1.3720e-01]),
             'left_knee': tensor([292.1508, 210.9718, 0.3591]),
             'right_knee': tensor([229.2797, 316.1175, 0.4900]),
             'left_ankle': tensor([294.5876, 311.2496, 0.4028]),
             'right_ankle': tensor([2.0686e+02, 2.9080e+02, 2.7917e-01])}


def convert_dict(tensor_dict):
    """
    returns converted dict
    """
    converted_dict = {}
    for keypoint_name, keypoint_coords in tensor_dict.items():
        converted_item = {keypoint_name: [coord.flatten().tolist()[0] for coord in keypoint_coords]}
        converted_dict.update(converted_item)
    return converted_dict


def get_angle(p1, p2, p3):
    """
    returns angle for tensor points
    """
    v1 = np.asarray(p1) - np.asarray(p2)
    v2 = np.asarray(p3) - np.asarray(p2)

    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def get_central_points(keypoint_dict):
    # calculate central points
    points_to_calculate_centers = [('left_hip', 'right_hip'), ('left_shoulder', 'right_shoulder')]
    points_to_connect_names = ['hip_center', 'neck_center', 'nose']
    points_to_connect = []

    for p in points_to_calculate_centers:
        x1, y1, z1 = keypoint_dict[p[0]]
        x2, y2, z2 = keypoint_dict[p[1]]
        x = float((x1 + x2) / 2)
        y = float((y1 + y2) / 2)
        z = float((z1 + z2) / 2)
        points_to_connect.append((x, y, z))
    x, y, z = keypoint_dict['nose']
    points_to_connect.append((int(x), int(y), float(z)))
    points_to_connect = {x[0]: list(x[1]) for x in zip(points_to_connect_names, points_to_connect)}

    return points_to_connect


def get_text_coords(keypoint_coords, scale):
    # (x - 18, y + 8) for 3
    # (x - 10, y + 5) for 1
    x, y = keypoint_coords
    x, y = x - int(8 * scale), y + int(4 * scale)
    return x, y


def get_updated_keypoint_dict(keypoint_dict):
    keypoint_dict_ = keypoint_dict.copy()
    points_to_connect = get_central_points(keypoint_dict)
    keypoint_dict_.update(points_to_connect)
    return keypoint_dict_


def draw_angles(keypoint_dict, side=None):
    # calculate central points
    keypoint_dict_ = get_updated_keypoint_dict(keypoint_dict)  # keypoint_dict.copy()
    # points_to_connect = get_central_points(keypoint_dict)
    # keypoint_dict_.update(points_to_connect)
    points_to_use = {
        'left_elbow': ['left_shoulder', 'left_elbow', 'left_wrist'],
        'right_elbow': ['right_shoulder', 'right_elbow', 'right_wrist'],
        'left_knee': ['left_hip', 'left_knee', 'left_ankle'],
        'right_knee': ['right_hip', 'right_knee', 'right_ankle'],
        'left_shoulder': ['neck_center', 'left_shoulder', 'left_elbow'],
        'right_shoulder': ['neck_center', 'right_shoulder', 'right_elbow'],
        'left_hip_center': ['neck_center', 'hip_center', 'left_hip'],
        'right_hip_center': ['neck_center', 'hip_center', 'right_hip'],
        'left_neck_center': ['hip_center', 'neck_center', 'right_shoulder'],
        'right_neck_center': ['hip_center', 'neck_center', 'right_shoulder'],
        'left_hip': ['hip_center', 'left_hip', 'left_knee'],
        'right_hip': ['hip_center', 'right_hip', 'right_knee']
    }
    visible_left = ['left_elbow', 'left_knee', 'left_shoulder', 'left_hip_center', 'left_neck_center', 'left_hip']
    visible_right = ['right_elbow', 'right_knee', 'right_shoulder', 'right_hip_center', 'right_neck_center',
                     'right_hip']

    angles_dict = {}
    for name, i in points_to_use.items():
        # set visibility for each side
        if side == "L":
            visible = visible_left
        elif side == "R":
            visible = visible_right
        else:
            visible = points_to_use.keys()

        if name in visible:
            a, b, c = keypoint_dict_[i[0]], keypoint_dict_[i[1]], keypoint_dict_[i[2]]
            angle = get_angle(a, b, c)
            text_coords = keypoint_dict_[i[1]]
            angles_dict.update({name: [angle, text_coords]})

    return angles_dict


def define_symmetry(angles_dict, allowed_diff=15):
    n1 = len([key for key in angles_dict.keys() if "right" in key])
    n2 = len([key for key in angles_dict.keys() if "left" in key])
    assert n1 == n2
    angles_to_analyze = ['_elbow', '_knee', '_shoulder', '_hip_center', '_neck_center', '_hip']
    is_wrong = {}
    for angle in angles_to_analyze:
        angle1 = angles_dict['right' + angle][0]
        angle2 = angles_dict['left' + angle][0]
        diff = angle1 - angle2
        # print(angle, angle1, angle2)
        if abs(diff) > allowed_diff:
            is_wrong.update({angle: diff})
    return is_wrong


def check_squats(keypoint_dict_updated, allowed_diff=5):
    buttock = keypoint_dict_updated["hip_center"]
    x1, y1, z1 = keypoint_dict_updated["left_knee"]
    x2, y2, z2 = keypoint_dict_updated["right_knee"]
    x = float((x1 + x2) / 2)
    y = float((y1 + y2) / 2)
    z = float((z1 + z2) / 2)
    knee = (x, y, z)
    is_wrong = (buttock[1] - knee[1]) > allowed_diff
    return is_wrong, (buttock, knee)


if __name__ == "__main__":

    converted = convert_dict(test_dict)
    # print(draw_angles(converted_dict))
    new_dict = get_updated_keypoint_dict(converted)
    for name, data in new_dict.items():
        print(name, data)
