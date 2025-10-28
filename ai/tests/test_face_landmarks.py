import pytest

from ai.data_preprocessing.face_landmarks import FaceLandmarks


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (10, True),
        (284, True),
        (389, True),
        (361, True),
        (400, True),
        (152, True),
        (172, True),
        (93, True),
        (234, True),
        (21, True),
        (16, False),
        (44, False),
        (33, False),
        (81, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_silhouette(landmark, expected):
    assert (landmark in FaceLandmarks.silhouette()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (61, True),
        (185, True),
        (39, True),
        (0, True),
        (269, True),
        (270, True),
        (409, True),
        (291, True),
        (40, True),
        (37, True),
        (16, False),
        (44, False),
        (33, False),
        (81, False),
        (421, False),
        (-20, False),
        (260, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_lips_upper_outer(landmark, expected):
    assert (landmark in FaceLandmarks.lips_upper_outer()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (146, True),
        (91, True),
        (181, True),
        (84, True),
        (17, True),
        (314, True),
        (405, True),
        (321, True),
        (375, True),
        (291, True),
        (16, False),
        (44, False),
        (33, False),
        (81, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_lips_lower_outer(landmark, expected):
    assert (landmark in FaceLandmarks.lips_lower_outer()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (78, True),
        (191, True),
        (80, True),
        (81, True),
        (82, True),
        (13, True),
        (312, True),
        (311, True),
        (310, True),
        (415, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_lips_upper_inner(landmark, expected):
    assert (landmark in FaceLandmarks.lips_upper_inner()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (78, True),
        (95, True),
        (88, True),
        (178, True),
        (87, True),
        (14, True),
        (317, True),
        (402, True),
        (318, True),
        (324, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_lips_lower_inner(landmark, expected):
    assert (landmark in FaceLandmarks.lips_lower_inner()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (246, True),
        (161, True),
        (160, True),
        (159, True),
        (158, True),
        (157, True),
        (173, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_right_eye_upper_0(landmark, expected):
    assert (landmark in FaceLandmarks.right_eye_upper_0()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (33, True),
        (7, True),
        (163, True),
        (144, True),
        (145, True),
        (153, True),
        (154, True),
        (155, True),
        (133, True),
        (16, False),
        (44, False),
        (633, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_right_eye_lower_0(landmark, expected):
    assert (landmark in FaceLandmarks.right_eye_lower_0()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (247, True),
        (30, True),
        (29, True),
        (27, True),
        (28, True),
        (56, True),
        (190, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_right_eye_upper_1(landmark, expected):
    assert (landmark in FaceLandmarks.right_eye_upper_1()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (130, True),
        (25, True),
        (110, True),
        (24, True),
        (23, True),
        (22, True),
        (26, True),
        (12, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_right_eye_lower_1(landmark, expected):
    assert (landmark in FaceLandmarks.right_eye_lower_1()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (113, True),
        (225, True),
        (224, True),
        (223, True),
        (222, True),
        (221, True),
        (189, True),
        (12, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_right_eye_upper_2(landmark, expected):
    assert (landmark in FaceLandmarks.right_eye_upper_2()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (226, True),
        (31, True),
        (228, True),
        (229, True),
        (230, True),
        (231, True),
        (232, True),
        (12, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_right_eye_lower_2(landmark, expected):
    assert (landmark in FaceLandmarks.right_eye_lower_2()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (143, True),
        (111, True),
        (117, True),
        (118, True),
        (119, True),
        (120, True),
        (121, True),
        (12, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_right_eye_lower_3(landmark, expected):
    assert (landmark in FaceLandmarks.right_eye_lower_3()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (156, True),
        (70, True),
        (63, True),
        (105, True),
        (66, True),
        (107, True),
        (55, True),
        (193.5, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_right_eyebrow_upper(landmark, expected):
    assert (landmark in FaceLandmarks.right_eyebrow_upper()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (35, True),
        (124, True),
        (46, True),
        (53, True),
        (52, True),
        (65, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_right_eyebrow_lower(landmark, expected):
    assert (landmark in FaceLandmarks.right_eyebrow_lower()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (473, True),
        (474, True),
        (475, True),
        (476, True),
        (477, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_right_eye_iris(landmark, expected):
    assert (landmark in FaceLandmarks.right_eye_iris()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (466, True),
        (388, True),
        (387, True),
        (386, True),
        (385, True),
        (384, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_left_eye_upper_0(landmark, expected):
    assert (landmark in FaceLandmarks.left_eye_upper_0()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (263, True),
        (249, True),
        (390, True),
        (373, True),
        (374, True),
        (380, True),
        (381, True),
        (382, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_left_eye_lower_0(landmark, expected):
    assert (landmark in FaceLandmarks.left_eye_lower_0()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (467, True),
        (260, True),
        (259, True),
        (257, True),
        (258, True),
        (286, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_left_eye_upper_1(landmark, expected):
    assert (landmark in FaceLandmarks.left_eye_upper_1()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (359, True),
        (255, True),
        (339, True),
        (254, True),
        (253, True),
        (252, True),
        (256, True),
        (341, True),
        (463, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_left_eye_lower_1(landmark, expected):
    assert (landmark in FaceLandmarks.left_eye_lower_1()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (342, True),
        (445, True),
        (444, True),
        (443, True),
        (442, True),
        (441, True),
        (413, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_left_eye_upper_2(landmark, expected):
    assert (landmark in FaceLandmarks.left_eye_upper_2()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (446, True),
        (261, True),
        (448, True),
        (449, True),
        (450, True),
        (451, True),
        (452, True),
        (453, True),
        (464, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_left_eye_lower_2(landmark, expected):
    assert (landmark in FaceLandmarks.left_eye_lower_2()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (372, True),
        (340, True),
        (346, True),
        (347, True),
        (348, True),
        (349, True),
        (350, True),
        (357, True),
        (465, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_left_eye_lower_3(landmark, expected):
    assert (landmark in FaceLandmarks.left_eye_lower_3()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (383, True),
        (300, True),
        (293, True),
        (334, True),
        (296, True),
        (336, True),
        (285, True),
        (417, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_left_eyebrow_upper(landmark, expected):
    assert (landmark in FaceLandmarks.left_eyebrow_upper()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (265, True),
        (353, True),
        (276, True),
        (283, True),
        (282, True),
        (295, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_left_eyebrow_lower(landmark, expected):
    assert (landmark in FaceLandmarks.left_eyebrow_lower()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (468, True),
        (469, True),
        (470, True),
        (471, True),
        (472, True),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_left_eye_iris(landmark, expected):
    assert (landmark in FaceLandmarks.left_eye_iris()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (168, True),
        (469, False),
        (470, False),
        (471, False),
        (472, False),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_midway_between_eyes(landmark, expected):
    assert (landmark in FaceLandmarks.midway_between_eyes()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (1, True),
        (469, False),
        (470, False),
        (471, False),
        (472, False),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_nose_tip(landmark, expected):
    assert (landmark in FaceLandmarks.nose_tip()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (2, True),
        (469, False),
        (470, False),
        (471, False),
        (472, False),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_nose_bottom(landmark, expected):
    assert (landmark in FaceLandmarks.nose_bottom()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (98, True),
        (469, False),
        (470, False),
        (471, False),
        (472, False),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_nose_right_corner(landmark, expected):
    assert (landmark in FaceLandmarks.nose_right_corner()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (327, True),
        (469, False),
        (470, False),
        (471, False),
        (472, False),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_nose_left_corner(landmark, expected):
    assert (landmark in FaceLandmarks.nose_left_corner()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (205, True),
        (469, False),
        (470, False),
        (471, False),
        (472, False),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_right_cheek(landmark, expected):
    assert (landmark in FaceLandmarks.right_cheek()) == expected


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (425, True),
        (469, False),
        (470, False),
        (471, False),
        (472, False),
        (16, False),
        (44, False),
        (33, False),
        (181, False),
        (421, False),
        (-20, False),
        (0, False),
        (5, False),
        (-234, False),
        (-21, False),
    ],
)
def test_facelandmarks_left_cheek(landmark, expected):
    assert (landmark in FaceLandmarks.left_cheek()) == expected
