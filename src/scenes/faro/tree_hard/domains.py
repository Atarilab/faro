from __future__ import annotations

from faro.graph.domains import AllowedEE, AllowedObj, AllowedSpec
from faro.utils.contact_type import CONTACT_TYPE_UNILATERAL


def default(scene) -> AllowedSpec:
    return AllowedSpec(
        ee={
            ("ee", "robot", "left_sole"): AllowedEE(
                env_ids=[scene.face_id('floor', 'surface')],
                contact_types=[CONTACT_TYPE_UNILATERAL],
            ),
            ("ee", "robot", "right_sole"): AllowedEE(
                env_ids=[scene.face_id('floor', 'surface')],
                contact_types=[CONTACT_TYPE_UNILATERAL],
            ),
            ("ee", "robot", "left_ee"): AllowedEE(
                env_ids=[scene.face_id('white_box', 'left'),scene.face_id('white_box', 'front'),],
                contact_types=[CONTACT_TYPE_UNILATERAL],
            ),
            ("ee", "robot", "right_ee"): AllowedEE(
                env_ids=[scene.face_id('white_box', 'right'),scene.face_id('white_box', 'back'),],
                contact_types=[CONTACT_TYPE_UNILATERAL],
            ),
        },
        obj={
            ("obj", "white_box"): AllowedObj(
                self_to_env_ids={
                    6: [scene.face_id('floor', 'surface'), scene.face_id('table_short', 'top'),],
                },
                contact_types=[CONTACT_TYPE_UNILATERAL],
            ),
        },
    )

DOMAINS = {'default': default}
