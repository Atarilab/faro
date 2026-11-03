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
                env_ids=[scene.face_id('box1', 'left')],
                contact_types=[CONTACT_TYPE_UNILATERAL],
            ),
            ("ee", "robot", "right_ee"): AllowedEE(
                env_ids=[scene.face_id('box1', 'right')],
                contact_types=[CONTACT_TYPE_UNILATERAL],
            ),
        },
        obj={
            ("obj", "box1"): AllowedObj(
                self_to_env_ids={
                    6: [scene.face_id('floor', 'surface'), scene.face_id('table1', 'top')],
                },
                contact_types=[CONTACT_TYPE_UNILATERAL],
            ),
        },
    )

DOMAINS = {'default': default}
