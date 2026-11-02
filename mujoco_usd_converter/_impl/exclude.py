# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from pxr import Tf, Usd, UsdPhysics

from .data import ConversionData, Tokens

__all__ = ["convert_excludes"]


def convert_excludes(data: ConversionData):
    if not data.spec.excludes:
        return

    for exclude in data.spec.excludes:
        if exclude.bodyname1 not in data.references[Tokens.PhysicsBodies] or exclude.bodyname2 not in data.references[Tokens.PhysicsBodies]:
            Tf.Warn(f"Body '{exclude.bodyname1}' or '{exclude.bodyname2}' not found for contact exclude")
            continue

        body1 = data.references[Tokens.PhysicsBodies][exclude.bodyname1].GetPrim()
        body2 = data.references[Tokens.PhysicsBodies][exclude.bodyname2].GetPrim()

        # FilteredPairsAPI targets colliders or rigid bodies, not arbitrary
        # static body frames. Expand only that MJCF body's own colliders: a
        # descendant body has separate contact-exclusion semantics.
        for source in get_exclude_targets(body1):
            for target in get_exclude_targets(body2):
                filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(source)
                filtered_pairs_api.GetFilteredPairsRel().AddTarget(target.GetPath())


def get_exclude_targets(body: Usd.Prim) -> list[Usd.Prim]:
    # Physics content consists of overrides, so include undefined children.
    colliders = [child for child in body.GetAllChildren() if child.HasAPI(UsdPhysics.CollisionAPI)]
    if not colliders:
        return []
    if body.HasAPI(UsdPhysics.RigidBodyAPI):
        return [body]
    return colliders
